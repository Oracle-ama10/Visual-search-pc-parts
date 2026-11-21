"""FastAPI Backend — ระบบค้นหาอะไหล่คอมมือสองด้วยรูปภาพ
รัน:  uvicorn main:app --reload      เอกสาร API:  http://127.0.0.1:8000/docs"""
import io, os, time, uuid, datetime as dt
import bcrypt, jwt
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import embedder
from db import get_conn, search_parts, get_threshold

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(os.path.join(UPLOAD_DIR, "parts"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, "queries"), exist_ok=True)

app = FastAPI(title="Visual Search for Used Computer Parts")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "ไม่สามารถอ่านไฟล์ภาพได้ กรุณาอัปโหลด .jpg หรือ .png")


@app.get("/health")
def health():
    with get_conn() as c:
        n = c.execute("SELECT count(*) FROM part_embeddings").fetchone()[0]
    return {"status": "ok", "embeddings": n}


# ---------------- ลูกค้า: ค้นหาด้วยภาพ (UC-01, UC-08, UC-09, UC-10) ----------------
@app.post("/api/search")
async def search(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    data = await file.read()
    img = read_image(data)
    qpath = os.path.join("queries", f"{uuid.uuid4()}.jpg")
    img.save(os.path.join(UPLOAD_DIR, qpath))
    vec = embedder.embed_image(img)
    with get_conn() as conn:
        results = search_parts(conn, vec)
        threshold = get_threshold(conn)
        top1 = results[0]["score"] if results else 0.0
        blocked = (not results) or top1 < threshold          # Hard Block
        latency = int((time.perf_counter() - t0) * 1000)
        conn.execute(
            "INSERT INTO search_logs(query_image_path, top1_score, is_hard_blocked, result_count, latency_ms) "
            "VALUES (%s,%s,%s,%s,%s)",
            (qpath, top1, blocked, 0 if blocked else len(results), latency))
    return {
        "top1_score": top1, "threshold": threshold, "is_hard_blocked": blocked, "latency_ms": latency,
        "results": [] if blocked else results,
        "fallback_message": "ไม่พบสินค้าที่ใกล้เคียงเพียงพอ กรุณาถ่ายภาพใหม่ในมุมที่ชัดขึ้น หรือดูสินค้าขายดี" if blocked else None,
    }


@app.get("/api/parts/{part_id}")            # UC-03 ดูรายละเอียดสินค้า
def part_detail(part_id: str):
    with get_conn() as c:
        p = c.execute("SELECT part_id,name,category,spec,price,condition,stock_qty FROM parts_metadata WHERE part_id=%s",
                      (part_id,)).fetchone()
        if not p:
            raise HTTPException(404, "ไม่พบสินค้า")
        imgs = [r[0] for r in c.execute(
            "SELECT image_path FROM part_images WHERE part_id=%s ORDER BY view_order", (part_id,)).fetchall()]
    return dict(part_id=str(p[0]), name=p[1], category=p[2], spec=p[3],
                price=float(p[4]) if p[4] is not None else None, condition=p[5], stock_qty=p[6], images=imgs)


# ---------------- ผู้ดูแลระบบ: เข้าสู่ระบบ JWT + bcrypt ----------------
class LoginBody(BaseModel):
    email: str
    password: str


@app.post("/api/admin/login")
def login(body: LoginBody):
    with get_conn() as c:
        r = c.execute("SELECT admin_id,password_hash,status FROM admin_users WHERE email=%s", (body.email,)).fetchone()
    if not r or r[2] != "active" or not bcrypt.checkpw(body.password.encode(), r[1].encode()):
        raise HTTPException(401, "อีเมลหรือรหัสผ่านไม่ถูกต้อง")
    token = jwt.encode({"sub": str(r[0]), "exp": dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)},
                       JWT_SECRET, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}


def require_admin(authorization: str = Header(default="")) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "ต้องเข้าสู่ระบบ")
    try:
        return jwt.decode(authorization[7:], JWT_SECRET, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        raise HTTPException(401, "โทเคนไม่ถูกต้องหรือหมดอายุ")


@app.get("/api/admin/threshold")           # UC-06
def read_threshold(admin: str = Depends(require_admin)):
    with get_conn() as c:
        return {"similarity_threshold": get_threshold(c)}


class ThresholdBody(BaseModel):
    value: float


@app.put("/api/admin/threshold")
def set_threshold(body: ThresholdBody, admin: str = Depends(require_admin)):
    if not 0.0 <= body.value <= 1.0:
        raise HTTPException(422, "ค่า Threshold ต้องอยู่ระหว่าง 0.0000 ถึง 1.0000")
    with get_conn() as c:
        c.execute("UPDATE system_settings SET setting_value=%s, updated_by=%s, updated_at=now() "
                  "WHERE setting_key='similarity_threshold'", (f"{body.value:.4f}", admin))
    return {"similarity_threshold": body.value}


@app.post("/api/admin/parts")              # UC-04 เพิ่มสินค้า + สกัดเวกเตอร์อัตโนมัติ
async def add_part(name: str = Form(...), category: str = Form(...), spec: str = Form(""),
                   price: float = Form(0), condition: str = Form(""), stock_qty: int = Form(1),
                   images: list[UploadFile] = File(...), admin: str = Depends(require_admin)):
    if not images:
        raise HTTPException(422, "ต้องอัปโหลดภาพอย่างน้อย 1 ภาพ")
    pics = [read_image(await f.read()) for f in images]
    with get_conn() as c:
        pid = c.execute("INSERT INTO parts_metadata(name,category,spec,price,condition,stock_qty) "
                        "VALUES (%s,%s,%s,%s,%s,%s) RETURNING part_id",
                        (name, category, spec, price, condition, stock_qty)).fetchone()[0]
        for n, img in enumerate(pics, start=1):
            rel = os.path.join("parts", f"{pid}_{n}.jpg")
            img.save(os.path.join(UPLOAD_DIR, rel))
            iid = c.execute("INSERT INTO part_images(part_id,image_path,view_order) VALUES (%s,%s,%s) RETURNING image_id",
                            (pid, rel, n)).fetchone()[0]
            c.execute("INSERT INTO part_embeddings(part_id,image_id,embedding) VALUES (%s,%s,%s)",
                      (pid, iid, embedder.embed_image(img)))
    return {"part_id": str(pid), "images": len(pics)}


@app.get("/api/admin/logs")                # UC-07
def logs(limit: int = 50, blocked_only: bool = False, admin: str = Depends(require_admin)):
    sql = "SELECT log_id,query_image_path,top1_score,is_hard_blocked,result_count,latency_ms,searched_at FROM search_logs"
    if blocked_only:
        sql += " WHERE is_hard_blocked"
    sql += " ORDER BY searched_at DESC LIMIT %s"
    with get_conn() as c:
        rows = c.execute(sql, (min(limit, 500),)).fetchall()
        total, blocked = c.execute("SELECT count(*), count(*) FILTER (WHERE is_hard_blocked) FROM search_logs").fetchone()
    return {"total": total, "hard_blocked": blocked,
            "items": [dict(log_id=str(r[0]), query_image=r[1], top1_score=float(r[2]) if r[2] is not None else None,
                           is_hard_blocked=r[3], result_count=r[4], latency_ms=r[5], searched_at=r[6].isoformat())
                      for r in rows]}
