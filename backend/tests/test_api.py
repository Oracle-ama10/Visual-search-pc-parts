"""ชุดทดสอบระบบ (Functional Test) — ใช้ฐานข้อมูลจริง แต่ใช้ตัวสกัดเวกเตอร์จำลอง (FAKE_EMBED)
เพื่อทดสอบตรรกะของ API: ค้นหา, Hard Block, Threshold, Login, บันทึก Log"""
import io, bcrypt
import numpy as np
from PIL import Image
from db import get_conn


def make_img(seed: int) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); return buf.getvalue()


def login(client):
    with get_conn() as c:
        c.execute("INSERT INTO admin_users(email,password_hash) VALUES (%s,%s) ON CONFLICT (email) DO NOTHING",
                  ("admin@test.com", bcrypt.hashpw(b"secret123", bcrypt.gensalt()).decode()))
    r = client.post("/api/admin/login", json={"email": "admin@test.com", "password": "secret123"})
    assert r.status_code == 200
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def add_part(client, h, name, seeds):
    files = [("images", (f"{i}.png", make_img(s), "image/png")) for i, s in enumerate(seeds)]
    r = client.post("/api/admin/parts", headers=h, files=files,
                    data={"name": name, "category": "GPU", "price": "1500", "stock_qty": "2"})
    assert r.status_code == 200, r.text
    return r.json()["part_id"]


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_login_wrong_password(client):
    login(client)
    r = client.post("/api/admin/login", json={"email": "admin@test.com", "password": "wrong"})
    assert r.status_code == 401


def test_admin_endpoints_require_token(client):
    assert client.get("/api/admin/threshold").status_code == 401
    assert client.get("/api/admin/logs").status_code == 401


def test_add_part_and_search_match(client):
    h = login(client)
    pid = add_part(client, h, "GPU_TEST_A", [1, 2])
    r = client.post("/api/search", files={"file": ("q.png", make_img(1), "image/png")})
    body = r.json()
    assert r.status_code == 200
    assert body["is_hard_blocked"] is False
    assert body["results"][0]["part_id"] == pid
    assert body["top1_score"] > 0.99


def test_hard_block_on_unknown_image(client):
    h = login(client)
    add_part(client, h, "GPU_TEST_B", [10])
    r = client.post("/api/search", files={"file": ("q.png", make_img(999), "image/png")})
    body = r.json()
    assert body["is_hard_blocked"] is True and body["results"] == []
    assert body["fallback_message"]


def test_threshold_update_and_validation(client):
    h = login(client)
    assert client.put("/api/admin/threshold", headers=h, json={"value": 1.5}).status_code == 422
    assert client.put("/api/admin/threshold", headers=h, json={"value": 0.0}).status_code == 200
    r = client.post("/api/search", files={"file": ("q.png", make_img(999), "image/png")})
    assert r.json()["is_hard_blocked"] is False          # threshold 0 -> ไม่ตัด
    client.put("/api/admin/threshold", headers=h, json={"value": 0.9241})
    assert client.get("/api/admin/threshold", headers=h).json()["similarity_threshold"] == 0.9241


def test_logs_recorded(client):
    h = login(client)
    client.post("/api/search", files={"file": ("q.png", make_img(5), "image/png")})
    d = client.get("/api/admin/logs", headers=h).json()
    assert d["total"] >= 1 and "latency_ms" in d["items"][0]


def test_invalid_image_rejected(client):
    r = client.post("/api/search", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
