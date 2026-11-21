"""นำภาพใน gallery/ (ชื่อไฟล์แบบ GPU_01_front.jpg) เข้าฐานข้อมูล: python ingest_gallery.py ../gallery"""
import sys, os, glob, shutil
from PIL import Image
import embedder
from db import get_conn

CATS = {"gpu": "GPU", "ram": "RAM", "hdd": "Storage", "ssd": "Storage", "motherboard": "Motherboard",
        "mb": "Motherboard", "mboard": "Motherboard", "cpu": "CPU", "cpu cooler": "CPU Cooler"}
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

def category_of(item_id: str) -> str:
    prefix = item_id.rsplit("_", 1)[0].lower()
    return CATS.get(prefix, prefix.upper())

folder = sys.argv[1] if len(sys.argv) > 1 else "gallery"
files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
if not files:
    sys.exit(f"ไม่พบภาพใน {folder} กรุณาตรวจสอบ path ของโฟลเดอร์ gallery")
os.makedirs(os.path.join(UPLOAD_DIR, "parts"), exist_ok=True)
added = skipped = 0
with get_conn() as c:
    for f in files:
        item_id = os.path.basename(f).rsplit("_", 1)[0]          # เช่น GPU_01
        row = c.execute("SELECT part_id FROM parts_metadata WHERE name=%s", (item_id,)).fetchone()
        if row:
            pid = row[0]
        else:
            pid = c.execute("INSERT INTO parts_metadata(name,category,stock_qty) VALUES (%s,%s,1) RETURNING part_id",
                            (item_id, category_of(item_id))).fetchone()[0]
        rel = os.path.join("parts", f"{item_id}__{os.path.basename(f)}")
        if c.execute("SELECT 1 FROM part_images WHERE image_path=%s", (rel,)).fetchone():
            skipped += 1; continue
        shutil.copy(f, os.path.join(UPLOAD_DIR, rel))
        n = c.execute("SELECT count(*)+1 FROM part_images WHERE part_id=%s", (pid,)).fetchone()[0]
        iid = c.execute("INSERT INTO part_images(part_id,image_path,view_order) VALUES (%s,%s,%s) RETURNING image_id",
                        (pid, rel, n)).fetchone()[0]
        c.execute("INSERT INTO part_embeddings(part_id,image_id,embedding) VALUES (%s,%s,%s)",
                  (pid, iid, embedder.embed_image(Image.open(f))))
        added += 1
        print("เพิ่ม", rel)
print(f"เสร็จสิ้น: เพิ่ม {added} ภาพ, ข้าม {skipped} ภาพ")
