"""วัดผลซ้ำผ่านฐานข้อมูลจริง (Top-1, Top-5, MRR, Latency):  python evaluate.py ../query
ชื่อไฟล์ใน query/ ต้องตรงกับชื่อชิ้นใน gallery เช่น GPU_01.jpg"""
import sys, os, glob, time, csv
from PIL import Image
import embedder
from db import get_conn, search_parts

folder = sys.argv[1] if len(sys.argv) > 1 else "query"
files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
if not files:
    sys.exit(f"ไม่พบภาพใน {folder} กรุณาตรวจสอบ path ของโฟลเดอร์ query")
rows, t1, t5, rr, lat = [], 0, 0, 0.0, []
embedder.embed_image(Image.open(files[0]))  # warm-up โหลดโมเดลก่อนจับเวลา
with get_conn() as c:
    for f in files:
        truth = os.path.splitext(os.path.basename(f))[0]
        t0 = time.perf_counter()
        res = search_parts(c, embedder.embed_image(Image.open(f)))
        lat.append((time.perf_counter() - t0) * 1000)
        names = [r["name"] for r in res]
        rank = names.index(truth) + 1 if truth in names else None
        t1 += rank == 1; t5 += rank is not None and rank <= 5; rr += (1 / rank) if rank else 0
        rows.append((os.path.basename(f), truth, rank or "ไม่ติด Top-5", res[0]["name"] if res else "", res[0]["score"] if res else 0))
        print(rows[-1])
n = len(files)
print(f"\nQuery {n} ภาพ | Top-1 {t1/n:.2%} | Top-5 {t5/n:.2%} | MRR {rr/n:.3f} | เวลาเฉลี่ย {sum(lat)/n:.0f} ms")
with open("evaluation_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["query_file", "truth", "rank", "top1_pred", "top1_score"]); w.writerows(rows)
