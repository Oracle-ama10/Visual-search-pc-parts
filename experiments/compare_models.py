# -*- coding: utf-8 -*-
"""
เปรียบเทียบโมเดลสกัดคุณลักษณะภาพ (image embedding) สำหรับโครงงานค้นหาอะไหล่คอมพิวเตอร์ด้วยรูปภาพ
ใช้ชุดข้อมูลและวิธีวัดเดียวกับ SE-01 / evaluate.py:
  - Gallery: ชื่อไฟล์ <ITEM_ID>_<เลขมุม>.jpg   เช่น RAM_08_2.jpg  (item_id = ตัดส่วนหลัง _ ตัวสุดท้ายออก)
  - Query  : ชื่อไฟล์ = ITEM_ID ที่ถูกต้อง เช่น RAM_08.jpg
             ถ้าเพิ่มภาพ Query ของชิ้นเดิมหลายภาพ ให้ตั้งชื่อ  RAM_08__2.jpg , RAM_08__3.jpg  (ใช้ส่วนก่อน "__" เป็นเฉลย)
  - ภาพนอกคลัง: Query ที่ ITEM_ID ไม่มีใน Gallery (เช่น GPU_03, GPU_06) จะถูกแยกรายงานต่างหาก

วิธีรัน (PowerShell ในโฟลเดอร์ backend):
  & $py compare_models.py ..\\..\\visual-search-test\\gallery ..\\..\\visual-search-test\\query
ตัวเลือก:
  --models a,b,c     เลือกเฉพาะบางโมเดล (ดูชื่อด้วย --list)
  --out โฟลเดอร์     ที่เก็บไฟล์ผล CSV (ค่าเริ่มต้น model_comparison)
  --device cpu|cuda  (ค่าเริ่มต้น: cuda ถ้ามี ไม่งั้น cpu)
  --list             แสดงรายชื่อโมเดลแล้วจบ
  --selftest         ทดสอบตรรกะการวัดด้วยข้อมูลจำลอง (ไม่โหลดโมเดลจริง)

หมายเหตุ: ถ้าบางโมเดลโหลดไม่ได้ (ไม่มีอินเทอร์เน็ต / ยังไม่ได้ติดตั้งไลบรารี) สคริปต์จะข้ามและรายงานเหตุผล
ต้องใช้: open_clip_torch, torch, torchvision, pillow, numpy ; สำหรับ DINOv2 ต้องมี transformers (pip install transformers)
"""
import argparse, csv, glob, os, sys, time
import numpy as np
from PIL import Image

# ------------------------------------------------------------------ โมเดลที่เปรียบเทียบ
# ชื่อ -> (ชนิด, พารามิเตอร์, คำอธิบายสั้น ๆ)
MODELS = {
    "clip_b32_openai_asis":   ("openclip", ("ViT-B-32", "openai"),
                               "ค่าที่ใช้อยู่ใน SE-01/Backend (น้ำหนัก OpenAI CLIP + นิยามโมเดลแบบไม่ quickgelu)"),
    "clip_b32_openai_qgelu":  ("openclip", ("ViT-B-32-quickgelu", "openai"),
                               "น้ำหนัก OpenAI CLIP เดิม แต่ตั้งค่า QuickGELU ให้ตรงกับน้ำหนัก"),
    "openclip_b32_laion2b":   ("openclip", ("ViT-B-32", "laion2b_s34b_b79k"),
                               "OpenCLIP ViT-B/32 ฝึกด้วย LAION-2B (สถาปัตยกรรมเดียวกัน ข้อมูลฝึกต่างกัน)"),
    "openclip_b16_datacomp":  ("openclip", ("ViT-B-16", "datacomp_xl_s13b_b90k"),
                               "OpenCLIP ViT-B/16 ฝึกด้วย DataComp-1B"),
    "siglip_b16":             ("openclip", ("ViT-B-16-SigLIP", "webli"),
                               "SigLIP ViT-B/16 (sigmoid loss, WebLI) โหลดผ่าน OpenCLIP"),
    "dinov2_small":           ("dinov2", ("facebook/dinov2-small", "cls"),
                               "DINOv2 ViT-S/14 เวกเตอร์ CLS 384 มิติ (โมเดลภาพล้วน ไม่มีตัวเข้ารหัสข้อความ)"),
    "dinov2_base":            ("dinov2", ("facebook/dinov2-base", "cls"),
                               "DINOv2 ViT-B/14 เวกเตอร์ CLS 768 มิติ"),
    "dinov2_base_clsmean":    ("dinov2", ("facebook/dinov2-base", "cls+mean"),
                               "DINOv2 ViT-B/14 ต่อ CLS กับค่าเฉลี่ย patch (1536 มิติ)"),
    "resnet50_imagenet":      ("resnet", ("resnet50", None),
                               "ResNet-50 (ImageNet, supervised) เวกเตอร์ 2048 มิติ  = ตัวแทน CNN พื้นฐาน [He et al. 2016]"),
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


# ------------------------------------------------------------------ ตัวสกัดเวกเตอร์
class Embedder:
    def __init__(self, name, device):
        self.name, self.device = name, device
        kind, params, _ = MODELS[name]
        import torch
        self.torch = torch
        if kind == "openclip":
            import open_clip
            arch, tag = params
            self.model, _, self.pre = open_clip.create_model_and_transforms(arch, pretrained=tag)
            self.model = self.model.to(device).eval()
            self.fn = self._openclip
        elif kind == "dinov2":
            from transformers import AutoModel, AutoImageProcessor
            mid, mode = params
            self.proc = AutoImageProcessor.from_pretrained(mid)
            self.model = AutoModel.from_pretrained(mid).to(device).eval()
            self.mode = mode
            self.fn = self._dinov2
        elif kind == "resnet":
            import torchvision
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            m = torchvision.models.resnet50(weights=w)
            m.fc = torch.nn.Identity()
            self.model = m.to(device).eval()
            self.pre = w.transforms()
            self.fn = self._resnet
        else:
            raise ValueError(kind)
        self.n_params = sum(p.numel() for p in self.model.parameters()) / 1e6

    def _norm(self, v):
        v = v.float()
        return (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

    def _openclip(self, img):
        x = self.pre(img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return self._norm(self.model.encode_image(x))

    def _dinov2(self, img):
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
            cls = out.last_hidden_state[:, 0]
            if self.mode == "cls+mean":
                mean = out.last_hidden_state[:, 1:].mean(dim=1)
                v = self.torch.cat([cls / cls.norm(dim=-1, keepdim=True), mean / mean.norm(dim=-1, keepdim=True)], dim=-1)
            else:
                v = cls
        return self._norm(v)

    def _resnet(self, img):
        x = self.pre(img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return self._norm(self.model(x))

    def embed(self, path):
        return self.fn(Image.open(path).convert("RGB"))


class FakeEmbedder:
    """ใช้เฉพาะ --selftest: สร้างเวกเตอร์จากชื่อไฟล์ เพื่อตรวจตรรกะการวัดผล"""
    name = "fake"; n_params = 0.0
    def __init__(self, noise=0.35):
        self.noise = noise
    def embed(self, path):
        base = os.path.splitext(os.path.basename(path))[0]
        in_gallery_dir = os.path.basename(os.path.dirname(path)) == "g"
        item = gallery_item_id(path) if in_gallery_dir else query_truth(path)
        rng = np.random.default_rng(abs(hash(item)) % (2**32))
        v = rng.normal(size=64)
        rng2 = np.random.default_rng(abs(hash(base + str(in_gallery_dir))) % (2**32))
        v = v + self.noise * rng2.normal(size=64)
        return v / np.linalg.norm(v)


# ------------------------------------------------------------------ ข้อมูล
def list_images(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(IMG_EXT))

def gallery_item_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 1)[0]

def query_truth(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("__")[0]


# ------------------------------------------------------------------ การวัดผล
def rank_items(qv, gal_vecs, gal_items):
    """คืนรายการ (item_id, คะแนน) เรียงจากมากไปน้อย โดยใช้ภาพที่คล้ายที่สุดของแต่ละชิ้น"""
    sims = gal_vecs @ qv
    best = {}
    for s, it in zip(sims, gal_items):
        if it not in best or s > best[it]:
            best[it] = float(s)
    return sorted(best.items(), key=lambda x: -x[1])

def evaluate(embedder, gal_paths, qry_paths, log=print):
    gal_items = [gallery_item_id(p) for p in gal_paths]
    gal_set = set(gal_items)
    t0 = time.perf_counter()
    gal_vecs = np.stack([embedder.embed(p) for p in gal_paths])
    # ทำซ้ำอีกครั้งเพื่อวัดเวลาแบบอุ่นเครื่องแล้ว
    t1 = time.perf_counter()
    lat = []
    q_rows = []
    for p in qry_paths:
        s = time.perf_counter()
        qv = embedder.embed(p)
        lat.append((time.perf_counter() - s) * 1000)
        ranked = rank_items(qv, gal_vecs, gal_items)
        truth = query_truth(p)
        in_gal = truth in gal_set
        rank = next((i + 1 for i, (it, _) in enumerate(ranked) if it == truth), None) if in_gal else None
        q_rows.append(dict(query=os.path.basename(p), truth=truth, in_gallery=in_gal, rank=rank,
                           top1_item=ranked[0][0], top1_score=ranked[0][1],
                           top2_score=ranked[1][1] if len(ranked) > 1 else float("nan")))
    return q_rows, lat, gal_vecs.shape[1]

def summarize(rows):
    def stats(sub):
        n = len(sub)
        if n == 0:
            return dict(n=0, top1=float("nan"), top5=float("nan"), mrr5=float("nan"), mrr_full=float("nan"))
        top1 = sum(1 for r in sub if r["rank"] == 1) / n
        top5 = sum(1 for r in sub if r["rank"] is not None and r["rank"] <= 5) / n
        mrr5 = sum((1.0 / r["rank"]) for r in sub if r["rank"] is not None and r["rank"] <= 5) / n
        mrr_full = sum((1.0 / r["rank"]) for r in sub if r["rank"] is not None) / n
        return dict(n=n, top1=top1, top5=top5, mrr5=mrr5, mrr_full=mrr_full)
    allq = stats(rows)
    ing = stats([r for r in rows if r["in_gallery"]])
    # ความสามารถของ Threshold ในการแยกภาพนอกคลัง
    correct_scores = [r["top1_score"] for r in rows if r["in_gallery"] and r["rank"] == 1]
    oog_scores = [r["top1_score"] for r in rows if not r["in_gallery"]]
    sep = dict(n_oog=len(oog_scores))
    if correct_scores and oog_scores:
        sep["margin"] = min(correct_scores) - max(oog_scores)   # > 0 = มี Threshold ที่แยกได้สมบูรณ์
        pos = [r["top1_score"] for r in rows if r["in_gallery"]]
        wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in oog_scores)
        sep["auc"] = wins / (len(pos) * len(oog_scores))
    else:
        sep["margin"] = float("nan"); sep["auc"] = float("nan")
    return allq, ing, sep

def pct(x):
    return "   - " if x != x else f"{100 * x:5.1f}%"

# ------------------------------------------------------------------ หลัก
def run(gallery, query, names, device, out_dir, fake=False):
    gal = list_images(gallery); qry = list_images(query)
    if not gal or not qry:
        sys.exit(f"ไม่พบภาพ: gallery={len(gal)} query={len(qry)} (ตรวจ path)")
    gal_set = {gallery_item_id(p) for p in gal}
    print(f"Gallery {len(gal)} ภาพ จาก {len(gal_set)} ชิ้น | Query {len(qry)} ภาพ "
          f"(ในคลัง {sum(query_truth(p) in gal_set for p in qry)}, นอกคลัง {sum(query_truth(p) not in gal_set for p in qry)}) | device={device}")
    os.makedirs(out_dir, exist_ok=True)
    summary, per_query = [], []
    for name in names:
        print(f"\n=== {name}: {MODELS[name][2] if name in MODELS else 'selftest'}")
        try:
            emb = FakeEmbedder() if fake else Embedder(name, device)
            # อุ่นเครื่อง 1 ครั้ง (ไม่นับเวลา)
            emb.embed(qry[0])
            rows, lat, dim = evaluate(emb, gal, qry)
        except Exception as e:
            print(f"   ข้ามโมเดลนี้: {type(e).__name__}: {e}")
            summary.append(dict(model=name, status=f"skipped: {type(e).__name__}: {e}"))
            continue
        a, i, sep = summarize(rows)
        med = float(np.median(lat))
        print(f"   มิติเวกเตอร์ {dim} | พารามิเตอร์ {emb.n_params:.0f}M | เวลาสกัดต่อภาพ (มัธยฐาน) {med:.0f} ms")
        print(f"   ทุก Query (n={a['n']}) : Top-1 {pct(a['top1'])}  Top-5 {pct(a['top5'])}  MRR@5 {a['mrr5']:.3f}  MRR(เต็ม) {a['mrr_full']:.3f}")
        print(f"   เฉพาะในคลัง (n={i['n']}): Top-1 {pct(i['top1'])}  Top-5 {pct(i['top5'])}  MRR@5 {i['mrr5']:.3f}  MRR(เต็ม) {i['mrr_full']:.3f}")
        print(f"   แยกภาพนอกคลัง (n={sep['n_oog']}): margin {sep['margin']:.4f}  AUC {sep['auc']:.3f}")
        summary.append(dict(model=name, status="ok", dim=dim, params_m=round(emb.n_params, 1), latency_ms_median=round(med, 1),
                            all_n=a["n"], all_top1=a["top1"], all_top5=a["top5"], all_mrr5=a["mrr5"], all_mrr_full=a["mrr_full"],
                            ing_n=i["n"], ing_top1=i["top1"], ing_top5=i["top5"], ing_mrr5=i["mrr5"], ing_mrr_full=i["mrr_full"],
                            oog_n=sep["n_oog"], oog_margin=sep["margin"], oog_auc=sep["auc"]))
        for r in rows:
            per_query.append(dict(model=name, **r))
    keys = sorted({k for r in summary for k in r}, key=lambda k: (k not in ("model", "status"), k))
    with open(os.path.join(out_dir, "model_comparison_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
    if per_query:
        with open(os.path.join(out_dir, "model_comparison_per_query.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(per_query[0].keys())); w.writeheader(); w.writerows(per_query)
    print(f"\nบันทึกผลที่ {out_dir}\\model_comparison_summary.csv และ model_comparison_per_query.csv")
    print("ตรวจสอบความถูกต้อง: แถว clip_b32_openai_asis ควรได้ Top-1 54.5%, Top-5 81.8%, MRR@5 ใกล้ 0.65 เท่า SE-01 "
          "ถ้าไม่ตรง แปลว่าชุดไฟล์หรือชื่อไฟล์ต่างจากเดิม ให้แจ้งผมก่อนอ่านผลโมเดลอื่น")
    return summary


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    g, q = os.path.join(d, "g"), os.path.join(d, "q")
    os.makedirs(g); os.makedirs(q)
    items = ["A_01", "A_02", "B_01", "B_02", "C_01"]
    for it in items:
        for k in (1, 2):
            Image.new("RGB", (8, 8)).save(os.path.join(g, f"{it}_{k}.jpg"))
    for it in items[:4]:
        Image.new("RGB", (8, 8)).save(os.path.join(q, f"{it}.jpg"))
    Image.new("RGB", (8, 8)).save(os.path.join(q, "A_01__2.jpg"))      # Query เพิ่มของชิ้นเดิม
    Image.new("RGB", (8, 8)).save(os.path.join(q, "Z_99.jpg"))         # นอกคลัง
    gal, qry = list_images(g), list_images(q)
    rows, lat, dim = evaluate(FakeEmbedder(noise=0.05), gal, qry)
    a, i, sep = summarize(rows)
    assert a["n"] == 6 and i["n"] == 5, (a, i)
    assert i["top1"] == 1.0 and i["top5"] == 1.0, i               # ไม่มี noise มาก ต้องหาถูกทุกภาพ
    assert sep["n_oog"] == 1
    assert gallery_item_id("x/CPU cooler_01_2.jpg") == "CPU cooler_01"
    assert query_truth("x/RAM_08__3.jpg") == "RAM_08" and query_truth("x/RAM_08.jpg") == "RAM_08"
    # ตรวจสูตร MRR ด้วยมือ
    fake = [dict(rank=1, in_gallery=True, top1_score=.9), dict(rank=3, in_gallery=True, top1_score=.8),
            dict(rank=None, in_gallery=False, top1_score=.7), dict(rank=8, in_gallery=True, top1_score=.6)]
    a2, i2, s2 = summarize(fake)
    assert abs(a2["mrr5"] - (1 + 1 / 3) / 4) < 1e-9 and abs(a2["mrr_full"] - (1 + 1 / 3 + 1 / 8) / 4) < 1e-9, a2
    assert abs(i2["top5"] - 2 / 3) < 1e-9
    print("selftest ผ่านทั้งหมด")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gallery", nargs="?"); ap.add_argument("query", nargs="?")
    ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--out", default="model_comparison")
    ap.add_argument("--device", default=None); ap.add_argument("--list", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    if a.list:
        for k, v in MODELS.items(): print(f"{k:26s} {v[2]}")
        sys.exit(0)
    if not a.gallery or not a.query:
        ap.error("ต้องระบุโฟลเดอร์ gallery และ query")
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    bad = [n for n in names if n not in MODELS]
    if bad: sys.exit(f"ไม่รู้จักโมเดล: {bad}  (ดู --list)")
    device = a.device
    if device is None:
        try:
            import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    run(a.gallery, a.query, names, device, a.out)
