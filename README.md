# ระบบค้นหาอะไหล่คอมพิวเตอร์มือสองด้วยรูปภาพ (Visual Search for Used PC Parts)

โครงงานวิศวกรรมซอฟต์แวร์ (ENGSE212) มหาวิทยาลัยเทคโนโลยีราชมงคลล้านนา
ผู้จัดทำ: นายภูริณัฐ เต๋จ๊ะ · อาจารย์ที่ปรึกษา: อ.นริศ กำแพงแก้ว

ผู้ใช้ถ่ายหรืออัปโหลดรูปอะไหล่ ครอบเฉพาะชิ้นที่สนใจ (Crop-to-Search) ระบบแปลงภาพเป็นเวกเตอร์ด้วยโมเดลสกัดคุณลักษณะภาพ
แล้วค้นหาอะไหล่ที่คล้ายที่สุดใน PostgreSQL + pgvector พร้อมกลไก Similarity Threshold / Hard Block

## โครงสร้าง

| โฟลเดอร์ | เนื้อหา |
|---|---|
| `backend/` | FastAPI, สคริปต์ฐานข้อมูล (`schema.sql`, `ingest_gallery.py`, `create_admin.py`, `evaluate.py`) |
| `frontend/` | เว็บ React (หน้าค้นหา, Crop, Admin Dashboard) |
| `experiments/` | สคริปต์ทดลอง เช่น `compare_models.py` (เปรียบเทียบโมเดล, TC-MODEL-01) และไฟล์ผลลัพธ์ CSV |
| `docs/` | เอกสารประกอบ หน้าเดโม และภาพหน้าจอ |

## เทคโนโลยี

Python 3.11, FastAPI, PyTorch + open_clip, PostgreSQL (Neon) + pgvector, JWT/bcrypt, React

## เริ่มใช้งาน (Backend)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy ..\.env.example .env   # แล้วแก้ DATABASE_URL ให้เป็นค่าจริง
uvicorn main:app --reload    # ปรับชื่อไฟล์ให้ตรงกับโค้ดจริง
```

## หมายเหตุ

- ไฟล์ `.env` (รหัสผ่านฐานข้อมูล) และโฟลเดอร์ภาพ Dataset ไม่ถูกเก็บใน Git
- ไลบรารีโอเพนซอร์สที่ใช้และสัญญาอนุญาต: open_clip (MIT), pgvector (PostgreSQL License), FastAPI (MIT), PyTorch (BSD-3)
