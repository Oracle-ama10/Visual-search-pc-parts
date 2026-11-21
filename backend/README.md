# Backend ระบบค้นหาอะไหล่คอมมือสองด้วยรูปภาพ (FastAPI + PostgreSQL/pgvector + OpenCLIP)

## 1) เปิดฐานข้อมูล (ต้องติดตั้ง Docker Desktop)
    docker compose up -d
สร้าง 6 ตาราง + HNSW index + ค่าเริ่มต้น Threshold 0.9241 ให้อัตโนมัติจาก schema.sql

## 2) ติดตั้ง Python (ใน venv)
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## 3) นำภาพ Gallery เข้าฐานข้อมูล + สร้างผู้ดูแลระบบ
    python ingest_gallery.py ..\visual-search-test\gallery
    python create_admin.py admin@example.com รหัสผ่านของคุณ

## 4) รัน API
    uvicorn main:app --reload
เปิด http://127.0.0.1:8000/docs  (ลอง POST /api/search ด้วยภาพจาก query/)

## 5) วัดผลซ้ำผ่านฐานข้อมูลจริง
    python evaluate.py ..\visual-search-test\query

## 6) รันชุดทดสอบ (ต้องมีฐานข้อมูลเปิดอยู่)
    pytest -q tests

## Endpoint
| Method | Path | ผู้ใช้ | หน้าที่ |
|---|---|---|---|
| POST | /api/search | ลูกค้า | ค้นหาด้วยภาพ + Hard Block + บันทึก log |
| GET | /api/parts/{id} | ลูกค้า | รายละเอียดสินค้า |
| POST | /api/admin/login | แอดมิน | เข้าสู่ระบบ (JWT + bcrypt) |
| GET/PUT | /api/admin/threshold | แอดมิน | อ่าน/ตั้งค่า Threshold |
| POST | /api/admin/parts | แอดมิน | เพิ่มสินค้า + สกัดเวกเตอร์อัตโนมัติ |
| GET | /api/admin/logs | แอดมิน | ดูประวัติค้นหาและ Hard Block |
