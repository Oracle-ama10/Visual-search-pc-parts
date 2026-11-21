"""สร้างบัญชีผู้ดูแลระบบ:  python create_admin.py admin@example.com รหัสผ่าน"""
import sys, bcrypt
from db import get_conn
email, pw = sys.argv[1], sys.argv[2]
with get_conn() as c:
    c.execute("INSERT INTO admin_users(email,password_hash) VALUES (%s,%s) "
              "ON CONFLICT (email) DO UPDATE SET password_hash=EXCLUDED.password_hash",
              (email, bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()))
print("สร้าง/อัปเดตผู้ดูแลระบบแล้ว:", email)
