import os
from contextlib import contextmanager
import psycopg
from pgvector.psycopg import register_vector

def database_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/visualsearch")

@contextmanager
def get_conn():
    conn = psycopg.connect(database_url())
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ค้นหา: เอา 30 ภาพที่ใกล้ที่สุด (ใช้ดัชนี HNSW) แล้วรวมเป็นรายชิ้น เรียงตามคะแนนสูงสุด
SEARCH_SQL = """
WITH nn AS (
    SELECT part_id, 1 - (embedding <=> %(v)s) AS score
    FROM part_embeddings
    ORDER BY embedding <=> %(v)s
    LIMIT %(pool)s
)
SELECT p.part_id, p.name, p.category, p.price, p.condition, p.stock_qty, MAX(nn.score) AS score,
       (SELECT image_path FROM part_images i WHERE i.part_id = p.part_id ORDER BY view_order LIMIT 1) AS image_path
FROM nn JOIN parts_metadata p USING (part_id)
GROUP BY p.part_id
ORDER BY score DESC
LIMIT %(k)s
"""

def search_parts(conn, vec, k=5, pool=30):
    rows = conn.execute(SEARCH_SQL, {"v": vec, "pool": pool, "k": k}).fetchall()
    return [dict(part_id=str(r[0]), name=r[1], category=r[2], price=float(r[3]) if r[3] is not None else None,
                 condition=r[4], stock_qty=r[5], score=round(float(r[6]), 4), image_path=r[7]) for r in rows]

def get_threshold(conn) -> float:
    r = conn.execute("SELECT setting_value FROM system_settings WHERE setting_key='similarity_threshold'").fetchone()
    return float(r[0]) if r else 0.9241
