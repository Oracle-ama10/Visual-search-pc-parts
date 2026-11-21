-- ============================================================
-- schema.sql — ฐานข้อมูลระบบค้นหาอะไหล่คอมมือสองด้วยรูปภาพ
-- PostgreSQL 16 + pgvector  (ตรงกับตารางที่ 9 ใน SE02: 6 ตาราง)
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS admin_users (
    admin_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email          VARCHAR(255) NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,              -- bcrypt
    status         VARCHAR(20)  NOT NULL DEFAULT 'active'
                   CHECK (status IN ('active','inactive')),
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS parts_metadata (
    part_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(200) NOT NULL,
    category    VARCHAR(50)  NOT NULL,                 -- GPU / Motherboard / RAM / Storage / ...
    spec        TEXT,
    price       NUMERIC(10,2),
    condition   VARCHAR(50),
    stock_qty   INTEGER NOT NULL DEFAULT 0 CHECK (stock_qty >= 0),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS part_images (
    image_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    part_id     UUID NOT NULL REFERENCES parts_metadata(part_id) ON DELETE CASCADE,
    image_path  VARCHAR(500) NOT NULL,
    view_order  INTEGER NOT NULL DEFAULT 1,            -- ลำดับมุมมอง (ใช้ 2-3 มุมต่อชิ้น)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS part_embeddings (
    embedding_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    part_id       UUID NOT NULL REFERENCES parts_metadata(part_id) ON DELETE CASCADE,
    image_id      UUID NOT NULL REFERENCES part_images(image_id)   ON DELETE CASCADE,
    embedding     vector(512) NOT NULL,                -- OpenCLIP ViT-B/32
    model_version VARCHAR(50)  NOT NULL DEFAULT 'ViT-B-32/openai',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
-- ดัชนี HNSW (ดัชนีหลัก) สำหรับ cosine distance
CREATE INDEX IF NOT EXISTS idx_part_embeddings_hnsw
    ON part_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS search_logs (
    log_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_image_path VARCHAR(500),
    top1_score       NUMERIC(6,4),
    is_hard_blocked  BOOLEAN NOT NULL DEFAULT FALSE,
    result_count     INTEGER NOT NULL DEFAULT 0,
    latency_ms       INTEGER,
    searched_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_settings (
    setting_key    VARCHAR(100) PRIMARY KEY,
    setting_value  VARCHAR(200) NOT NULL,
    updated_by     UUID REFERENCES admin_users(admin_id),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ค่าเริ่มต้นจากผลการทดลองจริงใน SE01 (Empirical Threshold)
INSERT INTO system_settings (setting_key, setting_value)
VALUES ('similarity_threshold', '0.9241')
ON CONFLICT (setting_key) DO NOTHING;
