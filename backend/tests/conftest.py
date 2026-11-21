import os, sys, pathlib
import psycopg, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ["FAKE_EMBED"] = "1"
BASE = os.getenv("TEST_DB_ADMIN_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
TEST_URL = BASE.rsplit("/", 1)[0] + "/visualsearch_test"
os.environ["DATABASE_URL"] = TEST_URL
os.environ["UPLOAD_DIR"] = str(pathlib.Path(__file__).parent / "_uploads")

@pytest.fixture(scope="session", autouse=True)
def test_database():
    with psycopg.connect(BASE, autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS visualsearch_test")
        c.execute("CREATE DATABASE visualsearch_test")
    schema = (pathlib.Path(__file__).resolve().parents[1] / "schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(TEST_URL, autocommit=True) as c:
        c.execute(schema)
    yield

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)
