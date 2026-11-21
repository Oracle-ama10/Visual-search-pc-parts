import os, psycopg
c = psycopg.connect(os.environ["DATABASE_URL"])
c.execute(open("schema.sql", encoding="utf-8").read())
c.commit()
n = c.execute("select count(*) from information_schema.tables where table_schema='public'").fetchone()[0]
print("tables:", n)
