import os
import json
import pymysql
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "db": os.getenv("DB_NAME"),
    "charset": "utf8mb4"
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "qid_locale.json")

def main():
    if not os.path.exists(INPUT_FILE):
        return
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        qid_data = json.load(f)

    unique_qids = sorted(list(set(item['qid'] for item in qid_data)))

    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()

    insert_query = "INSERT IGNORE INTO entity (qid, popularity_score) VALUES (%s, %s)"

    records = [(qid, None) for qid in unique_qids]

    try:
        batch_size = 10000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            cur.executemany(insert_query, batch)
            conn.commit()
    
    except Exception as e:
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()