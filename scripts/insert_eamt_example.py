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
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

def get_jsonl_files(root_path):
    jsonl_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def main():
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()

    insert_query = """
    INSERT IGNORE INTO eamt_example (split, source_locale, target_locale, source_text, dataset_id)
    VALUES (%s, %s, %s, %s, %s)
    """

    try:
        train_path = os.path.join(DATA_DIR, "train")
        for file_path in get_jsonl_files(train_path):
            records = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    records.append((
                        "train",
                        data.get("source_locale"),
                        data.get("target_locale"),
                        data.get("source"),
                        data.get("id")
                    ))
            cur.executemany(insert_query, records)
            conn.commit()

        val_path = os.path.join(DATA_DIR, "validation")
        for file_path in get_jsonl_files(val_path):
            records = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    records.append((
                        "validation",
                        data.get("source_locale"),
                        data.get("target_locale"),
                        data.get("source"),
                        data.get("id")
                    ))
            cur.executemany(insert_query, records)
            conn.commit()

    except Exception as e:
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()