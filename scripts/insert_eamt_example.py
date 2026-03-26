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
        for split_name in ["train", "validation", "test"]:
            split_path = os.path.join(DATA_DIR, split_name)

            if not os.path.exists(split_path):
                continue

            for file_path in get_jsonl_files(split_path):
                records = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        records.append((
                            split_name,
                            data.get("source_locale"),
                            data.get("target_locale"),
                            data.get("source"),
                            data.get("id")
                        ))
                
                if records:
                    cur.executemany(insert_query, records)
                    conn.commit()
                    

    except Exception as e:
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()