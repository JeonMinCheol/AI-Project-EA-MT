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

def get_example_id_map(cur):
    cur.execute("SELECT dataset_id, example_id FROM eamt_example")
    return {dataset_id: ex_id for dataset_id, ex_id in cur.fetchall()}

def main():
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()

    id_map = get_example_id_map(cur)
    
    insert_query = """
    INSERT IGNORE INTO example_entity (example_id, qid, entity_order)
    VALUES (%s, %s, %s)
    """

    try:
        train_path = os.path.join(DATA_DIR, "train")
        for file_path in get_jsonl_files(train_path):
            records = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    ex_id = id_map.get(data['id'])
                    entities = data.get('entities', [])
                    
                    if ex_id and entities:
                        for idx, qid in enumerate(entities):
                            records.append((ex_id, qid, idx))
            
            if records:
                cur.executemany(insert_query, records)
                conn.commit()

        val_path = os.path.join(DATA_DIR, "validation")
        for file_path in get_jsonl_files(val_path):
            records = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    ex_id = id_map.get(data['id'])
                    qid = data.get('wikidata_id')
                    
                    if ex_id and qid:
                        records.append((ex_id, qid, 0))
            
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
