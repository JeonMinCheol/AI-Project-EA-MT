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
DATA_DIRS = [
    os.path.join(BASE_DIR, "data", "raw", "validation"),
    os.path.join(BASE_DIR, "data", "raw", "test")
]

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

    insert_query = "INSERT IGNORE INTO entity_type (type_text, qid) VALUES (%s, %s)"

    unique_type_mappings = set()

    try:
        jsonl_files = []
        for directory in DATA_DIRS:
            if os.path.exists(directory):
                jsonl_files.extend(get_jsonl_files(directory))
        
        for file_path in jsonl_files:
            file_name = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    qid = data.get("wikidata_id")
                    types = data.get("entity_types", [])
                    
                    if qid and types:
                        for t in types:
                            unique_type_mappings.add((t, qid))
        
        if unique_type_mappings:
            records = list(unique_type_mappings)
            
            cur.executemany(insert_query, records)
            conn.commit()

    except Exception as e:
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()