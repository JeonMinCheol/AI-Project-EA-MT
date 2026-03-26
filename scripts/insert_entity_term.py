import os
import json
import time
import requests
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

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "SemEval-Task2-Research-Bot (yujin0124@khu.ac.kr)"}

def fetch_wikidata(qids, languages):
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "labels|descriptions|aliases",
        "languages": "|".join(languages),
        "languagefallback": 1,
        "format": "json"
    }
    try:
        response = requests.get(WIKIDATA_API_URL, params=params, headers=HEADERS, timeout=15)
        return response.json().get("entities", {})
    except Exception as e:
        return {}

def main():
    if not os.path.exists(INPUT_FILE):
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        qid_data = json.load(f)
    

    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("SELECT qid, lang_code FROM entity_term WHERE term_type = 'label'")
    existing_pairs = set(cur.fetchall()) 

    lang_map = {}
    new_count = 0
    for it in qid_data:
        if (it['qid'], it['locale']) not in existing_pairs:
            lang_map.setdefault(it['locale'], []).append(it['qid'])
            new_count += 1

    if new_count == 0:
        cur.close()
        conn.close()
        return

    insert_query = """
    INSERT IGNORE INTO entity_term (qid, lang_code, term_type, term_text, normalized_text)
    VALUES (%s, %s, %s, %s, %s)
    """

    for lang, qids in lang_map.items():
        unique_qids = sorted(list(set(qids)))

        for i in range(0, len(unique_qids), 50):
            batch = unique_qids[i:i+50]
            entities = fetch_wikidata(batch, [lang, "en"])
            
            records = []
            for qid, details in entities.items():
                if "missing" in details: continue

                for l_code, info in details.get("labels", {}).items():
                    records.append((qid, l_code, "label", info['value'], None))

                for a_code, alias_list in details.get("aliases", {}).items():
                    for alias in alias_list:
                        records.append((qid, a_code, "alias", alias['value'], None))

                for d_code, info in details.get("descriptions", {}).items():
                    records.append((qid, d_code, "description", info['value'], None))
            
            if records:
                try:
                    cur.executemany(insert_query, records)
                    conn.commit()

                except Exception as e:
                    conn.rollback()

            time.sleep(0.1)

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()