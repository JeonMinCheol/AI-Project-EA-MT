import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
VALIDATION_DIR = os.path.join(RAW_DATA_DIR, "validation")
TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "qid_locale.json")

if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

qid_locale_pairs = set() 


def process_file(file_path, is_mintaka = False):
    count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                locale = data.get('target_locale')

                if is_mintaka:
                    qids = data.get('entities', [])
                    for qid in qids:
                        if qid and locale:
                            qid_locale_pairs.add((qid, locale))
                            count += 1
                
                else:
                    qid = data.get('wikidata_id')
                    if qid and locale:
                        qid_locale_pairs.add((qid, locale))
                        count += 1
    except Exception as e:
        print(f"File Read Error ({file_path}): {e}")
    return count

if os.path.exists(VALIDATION_DIR):
    for filename in os.listdir(VALIDATION_DIR):
        if filename.endswith(".jsonl"):
            path = os.path.join(VALIDATION_DIR, filename)
            c = process_file(path, is_mintaka=False)
            print(f"{filename}: {c} ")
else:
    print(f"Not Found VALIDATION DIR")

if os.path.exists(TRAIN_DIR):
    for root, dirs, files in os.walk(TRAIN_DIR):
        for filename in files:
            if filename == "train.jsonl":
                path = os.path.join(root, filename)
                folder_name = os.path.basename(root)
                c = process_file(path, is_mintaka=True)
                print(f"{filename}: {c} ")
else:
    print(f"Not Found TRAIN DIR")

final_list = [{"qid": q, "locale": l} for q, l in sorted(list(qid_locale_pairs))]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_list, f, indent=4, ensure_ascii=False)


print(f"unique (QID, locale) pairs count : {len(final_list):}")