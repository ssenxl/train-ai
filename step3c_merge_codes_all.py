import os
import json

SRC_DIR = "sheet_data/codes"
OUT_FILE = "sheet_data/codes_all.json"

records = []

if not os.path.isdir(SRC_DIR):
    raise FileNotFoundError(f"❌ ไม่พบโฟลเดอร์ {SRC_DIR}")

for fn in sorted(os.listdir(SRC_DIR)):
    if not fn.lower().endswith(".json"):
        continue

    path = os.path.join(SRC_DIR, fn)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        records.append(data)

if not records:
    raise RuntimeError("❌ ไม่พบไฟล์ json ใด ๆ ใน sheet_data/codes")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"✅ สร้างไฟล์สำเร็จ: {OUT_FILE}")
print(f"📦 รวมทั้งหมด {len(records)} records")
