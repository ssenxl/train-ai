import os
import json
from glob import glob

# =========================
# CONFIG
# =========================
IN_DIR = "sheet_data/codes"          # output จาก STEP 3B
OUT_FILE = "sheet_data/codes_all.json"

# =========================
# MAIN
# =========================
if not os.path.exists(IN_DIR):
    raise FileNotFoundError(f"❌ ไม่พบโฟลเดอร์ {IN_DIR}")

json_files = glob(os.path.join(IN_DIR, "*.json"))
print(f"📂 พบไฟล์ใน codes/: {len(json_files)} ไฟล์")

merged = []
skipped = []

for path in json_files:
    name = os.path.basename(path)

    # กันไฟล์ที่ไม่ใช่ output STEP 3B
    if name in ("key_file_map.json", "keys.json", "product_codes.json"):
        skipped.append(name)
        continue

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ตรวจโครงสร้างขั้นต่ำ
        if not isinstance(data, dict) or "key" not in data or "codes" not in data:
            print(f"⚠️ โครงสร้างไม่ถูกต้อง ข้ามไฟล์: {name}")
            skipped.append(name)
            continue

        print(" - อ่าน:", name)
        merged.append(data)

    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่ได้: {name} | {e}")
        skipped.append(name)

# =========================
# WRITE OUTPUT
# =========================
if not merged:
    print("❌ ไม่พบข้อมูล STEP 3B ให้ merge")
else:
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("\n✅ STEP 4A COMPLETE")
    print(f"📄 saved: {OUT_FILE}")
    print(f"📊 รวมทั้งหมด {len(merged)} keys")

if skipped:
    print("\n⚠️ ไฟล์ที่ถูกข้าม:")
    for s in skipped:
        print(" -", s)
