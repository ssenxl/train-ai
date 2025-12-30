import os
import json
import time

# =====================
# CONFIG
# =====================
KEYS_FILE = "sheet_data/keys.json"
SEARCH_DIR = r"\\192.168.1.3\Master Spec- Costing"  # 🔧 ปรับ path ได้
OUT_FILE = "sheet_data/key_file_map.json"

# =====================
# LOAD KEYS
# =====================
with open(KEYS_FILE, "r", encoding="utf-8") as f:
    keys = json.load(f)

results = []

# =====================
# FIND EXCEL FILES
# =====================
for item in keys:
    key = item["key"]

    latest_path = None
    latest_mtime = -1

    # เดินค้นหาทุกไฟล์ที่ชื่อมี key นี้ แล้วเลือกไฟล์ที่แก้ไขล่าสุด
    for root, _, files in os.walk(SEARCH_DIR):
        for fname in files:
            # 🚫 ข้ามไฟล์ชั่วคราว Excel
            if fname.startswith("~$"):
                continue

            # รองรับทุก Excel
            if not fname.lower().endswith((".xlsx", ".xls", ".xlsm")):
                continue

            # match key
            if key.lower() in fname.lower():
                path = os.path.join(root, fname)
                mtime = os.path.getmtime(path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = path

    found_path = latest_path

    results.append({
        "row": item["row"],
        "key": key,
        "found": bool(found_path),
        "file_path": found_path
    })

    if found_path:
        print(f"✅ {key} → {found_path} (ล่าสุด: {time.ctime(latest_mtime)})")
    else:
        print(f"❌ {key} → ไม่พบไฟล์")

# =====================
# SAVE RESULT
# =====================
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

missing = [r for r in results if not r["found"]]
if missing:
    missing_path = os.path.join(os.path.dirname(OUT_FILE), "missing_keys.json")
    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)
    print(f"⚠️ มี {len(missing)} key ที่ยังไม่มีไฟล์ → บันทึกที่ {missing_path}")

print("\n🎉 STEP 2 DONE")
print(f"📁 บันทึกผลลัพธ์ที่: {OUT_FILE}")
