from openpyxl import load_workbook
import json
import os

# =====================
# CONFIG
# =====================
MASTER_FILE = "test_ms.xlsx"
OUT_DIR = "sheet_data"

KEY_COL = "F"     # คอลัมน์ที่เก็บ key
START_ROW = 3336    # 👈 แถวเริ่มต้น
END_ROW = 3355 # 👈 แถวสุดท้าย (None = ถึงท้ายชีท)

os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# LOAD EXCEL
# =====================
wb = load_workbook(MASTER_FILE, data_only=True)
ws = wb.active

records = []

last_row = END_ROW if END_ROW else ws.max_row

for row in range(START_ROW, last_row + 1):
    key = ws[f"{KEY_COL}{row}"].value
    if not key:
        continue

    records.append({
        "row": row,
        "key": str(key).strip()
    })

# =====================
# SAVE
# =====================
out_path = os.path.join(OUT_DIR, "keys.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("✅ STEP 1 DONE (กำหนดช่วงแถวได้)")
print(f"📄 ดึง key ทั้งหมด {len(records)} รายการ")
print(f"📁 บันทึกที่: {out_path}")
