import os
import json
import shutil
import hashlib
import win32com.client
import pythoncom
from PIL import Image

# =========================
# CONFIG
# =========================
MAP_FILE = "sheet_data/key_file_map.json"

TEMP_WEB_ROOT = r"C:\temp_web"
OUTPUT_ROOT   = "final_images"
SMALL_IMAGE_DIR = "small_images"

SMALL_NAME_WHITELIST = [
    "3mm",
    "suree",
    "sambal",
    "imported",
    "label",
]

MIN_WIDTH  = 50
MIN_HEIGHT = 50

# =========================
# FORCE KEEP KEYS (DO NOT DELETE)
# =========================
FORCE_KEEP_KEYS = [
    "GC14L9FECW003191106",
    "GA01BBFFCX0031911IW",
    "GD02A9229604270S112",
    "GE10B919AB03270S112",
    "GD02A9229604270S112",
    "GA01K912CZ00270S109",
    "BA17BLB022270S007",
]

# =========================
# UTIL
# =========================
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def is_image(fname):
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"))

def md5_of_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# =========================
# STEP 1: SAVE AS WEBPAGE
# =========================
def save_excel_as_webpage(excel_path, out_dir, key):
    pythoncom.CoInitialize()
    excel = win32com.client.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    try:
        wb = excel.Workbooks.Open(excel_path, ReadOnly=True, UpdateLinks=0)
        html_path = os.path.join(out_dir, f"{key}.html")
        wb.SaveAs(html_path, FileFormat=44)
        wb.Close(False)
        excel.Quit()
        return True
    except Exception as e:
        try:
            wb.Close(False)
            excel.Quit()
        except:
            pass
        print("❌ SaveAs Webpage failed:", e)
        return False

# =========================
# STEP 2: COLLECT IMAGES
# =========================
def collect_images(web_dir, out_dir):
    safe_mkdir(out_dir)
    count = 0
    for root, _, files in os.walk(web_dir):
        for f in files:
            if is_image(f):
                src = os.path.join(root, f)
                dst = os.path.join(out_dir, f"image_{count:03d}.png")
                shutil.copy(src, dst)
                count += 1
    return count

# =========================
# STEP 3: CLEAN IMAGES
# =========================
def clean_images(folder):
    seen_md5 = {}
    removed = 0

    for fname in sorted(os.listdir(folder)):
        if not is_image(fname):
            continue

        lower_name = fname.lower()

        # 🛑 FORCE KEEP BY KEY
        if any(k.lower() in lower_name for k in FORCE_KEEP_KEYS):
            print(f"🟢 FORCE KEEP: {fname}")
            continue

        path = os.path.join(folder, fname)

        try:
            with Image.open(path) as img:
                w, h = img.size
        except:
            continue

        # 1️⃣ remove small images
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            key = os.path.basename(folder)
            if any(t in lower_name for t in SMALL_NAME_WHITELIST):
                continue
            try:
                shutil.copy2(path, os.path.join(SMALL_IMAGE_DIR, f"{key}_{fname}"))
            except:
                pass
            try:
                os.remove(path)
                removed += 1
                print(f"🗑️ small: {fname}")
            except:
                pass
            continue

        # 2️⃣ remove exact duplicate
        try:
            md5 = md5_of_file(path)
        except:
            continue

        if md5 in seen_md5:
            try:
                os.remove(path)
                removed += 1
                print(f"🗑️ exact dup: {fname}")
            except:
                pass
        else:
            seen_md5[md5] = fname

    return removed

# =========================
# MAIN
# =========================
safe_mkdir(TEMP_WEB_ROOT)
safe_mkdir(OUTPUT_ROOT)
safe_mkdir(SMALL_IMAGE_DIR)

with open(MAP_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print("🚀 START WEBPAGE EXTRACT + CLEAN")

for item in records:
    key = item.get("key")
    excel_path = item.get("file_path")

    print("\n" + "=" * 60)
    print("🔑 KEY:", key)

    if not excel_path or not os.path.exists(excel_path):
        print("⚠️ ไม่พบไฟล์")
        continue

    temp_dir = os.path.join(TEMP_WEB_ROOT, key)
    out_dir  = os.path.join(OUTPUT_ROOT, key)

    safe_mkdir(temp_dir)
    safe_mkdir(out_dir)

    if not save_excel_as_webpage(excel_path, temp_dir, key):
        continue

    total = collect_images(temp_dir, out_dir)
    print(f"🖼️ extracted images: {total}")

    removed = clean_images(out_dir)
    print(f"🧹 removed images: {removed}")

    remain = len([f for f in os.listdir(out_dir) if is_image(f)])
    print(f"✅ remain images: {remain}")

print("\n🎉 DONE")
print("📁 OUTPUT:", OUTPUT_ROOT)
