import os
import json
import shutil
import torch
import clip
from PIL import Image

print("===================================")
print("🚀 STEP 5 AI CLASSIFICATION START")
print("===================================")

# =========================
# CONFIG
# =========================
CODES_FILE = "sheet_data/codes_all.json"
IMAGE_ROOT = "final_images"
OUT_DIR = "images_by_key"

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

TEXT_CLASSES = [
    "ขวดสินค้า",
    "ฝาขวด",
    "ฟอยด์ปิดฝา",
    "ฉลากสินค้า",
    "สติ๊กเกอร์สินค้า",
    "กล่องบรรจุภัณฑ์"
]

CLASS_MAP = {
    "ขวดสินค้า": "ขวด_train",
    "ฝาขวด": "ฝา_train",
    "ฟอยด์ปิดฝา": "ฟอยด์_train",
    "ฉลากสินค้า": "แบบฉลาก_train",
    "สติ๊กเกอร์สินค้า": "สติ๊กเกอร์_train",
    "กล่องบรรจุภัณฑ์": "กล่อง_train"
}

# =========================
# CHECK INPUT FILES
# =========================
print("🔍 Checking input files...")

if not os.path.exists(CODES_FILE):
    print(f"❌ ไม่พบไฟล์ {CODES_FILE}")
    exit(1)

if not os.path.isdir(IMAGE_ROOT):
    print(f"❌ ไม่พบโฟลเดอร์ {IMAGE_ROOT}")
    exit(1)

# =========================
# LOAD MODEL
# =========================
print("🤖 Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("   Device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)
text_tokens = clip.tokenize(TEXT_CLASSES).to(device)

print("✅ CLIP model loaded")

# =========================
# LOAD CODES
# =========================
print("📄 Loading codes_all.json...")

with open(CODES_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"📊 Total records: {len(records)}")

if not records:
    print("❌ codes_all.json ว่าง ไม่มีข้อมูล")
    exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# PROCESS
# =========================
total_images = 0
total_saved = 0

for item in records:
    key = item.get("key")
    codes = item.get("codes", {})

    print("\n-----------------------------------")
    print(f"🔑 KEY: {key}")

    src_dir = os.path.join(IMAGE_ROOT, key)
    print("📂 Image folder:", src_dir)

    if not os.path.isdir(src_dir):
        print("⚠️ ไม่พบโฟลเดอร์รูป → ข้าม")
        continue

    images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    print(f"🖼️ Found images: {len(images)}")

    if not images:
        print("⚠️ ไม่มีไฟล์รูป → ข้าม")
        continue

    out_dir = os.path.join(OUT_DIR, key)
    os.makedirs(out_dir, exist_ok=True)

    for img_name in images:
        total_images += 1
        img_path = os.path.join(src_dir, img_name)

        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        except Exception as e:
            print(f"❌ เปิดรูปไม่ได้: {img_name} → {e}")
            continue

        with torch.no_grad():
            logits, _ = model(image, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        best_idx = probs.argmax()
        confidence = probs[best_idx]
        ai_label = TEXT_CLASSES[best_idx]
        cls = CLASS_MAP[ai_label]

        print(f"🤖 {img_name} → AI={cls} ({confidence:.2f})")

        if cls not in codes:
            print("⚠️ ไม่มี code สำหรับ class นี้ → ข้าม")
            continue

        code = codes[cls]
        ext = os.path.splitext(img_name)[1]

        dst = os.path.join(out_dir, code + ext)

        # กัน overwrite
        if os.path.exists(dst):
            base = code
            i = 2
            while os.path.exists(os.path.join(out_dir, f"{base}_{i}{ext}")):
                i += 1
            dst = os.path.join(out_dir, f"{base}_{i}{ext}")

        shutil.copy2(img_path, dst)
        total_saved += 1

        print(f"✅ saved → {os.path.basename(dst)}")

# =========================
# SUMMARY
# =========================
print("\n===================================")
print("🎉 STEP 5 AI CLASSIFICATION COMPLETE")
print("===================================")
print(f"🖼️ Total images scanned: {total_images}")
print(f"📁 Total images saved: {total_saved}")
print(f"📂 Output folder: {OUT_DIR}")
