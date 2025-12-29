import os, json, shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

print("===================================")
print("🚀 STEP D APPLY ARROW MODEL START")
print("===================================")

# =========================
# CONFIG
# =========================
CODES_FILE = "sheet_data/codes_all.json"
IMAGE_ROOT = "final_images"
OUT_DIR = "images_by_key"
MODEL_PATH = "arrow_model.pth"

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# class ที่อนุญาต 2 รูป
ALLOW_TWO = {"กล่อง_train", "แบบฉลาก_train"}
# class ที่ต้องการให้เลือกอันแรกที่เจอเป็นอันดับแรก (ไม่ใช่โดยความมั่นใจ)
PRIORITIZE_FIRST = {"ฝา_train"}

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

ckpt = torch.load(MODEL_PATH, map_location=device)
classes = ckpt["classes"]  # ['arrow','no_arrow']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(ckpt["model"])
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_arrow(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()
        return classes[idx], float(probs[idx])
    except Exception:
        return None, 0.0

# =========================
# LOAD CODES
# =========================
with open(CODES_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

os.makedirs(OUT_DIR, exist_ok=True)

total = 0

for item in records:
    key = item["key"]
    codes = item["codes"]

    print(f"\n🔑 KEY: {key}")
    src_dir = os.path.join(IMAGE_ROOT, key)
    if not os.path.isdir(src_dir):
        print("⚠️ no image folder")
        continue

    # เก็บ candidate ต่อ class
    cand = {cls: {"normal": [], "arrow": []} for cls in codes.keys()}

    for img in os.listdir(src_dir):
        if not img.lower().endswith(IMAGE_EXTS):
            continue

        img_path = os.path.join(src_dir, img)
        label, conf = predict_arrow(img_path)
        if label is None:
            continue

        for cls in codes.keys():
            # ใช้โมเดลลูกศร “เฉพาะ” class ที่ต้องการ 2 รูป
            if cls in ALLOW_TWO:
                if label == "arrow":
                    cand[cls]["arrow"].append((img_path, conf))
                else:
                    cand[cls]["normal"].append((img_path, conf))
            else:
                # class ปกติ: เก็บเป็น normal อย่างเดียว
                cand[cls]["normal"].append((img_path, conf))

    out_dir = os.path.join(OUT_DIR, key)
    os.makedirs(out_dir, exist_ok=True)

    for cls, code in codes.items():
        picked = []

        # เลือก normal 1 รูป
        if cand[cls]["normal"]:
            # บาง class (เช่น ฝา) ต้องการให้ใช้รูปแรกที่เจอเป็นอันดับแรก
            if cls in PRIORITIZE_FIRST:
                picked.append(("normal", cand[cls]["normal"][0]))
            else:
                cand[cls]["normal"].sort(key=lambda x: x[1], reverse=True)
                picked.append(("normal", cand[cls]["normal"][0]))

        # เลือกลูกศรเพิ่ม ถ้าอนุญาต
        if cls in ALLOW_TWO and cand[cls]["arrow"]:
            cand[cls]["arrow"].sort(key=lambda x: x[1], reverse=True)
            picked.append(("arrow", cand[cls]["arrow"][0]))

        for i, (kind, (src, conf)) in enumerate(picked, start=1):
            ext = os.path.splitext(src)[1]
            base = code
            # พยายามใช้ชื่อปกติก่อน ถ้ามีไฟล์อยู่แล้ว ให้เติม suffix -1, -2 ไล่ไป
            name = f"{base}{ext}"
            dst = os.path.join(out_dir, name)
            n = 1
            while os.path.exists(dst):
                name = f"{base}-{n}{ext}"
                dst = os.path.join(out_dir, name)
                n += 1
            shutil.copy2(src, dst)
            total += 1
            print(f"✅ {cls} → {name} ({kind}, conf={conf:.2f})")

print("\n🎉 STEP D COMPLETE")
print("📁 created files:", total)
