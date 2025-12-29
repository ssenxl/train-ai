import os, json, shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import defaultdict
from PIL import Image

# =========================
# CONFIG
# =========================
IMAGE_ROOT = "final_images"
OUT_DIR = "images_by_key"
CODES_FILE = "sheet_data/codes_all.json"

CLASSIFIER_PATH = "packaging_classifier.pth"
ARROW_MODEL_PATH = "arrow_model.pth"

CAN_CLASS = "กระป๋อง_train"
BOTTLE_CLASS = "ขวด_train"
LABEL_CLASS = "แบบฉลาก_train"

BOX_PRED_CLASS_TO_CANONICAL = {
    "กล่อง_train": "กล่อง_train",
    "กล่องและถาด_train": "กล่องและถาด_train",
    "แบบถาด-กล่อง_train": "แบบถาด-กล่อง_train",
}

BOX_CLASSES = set(BOX_PRED_CLASS_TO_CANONICAL.keys())
BOX_CANONICAL_CLASSES = set(BOX_PRED_CLASS_TO_CANONICAL.values())

FINAL_CLASS = "รูปสินค้าสำเร็จ_train"
DISCARD_CLASS = "discard_train"

EXTS = (".png", ".jpg", ".jpeg")

UNCERTAIN_THRESHOLD = 0.5
UNCERTAIN_DIR = "uncertain_images"

CLASS_EQUIV = {
    BOTTLE_CLASS: CAN_CLASS,
    CAN_CLASS: BOTTLE_CLASS,
}

# =========================
# UTILS
# =========================

def resolve_filename(dst_dir, base, ext=".png"):
    name = f"{base}{ext}"
    idx = 1
    while os.path.exists(os.path.join(dst_dir, name)):
        name = f"{base}-{idx}{ext}"
        idx += 1
    return name


def merge_images_vertically(paths):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    widths = [img.width for img in imgs]
    heights = [img.height for img in imgs]
    total_h = sum(heights)
    max_w = max(widths)
    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for img in imgs:
        canvas.paste(img, (0, y))
        y += img.height
    return canvas

# =========================
# LOAD MODELS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CLASSIFIER_PATH, map_location=device)
class_to_idx = ckpt["class_to_idx"]
idx_to_class = ckpt.get("idx_to_class", {v: k for k, v in class_to_idx.items()})

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()

arrow_ckpt = torch.load(ARROW_MODEL_PATH, map_location=device)
arrow_classes = arrow_ckpt["classes"]

arrow_model = models.resnet18(weights=None)
arrow_model.fc = nn.Linear(arrow_model.fc.in_features, len(arrow_classes))
arrow_model.load_state_dict(arrow_ckpt["model"])
arrow_model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_class(img):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    idx = int(p.argmax())
    return idx_to_class[idx], float(p[idx])

def predict_arrow(img):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(arrow_model(x), dim=1)[0]
    idx = int(p.argmax())
    return arrow_classes[idx], float(p[idx])


def load_records():
    with open(CODES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(UNCERTAIN_DIR, exist_ok=True)


def run_pipeline(records, key_filter, allowed_classes=None, merge_classes=None):
    ensure_dirs()
    created = 0
    for rec in records:
        key = rec["key"]
        if not key_filter(key):
            continue
        codes = rec["codes"]

        src_dir = os.path.join(IMAGE_ROOT, key)
        if not os.path.isdir(src_dir):
            continue

        out_dir = os.path.join(OUT_DIR, key)
        os.makedirs(out_dir, exist_ok=True)

        bottle_items = defaultdict(list)
        label_records = []
        sticker_images = []
        box_normal = {}
        box_arrow = {}
        final_imgs = []
        other_items = {}
        other_conf = {}

        # ---------- scan images ----------
        for fn in sorted(os.listdir(src_dir)):
            if not fn.lower().endswith(EXTS):
                continue

            path = os.path.join(src_dir, fn)
            img = Image.open(path).convert("RGB")

            cls, conf = predict_class(img)
            if allowed_classes is not None and cls not in allowed_classes:
                continue

            if conf < UNCERTAIN_THRESHOLD:
                fname = resolve_filename(UNCERTAIN_DIR, f"{key}_{fn}")
                shutil.copy2(path, os.path.join(UNCERTAIN_DIR, fname))

            if cls == DISCARD_CLASS:
                continue

            if cls == FINAL_CLASS:
                final_imgs.append(path)
                continue

            if cls in {BOTTLE_CLASS, CAN_CLASS}:
                bottle_items[cls].append((conf, path))
                continue

            if cls == LABEL_CLASS:
                arrow_tag, _ = predict_arrow(img)
                label_records.append((path, arrow_tag))
                continue

            if cls == "สติ๊กเกอร์_train":
                sticker_images.append(path)
                continue

            if cls in BOX_CLASSES:
                canon = BOX_PRED_CLASS_TO_CANONICAL[cls]
                arrow_tag, _ = predict_arrow(img)
                if arrow_tag == "arrow":
                    box_arrow.setdefault(canon, []).append(path)
                else:
                    box_normal.setdefault(canon, path)
                continue

            if cls not in other_items or conf > other_conf.get(cls, 0):
                other_items[cls] = path
                other_conf[cls] = conf

        # ---------- FINAL ----------
        if final_imgs:
            shutil.copy2(final_imgs[0], os.path.join(out_dir, resolve_filename(out_dir, key)))
            created += 1

        # ---------- BOTTLE / CAN ----------
        for cls in (BOTTLE_CLASS, CAN_CLASS):
            code = codes.get(cls) or codes.get(CLASS_EQUIV.get(cls))
            entries = bottle_items.get(cls)
            if not code or not entries:
                continue
            if merge_classes and cls in merge_classes:
                paths = [p for _, p in sorted(entries, key=lambda item: item[0], reverse=True)]
                canvas = merge_images_vertically(paths)
                canvas_path = os.path.join(out_dir, resolve_filename(out_dir, code))
                canvas.save(canvas_path)
                created += 1
            else:
                best = sorted(entries, key=lambda item: item[0], reverse=True)[0][1]
                shutil.copy2(best, os.path.join(out_dir, resolve_filename(out_dir, code)))
                created += 1

        # ---------- LABEL ----------
        label_code = codes.get("แบบฉลาก_train")
        if label_code:
            normals = [p for p, a in label_records if a != "arrow"]
            arrows = [p for p, a in label_records if a == "arrow"]

            if normals:
                shutil.copy2(normals[0], os.path.join(out_dir, resolve_filename(out_dir, label_code)))
                created += 1

            for p in arrows:
                shutil.copy2(p, os.path.join(out_dir, resolve_filename(out_dir, key)))
                created += 1

        # ---------- STICKER (FIXED) ----------
        sticker_code = codes.get("สติ๊กเกอร์_train")
        if sticker_images and sticker_code:
            if isinstance(sticker_code, list):
                if len(sticker_images) >= 1:
                    shutil.copy2(
                        sticker_images[0],
                        os.path.join(out_dir, resolve_filename(out_dir, sticker_code[0]))
                    )
                    created += 1
                if len(sticker_images) >= 2:
                    shutil.copy2(
                        sticker_images[-1],
                        os.path.join(out_dir, resolve_filename(out_dir, sticker_code[-1]))
                    )
                    created += 1
            else:
                shutil.copy2(
                    sticker_images[0],
                    os.path.join(out_dir, resolve_filename(out_dir, sticker_code))
                )
                created += 1

        # ---------- BOX ----------
        for canon in BOX_CANONICAL_CLASSES:
            code = codes.get(canon)
            if not code:
                continue
            if canon in box_normal:
                shutil.copy2(box_normal[canon], os.path.join(out_dir, resolve_filename(out_dir, code)))
                created += 1
            for p in box_arrow.get(canon, []):
                shutil.copy2(p, os.path.join(out_dir, resolve_filename(out_dir, key)))
                created += 1

        # ---------- OTHER ----------
        for cls, p in other_items.items():
            code = codes.get(cls)
            if code:
                shutil.copy2(p, os.path.join(out_dir, resolve_filename(out_dir, code)))
                created += 1

    return created


if __name__ == "__main__":
    print("="*40)
    print(" STEP D FINAL PIPELINE START")
    print("="*40)

    records = load_records()
    created = run_pipeline(records, lambda key: not key.startswith("SI0"))

    print("\n STEP D FINAL COMPLETE")
    print(" created files:", created)
    print("="*40)
