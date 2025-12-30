import os, json, shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

print("="*60)
print(" STEP D – SI0 FINAL PIPELINE (ARROW = KEY, FIX LIST NAME)")
print("="*60)

# =========================
# CONFIG
# =========================
IMAGE_ROOT = "final_images"
OUT_DIR = "images_by_key"
CODES_FILE = "sheet_data/codes_all.json"

CLASSIFIER_PATH = "packaging_classifier.pth"
ARROW_MODEL_PATH = "arrow_model.pth"

BOTTLE_FRONT_CLASS = "SI0_front_train"
BOTTLE_BACK_CLASS  = "SI0_back_train"

BOTTLE_CLASS   = "ขวด_train"
BOX_CLASS      = "กล่อง_train"
TRAY_BOX_CLASS = "แบบถาด-กล่อง_train"
STICKER_CLASS  = "สติ๊กเกอร์_train"
DISCARD_CLASS  = "discard_train"

FINAL_CLASS_ALIASES = ("รูปสินค้าสำเร็จ_train", "รูปสินค้าสำเร็จ_trin")
EXTS = (".png", ".jpg", ".jpeg")

PAIR_ORDER = "front_back"
CLEAN_OUTPUT_PER_KEY = True

# =========================
# UTILS
# =========================
def resolve_filename(dst, base, ext=".png"):
    name = f"{base}{ext}"
    i = 1
    while os.path.exists(os.path.join(dst, name)):
        name = f"{base}-{i}{ext}"
        i += 1
    return name


def merge_side_by_side(paths):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = max(im.height for im in imgs)

    resized = []
    for im in imgs:
        if im.height != h:
            r = h / im.height
            im = im.resize((int(im.width * r), h), Image.LANCZOS)
        resized.append(im)

    w = sum(im.width for im in resized)
    canvas = Image.new("RGB", (w, h), (255,255,255))
    x = 0
    for im in resized:
        canvas.paste(im, (x,0))
        x += im.width
    return canvas


def ordered_pair(front, back):
    return [front, back] if PAIR_ORDER == "front_back" else [back, front]


def first_code(value, fallback):
    """🔒 GUARANTEE STRING (NO LIST EVER)"""
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return fallback
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback

# =========================
# LOAD MODELS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CLASSIFIER_PATH, map_location=device)
idx_to_class = ckpt["idx_to_class"]

clf = models.resnet18(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(ckpt["class_to_idx"]))
clf.load_state_dict(ckpt["model_state"])
clf.to(device).eval()

arrow_model = None
arrow_classes = ["arrow", "no_arrow"]
if os.path.exists(ARROW_MODEL_PATH):
    a = torch.load(ARROW_MODEL_PATH, map_location=device)
    arrow_classes = a["classes"]
    arrow_model = models.resnet18(weights=None)
    arrow_model.fc = nn.Linear(arrow_model.fc.in_features, len(arrow_classes))
    arrow_model.load_state_dict(a["model"])
    arrow_model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Arrow model was trained with ImageNet normalization; keep this separate so we don't
# accidentally change packaging-classifier behavior.
arrow_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict_class(img):
    x = transform(img).unsqueeze(0).to(device)
    return idx_to_class[int(torch.softmax(clf(x),1).argmax())]

@torch.no_grad()
def has_arrow(img):
    if arrow_model is None:
        return False
    x = arrow_transform(img).unsqueeze(0).to(device)
    return arrow_classes[int(torch.softmax(arrow_model(x),1).argmax())] == "arrow"

# =========================
# LOAD CODES
# =========================
with open(CODES_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# PROCESS SI0
# =========================
for rec in records:
    key = rec["key"]
    if not key.startswith("SI0"):
        continue

    codes = rec["codes"]
    src_dir = os.path.join(IMAGE_ROOT, key)
    if not os.path.isdir(src_dir):
        continue

    out_dir = os.path.join(OUT_DIR, key)
    os.makedirs(out_dir, exist_ok=True)

    if CLEAN_OUTPUT_PER_KEY:
        for f in os.listdir(out_dir):
            try:
                os.remove(os.path.join(out_dir, f))
            except:
                pass

    bottle_base = first_code(codes.get(BOTTLE_CLASS), BOTTLE_CLASS)
    tray_base   = first_code(codes.get(TRAY_BOX_CLASS) or codes.get(BOX_CLASS), TRAY_BOX_CLASS)

    fronts, backs, boxes = [], [], []
    sticker_imgs, final_imgs = [], []

    # ---------- CLASSIFY ----------
    for fn in sorted(os.listdir(src_dir)):
        if not fn.lower().endswith(EXTS):
            continue

        p = os.path.join(src_dir, fn)
        img = Image.open(p).convert("RGB")

        cls = predict_class(img)
        arrow = has_arrow(img)

        if cls == DISCARD_CLASS:
            continue
        if cls in FINAL_CLASS_ALIASES:
            final_imgs.append(p)
        elif cls in (BOX_CLASS, TRAY_BOX_CLASS):
            boxes.append((p, arrow))
        elif cls == BOTTLE_FRONT_CLASS:
            fronts.append((p, arrow))
        elif cls == BOTTLE_BACK_CLASS:
            backs.append((p, arrow))
        elif cls == STICKER_CLASS:
            sticker_imgs.append(p)

    # ---------- FINAL PRODUCT ----------
    # ตั้งชื่อสินค้าสำเร็จให้เหมือนกันทุก key: ใช้ชื่อ key เสมอ
    final_code = key

    # ตั้งชื่อลูกศรให้เหมือนกันทุก key: ใช้ชื่อ key และให้ resolve_filename ใส่ -1/-2 เพื่อไม่ทับสินค้าสำเร็จ
    arrow_base = key

    if final_imgs and final_code:
        shutil.copy2(
            final_imgs[0],
            os.path.join(out_dir, resolve_filename(out_dir, final_code))
        )

    # ---------- BOX / TRAY ----------
    for (p, arrow) in boxes:
        name = arrow_base if arrow else tray_base
        shutil.copy2(
            p,
            os.path.join(out_dir, resolve_filename(out_dir, name))
        )

    # ---------- BOTTLE (MERGE) ----------
    pair_count = min(len(fronts), len(backs))
    for i in range(pair_count):
        (f_path, f_arrow) = fronts[i]
        (b_path, b_arrow) = backs[i]

        name = arrow_base if (f_arrow or b_arrow) else bottle_base

        merge_side_by_side(
            ordered_pair(f_path, b_path)
        ).save(
            os.path.join(out_dir, resolve_filename(out_dir, name))
        )

    # ---------- STICKER ----------
    sticker_raw = codes.get(STICKER_CLASS)
    if sticker_imgs and sticker_raw:
        for i, p in enumerate(sticker_imgs):
            base = first_code(
                sticker_raw[i] if isinstance(sticker_raw, list) and i < len(sticker_raw) else sticker_raw,
                STICKER_CLASS
            )
            shutil.copy2(
                p,
                os.path.join(out_dir, resolve_filename(out_dir, base))
            )

print("\n✅ PIPELINE COMPLETE – LIST NAME BUG FIXED 100%")
print("="*60)
