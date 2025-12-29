import os, json, shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

print("="*60)
print(" STEP D – SI0 FINAL PIPELINE (FINAL)")
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

LABEL_CLASS    = "แบบฉลาก_train"
BOX_CLASS      = "กล่อง_train"
TRAY_BOX_CLASS = "แบบถาด-กล่อง_train"
STICKER_CLASS  = "สติ๊กเกอร์_train"
DISCARD_CLASS  = "discard_train"

FINAL_CLASS_ALIASES = ("รูปสินค้าสำเร็จ_train", "รูปสินค้าสำเร็จ_trin")
EXTS = (".png", ".jpg", ".jpeg")

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


# 🔥 FIXED MERGE FUNCTION (normalize height only)
def merge_side_by_side(paths, bg_color=(255,255,255)):
    imgs = [Image.open(p).convert("RGB") for p in paths]

    target_h = max(im.height for im in imgs)
    norm_imgs = []

    for im in imgs:
        if im.height != target_h:
            ratio = target_h / im.height
            new_w = int(im.width * ratio)
            im = im.resize((new_w, target_h), Image.LANCZOS)
        norm_imgs.append(im)

    total_w = sum(im.width for im in norm_imgs)
    canvas = Image.new("RGB", (total_w, target_h), bg_color)

    x = 0
    for im in norm_imgs:
        canvas.paste(im, (x, 0))
        x += im.width

    return canvas


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

def predict_class(img):
    with torch.no_grad():
        x = transform(img).unsqueeze(0).to(device)
        return idx_to_class[int(torch.softmax(clf(x),1).argmax())]

def predict_arrow(img):
    if arrow_model is None:
        return "no_arrow"
    with torch.no_grad():
        x = transform(img).unsqueeze(0).to(device)
        return arrow_classes[int(torch.softmax(arrow_model(x),1).argmax())]


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

    bottle_fronts, bottle_backs = [], []

    label_arrow_fronts, label_arrow_backs = [], []
    label_plain_fronts, label_plain_backs = [], []

    box_plain, box_arrow = [], []
    sticker_imgs, final_imgs = [], []

    key_counter = 1

    # ---------- CLASSIFY ----------
    for fn in sorted(os.listdir(src_dir)):
        if not fn.lower().endswith(EXTS):
            continue

        p = os.path.join(src_dir, fn)
        img = Image.open(p).convert("RGB")

        cls = predict_class(img)
        has_arrow = (predict_arrow(img) == "arrow")

        if cls == DISCARD_CLASS:
            continue

        if cls in FINAL_CLASS_ALIASES:
            final_imgs.append(p)
            continue

        if cls in (BOX_CLASS, TRAY_BOX_CLASS):
            (box_arrow if has_arrow else box_plain).append(p)
            continue

        if cls in (BOTTLE_FRONT_CLASS, BOTTLE_BACK_CLASS):
            if has_arrow:
                (label_arrow_fronts if cls == BOTTLE_FRONT_CLASS else label_arrow_backs).append(p)
            else:
                (label_plain_fronts if cls == BOTTLE_FRONT_CLASS else label_plain_backs).append(p)

            (bottle_fronts if cls == BOTTLE_FRONT_CLASS else bottle_backs).append(p)
            continue

        if cls == STICKER_CLASS:
            sticker_imgs.append(p)

    # ---------- FINAL ----------
    final_code = next((codes.get(n) for n in FINAL_CLASS_ALIASES if n in codes), None)
    if final_code and final_imgs:
        shutil.copy2(final_imgs[0], os.path.join(out_dir, resolve_filename(out_dir, final_code)))

    # ---------- BOX ----------
    tray_code = codes.get(TRAY_BOX_CLASS)
    for p in box_plain:
        name = tray_code if tray_code else f"{key}-{key_counter}"
        shutil.copy2(p, os.path.join(out_dir, resolve_filename(out_dir, name)))
        if not tray_code:
            key_counter += 1

    for p in box_arrow:
        shutil.copy2(p, os.path.join(out_dir, resolve_filename(out_dir, f"{key}-{key_counter}")))
        key_counter += 1

    # ---------- LABEL / BOTTLE WITH ARROW (PRIORITY) ----------
    while label_arrow_fronts and label_arrow_backs:
        f = label_arrow_fronts.pop(0)
        b = label_arrow_backs.pop(0)
        merge_side_by_side([f, b]).save(
            os.path.join(out_dir, resolve_filename(out_dir, f"{key}-{key_counter}"))
        )
        key_counter += 1

    while label_arrow_backs:
        b = label_arrow_backs.pop(0)
        f = label_plain_fronts[0] if label_plain_fronts else None
        if f:
            merge_side_by_side([f, b]).save(
                os.path.join(out_dir, resolve_filename(out_dir, f"{key}-{key_counter}"))
            )
        else:
            shutil.copy2(b, os.path.join(out_dir, resolve_filename(out_dir, f"{key}-{key_counter}")))
        key_counter += 1

    while label_arrow_fronts:
        f = label_arrow_fronts.pop(0)
        b = label_plain_backs[0] if label_plain_backs else None
        if b:
            merge_side_by_side([f, b]).save(
                os.path.join(out_dir, resolve_filename(out_dir, f"{key}-{key_counter}"))
            )
        else:
            shutil.copy2(f, os.path.join(out_dir, resolve_filename(out_dir, f"{key}-{key_counter}")))
        key_counter += 1

    # ---------- LABEL / BOTTLE NO ARROW ----------
    parts = []
    if bottle_fronts: parts.append(bottle_fronts[0])
    if bottle_backs:  parts.append(bottle_backs[0])

    label_code = codes.get(LABEL_CLASS)
    if parts and label_code:
        if len(parts) == 2:
            merge_side_by_side(parts).save(
                os.path.join(out_dir, resolve_filename(out_dir, label_code))
            )
        else:
            shutil.copy2(parts[0], os.path.join(out_dir, resolve_filename(out_dir, label_code)))

    # ---------- STICKER ----------
    sticker_code = codes.get(STICKER_CLASS)
    if isinstance(sticker_code, list):
        for i, p in enumerate(sticker_imgs):
            base = sticker_code[min(i, len(sticker_code)-1)]
            shutil.copy2(p, os.path.join(out_dir, resolve_filename(out_dir, base)))
    elif sticker_code:
        for p in sticker_imgs:
            shutil.copy2(p, os.path.join(out_dir, resolve_filename(out_dir, sticker_code)))

print("\n✅ PIPELINE COMPLETE – MERGE SIZE FIXED, LOGIC INTACT")
print("="*60)
