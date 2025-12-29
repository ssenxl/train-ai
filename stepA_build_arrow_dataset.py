import os
import shutil
import torch
import clip
from PIL import Image

print(" BUILD ARROW DATASET START")

IMAGE_ROOT = "final_images"
OUT_ROOT = "arrow_dataset"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

TEXT_CLASSES = [
    "ฉลากสนคาแบบปกต ไมมลกศร",
    "ฉลากสนคาทมลกศรหรอสญลกษณช",
    "กลองบรรจภณฑแบบปกต ไมมลกศร",
    "กลองบรรจภณฑทมลกศรหรอสญลกษณช"
]

CLASS_FOLDER = {
    0: "label_normal",
    1: "label_arrow",
    2: "box_normal",
    3: "box_arrow"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)
text_tokens = clip.tokenize(TEXT_CLASSES).to(device)

for folder in CLASS_FOLDER.values():
    os.makedirs(os.path.join(OUT_ROOT, folder), exist_ok=True)

total = 0

for key in os.listdir(IMAGE_ROOT):
    src_dir = os.path.join(IMAGE_ROOT, key)
    if not os.path.isdir(src_dir):
        continue

    for img in os.listdir(src_dir):
        if not img.lower().endswith(IMAGE_EXTS):
            continue

        img_path = os.path.join(src_dir, img)

        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        except Exception:
            continue

        with torch.no_grad():
            logits, _ = model(image, text_tokens)
            probs = logits.softmax(dim=-1)[0].cpu().numpy()

        idx = probs.argmax()
        folder = CLASS_FOLDER[idx]

        dst = os.path.join(OUT_ROOT, folder, f"{key}_{img}")
        shutil.copy2(img_path, dst)
        total += 1

print(" DONE, total images:", total)
