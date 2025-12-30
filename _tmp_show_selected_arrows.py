import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

KEY = "SI0311700000270S131"
ROOT = os.path.join("final_images", KEY)

BOTTLE_FRONT_CLASS = "SI0_front_train"
BOTTLE_BACK_CLASS = "SI0_back_train"
BOX_CLASS = "กล่อง_train"
TRAY_BOX_CLASS = "แบบถาด-กล่อง_train"
STICKER_CLASS = "สติ๊กเกอร์_train"
DISCARD_CLASS = "discard_train"
FINAL_CLASS_ALIASES = ("รูปสินค้าสำเร็จ_train", "รูปสินค้าสำเร็จ_trin")


def arrow_priority(cls_name: str) -> int:
    if cls_name in (BOTTLE_FRONT_CLASS, BOTTLE_BACK_CLASS):
        return 3
    if cls_name in (TRAY_BOX_CLASS, BOX_CLASS):
        return 2
    if cls_name == STICKER_CLASS:
        return 2
    if cls_name == DISCARD_CLASS:
        return 0
    return 1


# load packaging classifier
ckpt = torch.load("packaging_classifier.pth", map_location="cpu")
idx_to_class = ckpt["idx_to_class"]
clf = models.resnet18(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(ckpt["class_to_idx"]))
clf.load_state_dict(ckpt["model_state"])
clf.eval()

# load arrow model
a = torch.load("arrow_model.pth", map_location="cpu")
arrow_classes = a["classes"]
arrow = models.resnet18(weights=None)
arrow.fc = nn.Linear(arrow.fc.in_features, len(arrow_classes))
arrow.load_state_dict(a["model"])
arrow.eval()

# load side model
s = torch.load("models/side_SI0_front_back.pth", map_location="cpu")
side = models.resnet18(weights=None)
side.fc = nn.Linear(side.fc.in_features, len(s["idx_to_class"]))
side.load_state_dict(s["model_state"])
side.eval()

cls_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
arrow_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
side_tf = cls_tf


def predict_class(img):
    x = cls_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(clf(x), 1)[0]
    return idx_to_class[int(probs.argmax())]


def predict_arrow(img):
    x = arrow_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(arrow(x), 1)[0]
    idx = int(probs.argmax())
    return arrow_classes[idx], float(probs[idx])


def predict_side_probs(img):
    x = side_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(side(x), 1)[0]
    return float(probs[0]), float(probs[1])


arrow_candidates = []
for fn in sorted(os.listdir(ROOT)):
    if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    p = os.path.join(ROOT, fn)
    img = Image.open(p).convert("RGB")
    cls = predict_class(img)
    if cls in FINAL_CLASS_ALIASES or cls == DISCARD_CLASS:
        continue
    al, aconf = predict_arrow(img)
    if al != "arrow":
        continue
    front_prob, back_prob = predict_side_probs(img)

    # override for bottle class
    if cls == BOTTLE_FRONT_CLASS:
        front_prob, back_prob = 1.0, 0.0
    elif cls == BOTTLE_BACK_CLASS:
        front_prob, back_prob = 0.0, 1.0

    prio = arrow_priority(cls)
    arrow_candidates.append((prio, aconf, front_prob, back_prob, fn, cls))

print("arrow_candidates", len(arrow_candidates))
for c in sorted(arrow_candidates, key=lambda t: (-t[0], -t[1])):
    print(c)

if arrow_candidates:
    bf = max(arrow_candidates, key=lambda t: (t[0], t[1], t[2]))
    bb = max(arrow_candidates, key=lambda t: (t[0], t[1], t[3]))
    if bf[4] == bb[4]:
        cand_without = [c for c in arrow_candidates if c[4] != bf[4]]
        bb = max(cand_without, key=lambda t: (t[0], t[1], t[3])) if cand_without else None

    print("\nSELECTED FRONT:", bf)
    print("SELECTED BACK:", bb)
