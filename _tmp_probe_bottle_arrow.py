import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

KEY = "SI0311700000270S131"
ROOT = os.path.join("final_images", KEY)

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

cls_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
arrow_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

BOTTLE_FRONT_CLASS = "SI0_front_train"
BOTTLE_BACK_CLASS = "SI0_back_train"


def predict_class(img):
    x = cls_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(clf(x), 1)[0]
    return idx_to_class[int(probs.argmax())], float(probs.max())


def predict_arrow(img):
    x = arrow_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(arrow(x), 1)[0]
    idx = int(probs.argmax())
    return arrow_classes[idx], float(probs[idx])


rows = []
for f in sorted(os.listdir(ROOT)):
    if not f.lower().endswith(".png"):
        continue
    p = os.path.join(ROOT, f)
    img = Image.open(p).convert("RGB")
    cls, cconf = predict_class(img)
    if cls not in (BOTTLE_FRONT_CLASS, BOTTLE_BACK_CLASS):
        continue
    al, aconf = predict_arrow(img)
    rows.append((f, cls, round(cconf, 4), al, round(aconf, 4)))

print("BOTTLE imgs", len(rows))
for r in rows:
    print(r)
