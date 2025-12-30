import os
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


def load_arrow_model(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    classes = ckpt["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    idx_to_class = {i: c for i, c in enumerate(classes)}
    return model, idx_to_class, tf


def load_side_model(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    idx_to_class = ckpt["idx_to_class"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    return model, idx_to_class, tf


def predict(model, tf, idx_to_class, img: Image.Image):
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    idx = int(probs.argmax())
    return idx_to_class[idx], float(probs[idx])


def main():
    device = "cpu"
    key = "SI0311700000270S131"
    root = os.path.join("final_images", key)

    arrow_model, arrow_idx, arrow_tf = load_arrow_model("arrow_model.pth", device=device)
    side_model, side_idx, side_tf = load_side_model("models/side_SI0_front_back.pth", device=device)

    rows = []
    for f in sorted(os.listdir(root)):
        if not f.lower().endswith(".png"):
            continue
        p = os.path.join(root, f)
        img = Image.open(p).convert("RGB")
        a, ac = predict(arrow_model, arrow_tf, arrow_idx, img)
        if a == "arrow":
            s, sc = predict(side_model, side_tf, side_idx, img)
            rows.append((f, round(ac, 4), s, round(sc, 4)))

    print("ARROW count", len(rows))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
