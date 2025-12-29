import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn

# =========================
# CONFIG
# =========================
DATA_DIR = "dataset/train"   # 👈 ใช้ชุดที่ต้องการวัด
MODEL_PATH = "packaging_classifier.pth"
BATCH_SIZE = 16

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# LOAD DATA
# =========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# LOAD MODEL
# =========================
ckpt = torch.load(MODEL_PATH, map_location="cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
model.load_state_dict(ckpt["model"])
model.eval()

# =========================
# EVALUATE
# =========================
correct = 0
total = 0

with torch.no_grad():
    for x, y in loader:
        outputs = model(x)
        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / total * 100
print(f"✅ Accuracy: {accuracy:.2f}%")
print("Total images:", total)
print("Classes:", dataset.classes)
