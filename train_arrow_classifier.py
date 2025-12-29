import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# =========================
# CONFIG
# =========================
DATA_DIR = "arrow_seed"
MODEL_OUT = "arrow_model.pth"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# DATA
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

# =========================
# MODEL (ResNet18)
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.4f}")

# =========================
# SAVE
# =========================
torch.save({
    "model": model.state_dict(),
    "classes": dataset.classes
}, MODEL_OUT)

print("🎉 Training complete → saved:", MODEL_OUT)
