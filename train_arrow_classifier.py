import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

# =========================
# CONFIG
# =========================
DATA_DIR = "arrow_seed"
MODEL_OUT = "arrow_model.pth"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
VAL_SPLIT = 0.15

SEED = 42
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# DATA
# =========================
weights = models.ResNet18_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

full = ImageFolder(DATA_DIR, transform=train_transform)

val_size = max(1, int(len(full) * VAL_SPLIT))
train_size = len(full) - val_size
train_ds, val_ds = random_split(full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

# ให้ val ใช้ transform ของ val (ไม่ใช่ augment)
val_ds.dataset = ImageFolder(DATA_DIR, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", full.classes)
print("Total images:", len(full))
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
print("Normalize mean/std:", mean, std)

# =========================
# MODEL (ResNet18)
# =========================
model = models.resnet18(weights=weights)
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

    correct = 0
    seen = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(1)
        correct += int((preds == labels).sum().item())
        seen += int(labels.numel())

    train_acc = (correct / max(1, seen))

    # quick val
    model.eval()
    v_correct = 0
    v_seen = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            v_correct += int((preds == labels).sum().item())
            v_seen += int(labels.numel())
    val_acc = (v_correct / max(1, v_seen))

    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.4f} - train_acc: {train_acc:.3f} - val_acc: {val_acc:.3f}")

# =========================
# SAVE
# =========================
torch.save({
    "model": model.state_dict(),
    "classes": full.classes,
    "normalize": {"mean": mean, "std": std},
}, MODEL_OUT)

print("🎉 Training complete → saved:", MODEL_OUT)
