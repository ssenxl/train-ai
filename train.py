import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_DIR = "dataset/train"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
MODEL_OUT = "packaging_classifier.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# DATASET
# =========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_dl):.4f}")

# =========================
# SAVE (สำคัญ)
# =========================
torch.save(
    {
        "model_state": model.state_dict(),
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": {v: k for k, v in dataset.class_to_idx.items()}
    },
    MODEL_OUT
)

print("✅ Training finished")
print("✅ Model saved as:", MODEL_OUT)
