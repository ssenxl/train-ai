import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_DIR = "dataset/train"   # มี SI0_front_train / SI0_back_train
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
MODEL_OUT = "models/side_SI0_front_back.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# DATASET (เฉพาะ front / back)
# =========================
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# 🔥 กรองเฉพาะ 2 class ที่ต้องการ
target_classes = ["SI0_front_train", "SI0_back_train"]

indices = [
    i for i, (_, y) in enumerate(full_dataset.samples)
    if full_dataset.classes[y] in target_classes
]

dataset = torch.utils.data.Subset(full_dataset, indices)

# map class ใหม่ให้เหลือแค่ front / back
class_to_idx = {
    "front": 0,
    "back": 1
}

def remap_label(y):
    name = full_dataset.classes[y]
    return class_to_idx["front"] if name == "SI0_front_train" else class_to_idx["back"]

class RemapDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, remap_label(y)

dataset = RemapDataset(dataset)

print("✅ Training side model with classes:", list(class_to_idx.keys()))

# =========================
# SPLIT
# =========================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
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
# SAVE (เข้ากับ pipeline)
# =========================
os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state": model.state_dict(),
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()}
    },
    MODEL_OUT
)

print("\n✅ Side model training finished")
print("✅ Model saved as:", MODEL_OUT)
print("✅ Classes:", class_to_idx)