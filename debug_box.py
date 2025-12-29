import os
import torch
from torchvision import models, transforms
from PIL import Image

ckpt = torch.load('packaging_classifier.pth', map_location='cpu')
idx_to_class = {v:k for k,v in ckpt['class_to_idx'].items()}
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt['class_to_idx']))
model.load_state_dict(ckpt['model_state'])
model.eval()

arrow_ckpt = torch.load('arrow_model.pth', map_location='cpu')
arrow_model = models.resnet18(weights=None)
arrow_model.fc = torch.nn.Linear(arrow_model.fc.in_features, len(arrow_ckpt['classes']))
arrow_model.load_state_dict(arrow_ckpt['model'])
arrow_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

root = 'final_images/BA12AOA0222041106'
for fn in sorted(os.listdir(root)):
    if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    path = os.path.join(root, fn)
    img = Image.open(path).convert('RGB')
    with torch.no_grad():
        p = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)[0]
        cls = idx_to_class[int(p.argmax())]
        arrow_p = torch.softmax(arrow_model(transform(img).unsqueeze(0)), dim=1)[0]
        arrow = arrow_ckpt['classes'][int(arrow_p.argmax())]
    print(fn, cls, f"{float(p[int(p.argmax())]):.3f}", arrow, f"{float(arrow_p[int(arrow_p.argmax())]):.3f}")
