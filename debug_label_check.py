import os, torch
from torchvision import models, transforms
from PIL import Image

def load_model(path, model_key='model_state'):
    ckpt = torch.load(path, map_location='cpu')
    if model_key == 'model':
        classes = ckpt['classes']
    else:
        class_to_idx = ckpt['class_to_idx']
        classes = {v: k for k, v in class_to_idx.items()}
    idx_to_class = {v: k for k, v in classes.items()} if isinstance(classes, dict) else None
    model = models.resnet18(weights=None)
    out_dim = len(classes) if isinstance(classes, (list, dict)) else len(class_to_idx)
    model.fc = torch.nn.Linear(model.fc.in_features, out_dim)
    model.load_state_dict(ckpt[model_key])
    model.eval()
    return model, idx_to_class, classes

model, idx_to_class, _ = load_model('packaging_classifier.pth')
arrow_model, _, arrow_classes = load_model('arrow_model.pth', model_key='model')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(model, idx_to_class, path):
    img = Image.open(path).convert('RGB')
    with torch.no_grad():
        p = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)[0]
    idx = int(p.argmax())
    if idx_to_class:
        label = idx_to_class.get(idx, f"cls_{idx}")
    else:
        label = arrow_classes[idx]
    return label, float(p[idx])

root = 'final_images/GA01AYIS36001301066'
for fn in sorted(os.listdir(root)):
    if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    path = os.path.join(root, fn)
    cls, conf = predict(model, idx_to_class, path)
    arrow_cls, arrow_conf = predict(arrow_model, None, path)
    print(fn, cls, f"cls_conf={conf:.3f}", arrow_cls, f"arrow_conf={arrow_conf:.3f}")
