import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms

INPUT_DIR = "../Galaxy10_Images"
OUTPUT_DIR = "../Galaxy10_Embedding_vectors"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "galaxy10_resnet101_embeddings.npz")
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = models.resnet101(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
model.eval().to(DEVICE)

def extract_feature(x):
    with torch.no_grad():
        return model(x).squeeze(-1).squeeze(-1)

def process_split(split):
    split_path = os.path.join(INPUT_DIR, split)
    class_names = sorted(os.listdir(split_path))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    features, labels, paths = [], [], []

    for class_name in class_names:
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for fname in tqdm(os.listdir(class_path), desc=f"{split} - {class_name}"):
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                feat = extract_feature(img_tensor).squeeze(0).cpu().numpy()
                features.append(feat)
                labels.append(class_to_idx[class_name])
                paths.append(fpath)
            except Exception as e:
                print(f"Failed to process {fpath}: {e}")
    return np.array(features), np.array(labels), np.array(paths), class_names

data = {}
for split in ['train', 'val', 'test']:
    feats, lbls, pths, classes = process_split(split)
    data[f"{split}_features"] = feats
    data[f"{split}_labels"] = lbls
    data[f"{split}_paths"] = pths

data["class_names"] = classes
np.savez_compressed(OUTPUT_FILE, **data)
print(f"Saved to {OUTPUT_FILE}")