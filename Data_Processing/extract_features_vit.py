import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms
import timm  # pip install timm

# Configuration
INPUT_DIR = "../Galaxy10_Images"
OUTPUT_DIR = "../Galaxy10_Embedding_vectors"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "galaxy10_vit_embeddings.npz")
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Preprocessing (ViT requires 224x224, normalized to [-1, 1])
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Load pretrained ViT model
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()
model.to(DEVICE)

# Extract CLS token
def extract_cls_token(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output = model.forward_features(x)  # shape: [1, seq_len, dim]
    return output[:, 0]  # CLS token

# Process one split: train/val/test
def process_split(split: str):
    split_dir = os.path.join(INPUT_DIR, split)
    class_names = sorted(os.listdir(split_dir))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    features, labels, paths = [], [], []

    for class_name in class_names:
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for fname in tqdm(os.listdir(class_path), desc=f"{split} - {class_name}"):
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # shape: [1, 3, 224, 224]
                cls_embedding = extract_cls_token(img_tensor).squeeze(0).cpu().numpy()  # shape: [768]
                features.append(cls_embedding)
                labels.append(class_to_idx[class_name])
                paths.append(fpath)
            except Exception as e:
                print(f"Failed to process {fpath}: {e}")

    return np.array(features), np.array(labels), np.array(paths), class_names

# Run extraction for all splits
data = {}
for split in ['train', 'val', 'test']:
    feats, lbls, pths, classes = process_split(split)
    data[f"{split}_features"] = feats
    data[f"{split}_labels"] = lbls
    data[f"{split}_paths"] = pths

data["class_names"] = classes
np.savez_compressed(OUTPUT_FILE, **data)
print(f"Saved ViT embeddings to: {OUTPUT_FILE}")
