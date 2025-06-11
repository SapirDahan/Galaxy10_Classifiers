import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import open_clip

# Define image directory and output path
image_root = '../Galaxy10_Images'
output_path = '../Galaxy10_Embedding_vectors/galaxy10_clip_embeddings.npz'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
model = model.to(device)
model.eval()

# Prepare data structure
features_by_split = {'train': [], 'val': [], 'test': []}
labels_by_split = {'train': [], 'val': [], 'test': []}

# Determine consistent class order
all_classes = sorted(os.listdir(os.path.join(image_root, 'train')))
class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)}

# Traverse dataset folders
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(image_root, split)
    for class_name in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = class_to_idx[class_name]
        for image_name in tqdm(os.listdir(class_dir), desc=f'{split}/{class_name}'):
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(image_tensor)
                feature = feature.cpu().squeeze().numpy()
            features_by_split[split].append(feature)
            labels_by_split[split].append(label)

# Convert to arrays and save
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.savez_compressed(
    output_path,
    train_features=np.array(features_by_split['train']),
    val_features=np.array(features_by_split['val']),
    test_features=np.array(features_by_split['test']),
    train_labels=np.array(labels_by_split['train']),
    val_labels=np.array(labels_by_split['val']),
    test_labels=np.array(labels_by_split['test']),
    class_names=np.array(all_classes)
)

print(f"Saved feature vectors to: {output_path}")