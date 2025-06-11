import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from collections import Counter
import random

# Paths and constants
INPUT_DIR = "../Galaxy10_Images"
OUTPUT_DIR = "../Galaxy10_Embedding_vectors"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "galaxy10_vit_embeddings_augmented_balanced.npz")
IMAGE_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIXUP_ALPHA = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base Transform (no augmentation)
base_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Strong Augmentation Transform for training
strong_aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=20, shear=10, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def simple_augmentations(image):
    operations = [
        lambda x: x,
        lambda x: ImageOps.mirror(x),
        lambda x: ImageOps.flip(x),
        lambda x: x.rotate(90),
        lambda x: x.rotate(180),
        lambda x: x.rotate(270)
    ]
    return random.choice(operations)(image)

class GalaxyDataset(Dataset):
    def __init__(self, img_paths, labels, transform, augment=False):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.augment:
            img = simple_augmentations(img)
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

def load_data(split):
    split_path = os.path.join(INPUT_DIR, split)
    class_names = sorted(os.listdir(split_path))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    img_paths, labels = [], []

    for class_name in class_names:
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            img_paths.append(fpath)
            labels.append(class_to_idx[class_name])

    return img_paths, labels, class_names

def prepare_balanced_train_set(orig_img_paths, orig_labels):
    counter = Counter(orig_labels)
    max_count = max(counter.values())

    new_img_paths = orig_img_paths.copy()
    new_labels = orig_labels.copy()

    augmented_imgs, augmented_labels = [], []

    for class_idx in counter:
        num_to_add = max_count - counter[class_idx]
        class_imgs = [img for img, lbl in zip(orig_img_paths, orig_labels) if lbl == class_idx]
        for _ in range(num_to_add):
            augmented_imgs.append(random.choice(class_imgs))
            augmented_labels.append(class_idx)

    return new_img_paths + augmented_imgs, new_labels + augmented_labels

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Fixed Feature Extractor for TorchVision ViT
class ViTFeatureExtractor(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.model = vit_model

    def forward(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)
        x = self.model.encoder.layers(x)
        x = self.model.encoder.ln(x)
        cls_token_final = x[:, 0]
        return cls_token_final

def extract_features(img_paths, labels, feature_extractor, augment_flags):
    dataset = GalaxyDataset(img_paths, labels, base_transform, augment=augment_flags)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    features, labels_out = [], []

    for inputs, lbls in tqdm(dataloader, desc=f"Extracting features"):
        inputs = inputs.to(DEVICE)
        feats = feature_extractor(inputs).detach().cpu().numpy()
        features.append(feats)
        labels_out.append(lbls.numpy())

    features = np.concatenate(features, axis=0)
    labels_out = np.concatenate(labels_out, axis=0)
    return features, labels_out

def main():
    train_img_paths, train_labels, class_names = load_data('train')
    val_img_paths, val_labels, _ = load_data('val')
    test_img_paths, test_labels, _ = load_data('test')

    balanced_img_paths, balanced_labels = prepare_balanced_train_set(train_img_paths, train_labels)

    train_dataset = GalaxyDataset(balanced_img_paths, balanced_labels, strong_aug_transform, augment=True)
    val_dataset = GalaxyDataset(val_img_paths, val_labels, base_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = models.vit_b_16(weights='IMAGENET1K_V1')
    model.encoder.dropout.p = 0.2

    num_ftrs = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)

            optimizer.zero_grad()
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (lam * preds.eq(targets_a).sum().item() + (1 - lam) * preds.eq(targets_b).sum().item())
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=(running_loss / total), acc=(correct / total))

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Val   | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

        scheduler.step(val_loss)

    print("Training completed.")

    feature_extractor = ViTFeatureExtractor(model)
    feature_extractor.eval().to(DEVICE)

    data = {}
    print("Extracting features for train set...")
    feats_train, lbls_train = extract_features(train_img_paths, train_labels, feature_extractor, augment_flags=False)
    data["train_features"] = feats_train
    data["train_labels"] = lbls_train

    print("Extracting features for val set...")
    feats_val, lbls_val = extract_features(val_img_paths, val_labels, feature_extractor, augment_flags=False)
    data["val_features"] = feats_val
    data["val_labels"] = lbls_val

    print("Extracting features for test set...")
    feats_test, lbls_test = extract_features(test_img_paths, test_labels, feature_extractor, augment_flags=False)
    data["test_features"] = feats_test
    data["test_labels"] = lbls_test

    data["class_names"] = class_names
    np.savez_compressed(OUTPUT_FILE, **data)
    print(f"Saved fine-tuned embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()