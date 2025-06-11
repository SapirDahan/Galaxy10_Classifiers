import h5py
import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split

# Mapping numeric labels to class names
label_names = {
    0: "round_smooth",
    1: "inbetween_smooth",
    2: "cigar_smooth",
    3: "edgeon_no_bulge",
    4: "edgeon_with_bulge",
    5: "spiral_tight",
    6: "spiral_medium",
    7: "spiral_loose",
    8: "barred_spiral",
    9: "merger"
}

# Set random seed for reproducibility
random.seed(42)

# Create output directory one level above script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
output_dir = os.path.join(parent_dir, 'Galaxy10_Images')
os.makedirs(output_dir, exist_ok=True)

# Load images and labels from Galaxy10.h5
with h5py.File(os.path.join(script_dir, 'Galaxy10.h5'), 'r') as f:
    images = f['images'][:]
    labels = f['ans'][:]

# Prepare (image_index, label) tuples
all_samples = list(zip(range(len(images)), labels))

# Split into train (70%), val (15%), test (15%)
train_val, test = train_test_split(all_samples, test_size=0.15, stratify=labels, random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, stratify=[label for _, label in train_val], random_state=42)  # 0.1765 ≈ 15% of total

# Helper function to save images to folders
def save_split(split_name, data):
    for i, label in data:
        class_name = label_names[int(label)]
        split_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(split_dir, exist_ok=True)
        img = Image.fromarray(images[i])
        filename = f'{class_name}_{i:05d}.png'
        img.save(os.path.join(split_dir, filename))

# Save all splits
save_split('train', train)
save_split('val', val)
save_split('test', test)