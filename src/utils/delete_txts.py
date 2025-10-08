import os

# Paths
images_path = "data/raw/all/images"
labels_path = "data/raw/all/labels"

# Get base filenames (without extensions)
image_files = {os.path.splitext(f)[0] for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
label_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]

# Track how many deleted
deleted_count = 0

for label in label_files:
    base_name = os.path.splitext(label)[0]
    if base_name not in image_files:
        file_path = os.path.join(labels_path, label)
        os.remove(file_path)
        deleted_count += 1
        print(f"Deleted: {file_path}")

print(f"\n✅ Done! Deleted {deleted_count} label files without matching images.")
