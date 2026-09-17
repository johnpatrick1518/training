"""
prepare_dataset.py
Converts the ThumbsUp_OpenPalm image folders into YOLO object detection format.
- Splits data 80/20 into train/val sets
- Auto-generates full-image bounding box labels (since images have no annotations)
- Classes: 0 = openPalm, 1 = thumbsUp
"""

import os
import sys
import shutil
import random
from pathlib import Path
from PIL import Image

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SOURCE_CLASSES = {
    "openPalm": 0,   # class index 0
    "thumbsUp": 1,   # class index 1
}
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)


def create_directory_structure():
    """Create YOLO-format directory structure."""
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print("[✓] Created dataset directory structure")


def generate_yolo_label(image_path, class_id, label_path):
    """
    Generate a YOLO-format label file with a full-image bounding box.
    YOLO format: <class> <x_center> <y_center> <width> <height>
    All values are normalized to [0, 1].
    
    We use a slightly padded bounding box (0.05 margin from edges) to avoid
    including too much background, centering on the hand gesture.
    """
    # For full-image bounding box: center = 0.5, size = 1.0
    # Using a slightly smaller box to better frame the hand
    margin = 0.02
    x_center = 0.5
    y_center = 0.5
    width = 1.0 - (2 * margin)
    height = 1.0 - (2 * margin)

    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def process_class(class_name, class_id):
    """Process all images for a given class."""
    source_dir = BASE_DIR / class_name
    if not source_dir.exists():
        print(f"[✗] Source directory not found: {source_dir}")
        return 0, 0

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [
        f for f in source_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not images:
        print(f"[✗] No images found in {source_dir}")
        return 0, 0

    # Shuffle and split
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_count = 0
    val_count = 0

    # Process training images
    for img_path in train_images:
        # Copy image (convert to .jpg for consistency)
        dest_name = f"{class_name}_{img_path.stem}.jpg"
        dest_image = DATASET_DIR / "images" / "train" / dest_name
        dest_label = DATASET_DIR / "labels" / "train" / f"{class_name}_{img_path.stem}.txt"

        try:
            # Convert to JPG if needed
            if img_path.suffix.lower() != ".jpg":
                img = Image.open(img_path).convert("RGB")
                img.save(dest_image, "JPEG", quality=95)
            else:
                shutil.copy2(img_path, dest_image)

            generate_yolo_label(img_path, class_id, dest_label)
            train_count += 1
        except Exception as e:
            print(f"  [!] Error processing {img_path.name}: {e}")

    # Process validation images
    for img_path in val_images:
        dest_name = f"{class_name}_{img_path.stem}.jpg"
        dest_image = DATASET_DIR / "images" / "val" / dest_name
        dest_label = DATASET_DIR / "labels" / "val" / f"{class_name}_{img_path.stem}.txt"

        try:
            if img_path.suffix.lower() != ".jpg":
                img = Image.open(img_path).convert("RGB")
                img.save(dest_image, "JPEG", quality=95)
            else:
                shutil.copy2(img_path, dest_image)

            generate_yolo_label(img_path, class_id, dest_label)
            val_count += 1
        except Exception as e:
            print(f"  [!] Error processing {img_path.name}: {e}")

    return train_count, val_count


def main():
    print("=" * 60)
    print("  YOLO Dataset Preparation — ThumbsUp & OpenPalm")
    print("=" * 60)

    # Clean existing dataset
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        print("[✓] Cleaned existing dataset directory")

    create_directory_structure()

    total_train = 0
    total_val = 0

    for class_name, class_id in SOURCE_CLASSES.items():
        print(f"\nProcessing class '{class_name}' (id={class_id})...")
        train_count, val_count = process_class(class_name, class_id)
        total_train += train_count
        total_val += val_count
        print(f"  → Train: {train_count}, Val: {val_count}")

    print("\n" + "=" * 60)
    print(f"  Dataset preparation complete!")
    print(f"  Total training images:   {total_train}")
    print(f"  Total validation images: {total_val}")
    print(f"  Dataset directory:       {DATASET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
