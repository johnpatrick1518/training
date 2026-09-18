"""
prepare_dataset.py
Combines two dataset sources into a unified YOLO dataset:
  - ThumbsUp:  From the Roboflow "Hand gestures.v9" dataset (real bounding box annotations)
  - OpenPalm:  From the local "openPalm/" folder (auto-generated full-image bounding boxes)

Output classes:
  0 = openPalm
  1 = thumbsUp

The combined dataset is written to ./dataset/ in YOLO format with train/val splits.
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
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_RATIO = 0.8
SEED = 42

# ─── Source paths ────────────────────────────────────────────────────────────
# Roboflow ThumbsUp dataset (has train/valid/test splits with real annotations)
ROBOFLOW_DIR = Path(r"C:\Users\User\Downloads\Hand gestures.v9-latest-dataset.yolov8")
# In the Roboflow data.yaml: 0 = thumbs_down, 1 = thumbs_up
ROBOFLOW_THUMBSUP_CLASS_ID = 1   # class id for thumbs_up in the source dataset

# Local OpenPalm images (raw images, no annotations)
OPENPALM_DIR = BASE_DIR / "openPalm"

# Target class mapping for our combined dataset
TARGET_OPENPALM_CLASS = 0
TARGET_THUMBSUP_CLASS = 1

random.seed(SEED)


def create_directory_structure():
    """Create YOLO-format directory structure."""
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print("[OK] Created dataset directory structure")


def generate_openpalm_label(class_id, label_path):
    """
    Generate a YOLO-format label file with a near-full-image bounding box.
    Used for openPalm images that don't have annotations.
    """
    margin = 0.02
    x_center = 0.5
    y_center = 0.5
    width = 1.0 - (2 * margin)
    height = 1.0 - (2 * margin)

    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def remap_label_file(src_label_path, dst_label_path, src_class_id, target_class_id):
    """
    Copy a YOLO label file, keeping only lines matching src_class_id
    and remapping them to target_class_id.
    Returns the number of annotations written.
    """
    count = 0
    with open(src_label_path, "r") as fin, open(dst_label_path, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == src_class_id:
                parts[0] = str(target_class_id)
                fout.write(" ".join(parts) + "\n")
                count += 1
    return count


def process_roboflow_thumbsup():
    """
    Extract thumbs_up images and labels from the Roboflow dataset.
    Returns lists of (image_path, label_path) tuples for train and val.
    """
    train_pairs = []
    val_pairs = []

    # Process each split from the Roboflow dataset
    split_map = {
        "train": train_pairs,
        "valid": val_pairs,
        "test": val_pairs,  # merge test into val since our project only uses train/val
    }

    for split_name, pair_list in split_map.items():
        img_dir = ROBOFLOW_DIR / split_name / "images"
        lbl_dir = ROBOFLOW_DIR / split_name / "labels"

        if not img_dir.exists():
            print(f"  [!] Roboflow {split_name}/images not found, skipping")
            continue

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in image_extensions:
                continue

            # Check if corresponding label exists and contains thumbs_up
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Check if this label file contains thumbs_up annotations
            has_thumbsup = False
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == ROBOFLOW_THUMBSUP_CLASS_ID:
                        has_thumbsup = True
                        break

            if has_thumbsup:
                pair_list.append((img_path, lbl_path))

    print(f"  [OK] Found {len(train_pairs)} ThumbsUp train images from Roboflow")
    print(f"  [OK] Found {len(val_pairs)} ThumbsUp val images from Roboflow")

    return train_pairs, val_pairs


def process_openpalm():
    """
    Collect openPalm images from the local folder and split into train/val.
    Returns lists of image paths for train and val.
    """
    if not OPENPALM_DIR.exists():
        print(f"  [!] OpenPalm directory not found: {OPENPALM_DIR}")
        return [], []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted([
        f for f in OPENPALM_DIR.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not images:
        print(f"  [!] No images found in {OPENPALM_DIR}")
        return [], []

    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"  [OK] Found {len(train_images)} OpenPalm train images")
    print(f"  [OK] Found {len(val_images)} OpenPalm val images")

    return train_images, val_images


def copy_thumbsup_to_dataset(pairs, split):
    """Copy ThumbsUp images and remapped labels to the dataset directory."""
    count = 0
    for img_path, lbl_path in pairs:
        dest_name = f"thumbsUp_{img_path.stem}"
        dest_image = DATASET_DIR / "images" / split / (dest_name + ".jpg")
        dest_label = DATASET_DIR / "labels" / split / (dest_name + ".txt")

        try:
            # Copy image (convert to JPG if needed)
            if img_path.suffix.lower() not in (".jpg", ".jpeg"):
                img = Image.open(img_path).convert("RGB")
                img.save(dest_image, "JPEG", quality=95)
            else:
                shutil.copy2(img_path, dest_image)

            # Remap label: filter only thumbs_up annotations and remap to target class
            annotations = remap_label_file(
                lbl_path, dest_label,
                src_class_id=ROBOFLOW_THUMBSUP_CLASS_ID,
                target_class_id=TARGET_THUMBSUP_CLASS
            )
            if annotations > 0:
                count += 1
            else:
                # No thumbs_up annotations written; clean up
                dest_image.unlink(missing_ok=True)
                dest_label.unlink(missing_ok=True)
        except Exception as e:
            print(f"  [!] Error processing {img_path.name}: {e}")

    return count


def copy_openpalm_to_dataset(images, split):
    """Copy OpenPalm images and generate labels to the dataset directory."""
    count = 0
    for img_path in images:
        dest_name = f"openPalm_{img_path.stem}"
        dest_image = DATASET_DIR / "images" / split / (dest_name + ".jpg")
        dest_label = DATASET_DIR / "labels" / split / (dest_name + ".txt")

        try:
            if img_path.suffix.lower() not in (".jpg", ".jpeg"):
                img = Image.open(img_path).convert("RGB")
                img.save(dest_image, "JPEG", quality=95)
            else:
                shutil.copy2(img_path, dest_image)

            generate_openpalm_label(TARGET_OPENPALM_CLASS, dest_label)
            count += 1
        except Exception as e:
            print(f"  [!] Error processing {img_path.name}: {e}")

    return count


def main():
    print("=" * 60)
    print("  YOLO Dataset Preparation — ThumbsUp & OpenPalm")
    print("  (Combined: Roboflow + Local Sources)")
    print("=" * 60)

    # Clean existing dataset
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        print("[OK] Cleaned existing dataset directory")

    create_directory_structure()

    # ─── Process ThumbsUp from Roboflow ──────────────────────────────────
    print("\n--- ThumbsUp (from Roboflow Hand Gestures v9 dataset) ---")
    tu_train, tu_val = process_roboflow_thumbsup()

    tu_train_count = copy_thumbsup_to_dataset(tu_train, "train")
    tu_val_count = copy_thumbsup_to_dataset(tu_val, "val")
    print(f"  -> ThumbsUp copied: Train={tu_train_count}, Val={tu_val_count}")

    # ─── Process OpenPalm from local folder ──────────────────────────────
    print("\n--- OpenPalm (from local openPalm/ folder) ---")
    op_train, op_val = process_openpalm()

    op_train_count = copy_openpalm_to_dataset(op_train, "train")
    op_val_count = copy_openpalm_to_dataset(op_val, "val")
    print(f"  -> OpenPalm copied: Train={op_train_count}, Val={op_val_count}")

    # ─── Summary ─────────────────────────────────────────────────────────
    total_train = tu_train_count + op_train_count
    total_val = tu_val_count + op_val_count

    print("\n" + "=" * 60)
    print("  Dataset preparation complete!")
    print(f"  ThumbsUp (Roboflow):  Train={tu_train_count}, Val={tu_val_count}")
    print(f"  OpenPalm (local):     Train={op_train_count}, Val={op_val_count}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total training images:   {total_train}")
    print(f"  Total validation images: {total_val}")
    print(f"  Dataset directory:       {DATASET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
