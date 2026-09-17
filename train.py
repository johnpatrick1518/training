"""
train.py
Trains a YOLOv9 model on the ThumbsUp/OpenPalm hand gesture dataset
using the Ultralytics library.

Usage:
    python train.py

The trained model weights will be saved to:
    runs/detect/hand_gesture/weights/best.pt
"""

from ultralytics import YOLO
from pathlib import Path
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def main():
    # ─── Configuration ───────────────────────────────────────────────────
    BASE_DIR = Path(__file__).parent
    DATA_YAML = str(BASE_DIR / "dataset.yaml")
    MODEL = "yolov9t.pt"        # YOLOv9 Tiny — fast, good for small datasets
    EPOCHS = 50                  # Number of training epochs
    IMG_SIZE = 640               # Input image size
    BATCH_SIZE = 16              # Batch size (reduce to 8 if you run out of memory)
    PROJECT = str(BASE_DIR / "runs" / "detect")
    NAME = "hand_gesture"

    print("=" * 60)
    print("  YOLOv9 Training — Hand Gesture Detection")
    print("=" * 60)
    print(f"  Model:      {MODEL}")
    print(f"  Dataset:    {DATA_YAML}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print("=" * 60)

    # ─── Load Model ──────────────────────────────────────────────────────
    # This will download the pretrained YOLOv9t weights automatically
    model = YOLO(MODEL)

    # ─── Train ───────────────────────────────────────────────────────────
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,           # Overwrite if exists
        pretrained=True,         # Use pretrained weights (transfer learning)
        optimizer="auto",        # Automatic optimizer selection
        patience=10,             # Early stopping patience
        save=True,               # Save checkpoints
        save_period=-1,          # Save only best
        verbose=True,            # Detailed logging
        plots=True,              # Generate training plots
    )

    # ─── Results ─────────────────────────────────────────────────────────
    best_weights = Path(PROJECT) / NAME / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Best weights saved to: {best_weights}")
    print("=" * 60)

    # ─── Validate ────────────────────────────────────────────────────────
    print("\nRunning validation on best model...")
    best_model = YOLO(str(best_weights))
    val_results = best_model.val(data=DATA_YAML)
    
    print(f"\n  mAP50:    {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
