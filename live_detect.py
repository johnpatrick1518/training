"""
live_detect.py
Real-time hand gesture detection using a trained YOLOv9 model and webcam.

Usage:
    python live_detect.py

Controls:
    q - Quit the application
    s - Save current frame as screenshot

The script loads the trained model from:
    runs/detect/hand_gesture/weights/best.pt
"""

import cv2
import sys
import time
from pathlib import Path
from ultralytics import YOLO


# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / "hand_gesture_to50" / "weights" / "best.pt"
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0        # Default webcam (change to 1, 2, etc. for other cameras)
WINDOW_NAME = "YOLOv9 Hand Gesture Detection"

# Colors for each class (BGR format)
CLASS_COLORS = {
    0: (0, 200, 0),      # openPalm  → Green
    1: (255, 165, 0),    # thumbsUp  → Blue-ish (BGR)
}

CLASS_EMOJIS = {
    0: "OPEN PALM",
    1: "THUMBS UP",
}


def draw_fancy_label(frame, text, x1, y1, color, font_scale=0.7, thickness=2):
    """Draw a label with a background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Background rectangle
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    # Text
    cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)


def draw_info_panel(frame, fps, detections_count):
    """Draw an info panel on the top-left corner."""
    h, w = frame.shape[:2]
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    # Text
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {detections_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    # ─── Check model exists ──────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"[ERROR] Trained model not found at: {MODEL_PATH}")
        print("        Please run 'python train.py' first to train the model.")
        sys.exit(1)

    # ─── Load model ──────────────────────────────────────────────────────
    print(f"Loading YOLOv9 model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("[✓] Model loaded successfully")

    # ─── Open webcam ─────────────────────────────────────────────────────
    print(f"Opening webcam (index={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[✓] Webcam opened successfully")
    print("=" * 50)
    print("  Live Detection Running!")
    print("  Show 'Thumbs Up' or 'Open Palm' to the camera")
    print("  Press 'q' to quit | Press 's' to screenshot")
    print("=" * 50)

    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    screenshot_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to read frame from webcam")
                break

            # ─── Run inference ────────────────────────────────────────
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # ─── Draw detections ──────────────────────────────────────
            detections_count = 0

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])

                        # Get class info
                        cls_name = model.names[cls_id]
                        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                        emoji = CLASS_EMOJIS.get(cls_id, cls_name)

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                        # Draw label
                        label = f"{emoji} {conf:.0%}"
                        draw_fancy_label(frame, label, x1, y1, color)

                        detections_count += 1

            # ─── Calculate FPS ────────────────────────────────────────
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = current_time

            # ─── Draw info panel ──────────────────────────────────────
            draw_info_panel(frame, fps, detections_count)

            # ─── Display frame ────────────────────────────────────────
            cv2.imshow(WINDOW_NAME, frame)

            # ─── Handle key presses ───────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[✓] Quitting...")
                break
            elif key == ord('s'):
                screenshot_name = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(str(BASE_DIR / screenshot_name), frame)
                screenshot_count += 1
                print(f"[✓] Screenshot saved: {screenshot_name}")

    except KeyboardInterrupt:
        print("\n[✓] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[✓] Cleanup complete")


if __name__ == "__main__":
    main()
