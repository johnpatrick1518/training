"""
detect_camera.py
Real-time Hand Gesture Detection using YOLOv9 (best.pt) with Camera Selection.

Features:
- Auto-detects and lists available camera devices (Camera 0, 1, 2, etc.)
- Interactive camera selection prompt at startup
- Command-line arguments support (e.g., --camera 1, --conf 0.5)
- In-app live camera switching (press 'c' to cycle or keys 0-9)
- Real-time confidence adjustment (press '+' or '-')
- Screenshot capture (press 's')

Usage:
    python detect_camera.py
    python detect_camera.py --camera 1
    python detect_camera.py --camera 2 --conf 0.6
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Default Configurations ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR / "best.pt",
    BASE_DIR / "runs" / "detect" / "hand_gesture" / "weights" / "best.pt",
]

# Color palette for classes (BGR)
CLASS_COLORS = {
    0: (0, 200, 0),      # openPalm  -> Vibrant Green
    1: (255, 140, 0),    # thumbsUp  -> Deep Sky Blue
}

CLASS_DISPLAY = {
    0: "OPEN PALM",
    1: "THUMBS UP",
}


def find_model_file(custom_path=None):
    """Find the best.pt model weights file."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists():
            return p
        print(f"[!] Warning: Specified model '{custom_path}' not found.")

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    # Search recursively for any best.pt under runs/
    runs_dir = BASE_DIR / "runs"
    if runs_dir.exists():
        found = list(runs_dir.rglob("best.pt"))
        if found:
            return found[0]

    return None


def scan_available_cameras(max_tested=6):
    """Scan and return indices of accessible video capture devices."""
    print("🔍 Scanning for available camera devices...")
    available_cameras = []
    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(index)
            cap.release()
    return available_cameras


def select_camera_interactively(available_cameras, preselected=None):
    """Prompt user to select a camera index if not explicitly supplied."""
    if preselected is not None:
        if preselected in available_cameras:
            return preselected
        else:
            print(f"[!] Warning: Camera {preselected} was not detected.")
            print(f"    Available devices: {available_cameras}")

    if not available_cameras:
        print("[!] No working cameras detected. Falling back to camera 0.")
        return 0

    if len(available_cameras) == 1:
        print(f"[✓] Single camera detected: Camera {available_cameras[0]}")
        return available_cameras[0]

    print("\n" + "=" * 50)
    print("  AVAILABLE CAMERAS DETECTED")
    print("=" * 50)
    for cam_idx in available_cameras:
        marker = " (Default)" if cam_idx == 0 else ""
        print(f"  [{cam_idx}] Camera index {cam_idx}{marker}")
    print("=" * 50)

    default_choice = available_cameras[0]
    while True:
        choice = input(f"Select camera index [{default_choice}]: ").strip()
        if choice == "":
            return default_choice
        try:
            val = int(choice)
            if val in available_cameras:
                return val
            else:
                print(f"Please select an index from {available_cameras}")
        except ValueError:
            print("Invalid input. Enter a valid number.")


def init_camera(camera_idx, width=1280, height=720):
    """Initialize camera capture object with desired resolution."""
    print(f"[*] Opening Camera {camera_idx}...")
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[!] Failed to open camera {camera_idx}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_label(frame, text, x1, y1, color, font_scale=0.65, thickness=2):
    """Draw a styled background badge for the detected class label."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    badge_top = max(0, y1 - th - 12)
    badge_bottom = max(th + 12, y1)
    badge_right = x1 + tw + 14

    # Background badge
    cv2.rectangle(frame, (x1, badge_top), (badge_right, badge_bottom), color, -1)
    # White text with border
    cv2.putText(frame, text, (x1 + 7, badge_bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hud(frame, fps, current_cam, available_cams, conf_thresh, detections):
    """Draw modern semi-transparent HUD overlay on screen."""
    h, w = frame.shape[:2]
    hud_w = 340
    hud_h = 135

    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (12, 12), (12 + hud_w, 12 + hud_h), (80, 80, 80), 1)

    # Title / Camera
    cams_str = ",".join(str(c) for c in available_cams)
    cv2.putText(frame, f"CAM: {current_cam} (Avail: [{cams_str}])", (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA)

    # FPS & Confidence
    cv2.putText(frame, f"FPS: {fps:4.1f}   |   Conf: {conf_thresh:.0%}", (24, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

    # Detections info
    if detections:
        det_summary = ", ".join(f"{cnt}x {name}" for name, cnt in detections.items())
        cv2.putText(frame, f"Detected: {det_summary}", (24, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Detected: None", (24, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    # Controls help
    cv2.putText(frame, "[C] Next Cam | [0-9] Select | [S] Shot | [Q] Quit", (24, 126),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="Live Hand Gesture Detection with Camera Selection")
    parser.add_argument("-c", "--camera", type=int, default=None,
                        help="Camera device index (e.g. 0, 1, 2)")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="Path to trained YOLO model weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Confidence threshold (0.1 - 0.95)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 56)
    print("      YOLOv9 HAND GESTURE DETECTION (best.pt)")
    print("=" * 56)

    # 1. Locate Model
    model_path = find_model_file(args.model)
    if not model_path:
        print("[ERROR] Could not find 'best.pt'!")
        print("Expected at one of:")
        for cand in DEFAULT_MODEL_CANDIDATES:
            print(f"  - {cand}")
        print("\nPlease ensure you have trained the model or pass --model path/to/best.pt")
        sys.exit(1)

    print(f"[✓] Loading model from: {model_path}")
    model = YOLO(str(model_path))
    print(f"[✓] Model loaded! Classes: {model.names}")

    # 2. Camera Discovery & Selection
    available_cameras = scan_available_cameras()
    selected_cam = select_camera_interactively(available_cameras, args.camera)

    cap = init_camera(selected_cam)
    if cap is None:
        sys.exit(1)

    window_name = f"Hand Gesture Detection - Camera {selected_cam}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    conf_thresh = args.conf
    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    screenshot_idx = 1

    print("\n" + "-" * 56)
    print("  Application running!")
    print("  Keyboard Shortcuts:")
    print("    'c'       : Switch to next available camera")
    print("    '0' - '9' : Switch directly to camera index")
    print("    '+' / '-' : Increase / decrease confidence threshold")
    print("    's'       : Take screenshot")
    print("    'q' / ESC : Quit application")
    print("-" * 56 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[!] Warning: Failed to grab frame from Camera {selected_cam}")
                time.sleep(0.1)
                continue

            # Run inference
            results = model(frame, conf=conf_thresh, verbose=False)

            detected_counts = {}

            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        raw_name = model.names.get(cls_id, f"cls_{cls_id}")
                        display_name = CLASS_DISPLAY.get(cls_id, raw_name.upper())

                        # Color
                        color = CLASS_COLORS.get(cls_id, (0, 255, 255))

                        # Bounding Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                        # Label badge
                        label_text = f"{display_name} {conf:.0%}"
                        draw_label(frame, label_text, x1, y1, color)

                        detected_counts[display_name] = detected_counts.get(display_name, 0) + 1

            # FPS Calculation
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = now

            # Draw HUD
            draw_hud(frame, fps, selected_cam, available_cameras, conf_thresh, detected_counts)

            # Display
            cv2.imshow(window_name, frame)

            # Handle Key Presses
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # 'q' or ESC
                print("\n[✓] Exiting...")
                break

            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = f"gesture_cam{selected_cam}_{ts}_{screenshot_idx}.jpg"
                save_path = BASE_DIR / filename
                cv2.imwrite(str(save_path), frame)
                print(f"[✓] Screenshot saved: {filename}")
                screenshot_idx += 1

            elif key in (ord('+'), ord('=')):
                conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
                print(f"[*] Confidence threshold increased to: {conf_thresh:.0%}")

            elif key in (ord('-'), ord('_')):
                conf_thresh = max(0.10, round(conf_thresh - 0.05, 2))
                print(f"[*] Confidence threshold decreased to: {conf_thresh:.0%}")

            elif key == ord('c'):
                # Cycle to next available camera
                if len(available_cameras) > 1:
                    curr_pos = available_cameras.index(selected_cam) if selected_cam in available_cameras else -1
                    next_pos = (curr_pos + 1) % len(available_cameras)
                    new_cam = available_cameras[next_pos]

                    print(f"\n[*] Switching from Camera {selected_cam} to Camera {new_cam}...")
                    cap.release()
                    new_cap = init_camera(new_cam)
                    if new_cap is not None:
                        cap = new_cap
                        selected_cam = new_cam
                        cv2.setWindowTitle(window_name, f"Hand Gesture Detection - Camera {selected_cam}")
                    else:
                        print(f"[!] Re-opening previous Camera {selected_cam}...")
                        cap = init_camera(selected_cam)
                else:
                    print("[!] Only one camera is available.")

            elif ord('0') <= key <= ord('9'):
                target_cam = key - ord('0')
                if target_cam in available_cameras and target_cam != selected_cam:
                    print(f"\n[*] Switching to Camera {target_cam}...")
                    cap.release()
                    new_cap = init_camera(target_cam)
                    if new_cap is not None:
                        cap = new_cap
                        selected_cam = target_cam
                        cv2.setWindowTitle(window_name, f"Hand Gesture Detection - Camera {selected_cam}")
                    else:
                        print(f"[!] Re-opening previous Camera {selected_cam}...")
                        cap = init_camera(selected_cam)

    except KeyboardInterrupt:
        print("\n[✓] Interrupted by user.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[✓] Camera closed and cleanup complete.")


if __name__ == "__main__":
    main()
