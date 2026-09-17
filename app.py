"""
app.py
Flask Web GUI for Real-time Hand Gesture Detection using YOLOv9 (best.pt).

Features:
- Live MJPEG video stream with YOLO annotations
- Interactive camera selection & real-time switching
- Dynamic confidence threshold adjustment
- Live telemetry: FPS, inference time, detection counts
- One-click snapshot saving & gallery viewer
- Mirror flip toggle
"""

import os
import sys
import time
import threading
from pathlib import Path
import cv2
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from ultralytics import YOLO

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent.resolve()
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Find model path
def get_model_path():
    candidates = [
        BASE_DIR / "best.pt",
        BASE_DIR / "runs" / "detect" / "hand_gesture" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    runs_dir = BASE_DIR / "runs"
    if runs_dir.exists():
        found = list(runs_dir.rglob("best.pt"))
        if found:
            return found[0]
    return None


MODEL_PATH = get_model_path()
if not MODEL_PATH:
    print("[ERROR] best.pt not found! Please check model path.")
    sys.exit(1)

print(f"[✓] Loading YOLO model from: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print(f"[✓] Model loaded! Class names: {model.names}")

CLASS_COLORS = {
    0: (0, 210, 80),     # openPalm  -> Emerald Green (BGR)
    1: (255, 140, 0),    # thumbsUp  -> Sky/Cyan Blue (BGR)
}

CLASS_DISPLAY = {
    0: "Open Palm",
    1: "Thumbs Up",
}


class CameraManager:
    """Thread-safe camera capture and YOLO inference manager."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current_camera_idx = 0
        self.cap = None
        self.is_running = True
        self.mirror = True
        self.confidence_threshold = 0.50
        self.fps = 0.0
        self.inference_time_ms = 0.0
        self.current_detections = {}
        self.latest_frame = None
        self.available_cameras = []
        self.scan_cameras()
        self.init_camera(self.current_camera_idx)

    def scan_cameras(self, max_tested=6):
        """Scan available camera devices."""
        print("[*] Scanning for available camera indices...")
        available = []
        for idx in range(max_tested):
            temp_cap = cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    available.append(idx)
                temp_cap.release()
        self.available_cameras = available if available else [0]
        print(f"[✓] Discovered cameras: {self.available_cameras}")
        return self.available_cameras

    def init_camera(self, cam_idx):
        """Open or switch to a camera index."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            print(f"[*] Initializing camera index {cam_idx}...")
            self.cap = cv2.VideoCapture(cam_idx)
            if not self.cap.isOpened():
                print(f"[!] Warning: Could not open camera {cam_idx}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.current_camera_idx = cam_idx
            return True

    def switch_camera(self, new_idx):
        """Switch active camera to new index."""
        if new_idx == self.current_camera_idx and self.cap and self.cap.isOpened():
            return True
        success = self.init_camera(new_idx)
        return success

    def get_frame(self):
        """Capture frame, run YOLO detection, draw overlays, and encode as JPEG."""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return self._generate_error_frame("Camera disconnected or unavailable")

            ret, frame = self.cap.read()
            if not ret or frame is None:
                return self._generate_error_frame("Waiting for video feed...")

            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Store clean frame copy for high-res screenshots
            self.latest_frame = frame.copy()

            # YOLO inference
            start_infer = time.time()
            results = model(frame, conf=self.confidence_threshold, verbose=False)
            self.inference_time_ms = (time.time() - start_infer) * 1000.0

            detections_map = {}
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        raw_name = model.names.get(cls_id, f"class_{cls_id}")
                        disp_name = CLASS_DISPLAY.get(cls_id, raw_name)
                        color = CLASS_COLORS.get(cls_id, (0, 255, 255))

                        # Bounding Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                        # Label badge
                        label_text = f"{disp_name}  {conf:.0%}"
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                        badge_y1 = max(0, y1 - th - 12)
                        badge_y2 = max(th + 12, y1)
                        cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 14, badge_y2), color, -1)
                        cv2.putText(frame, label_text, (x1 + 7, badge_y2 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                        if disp_name not in detections_map:
                            detections_map[disp_name] = {"count": 0, "max_conf": conf}
                        detections_map[disp_name]["count"] += 1
                        detections_map[disp_name]["max_conf"] = max(detections_map[disp_name]["max_conf"], conf)

            self.current_detections = detections_map

            # Encode frame to JPEG
            ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret_encode:
                return None
            return buffer.tobytes()

    def _generate_error_frame(self, message):
        """Generate an informative placeholder frame when camera is not ready."""
        placeholder = cv2.imread(None) if False else None
        import numpy as np
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (w - 20, h - 20), (30, 30, 45), -1)
        cv2.putText(img, "YOLOv9 Hand Gesture AI", (w // 2 - 170, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(img, message, (w // 2 - 190, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()

    def capture_snapshot(self):
        """Save the latest frame to disk and return file info."""
        with self.lock:
            if self.latest_frame is None:
                return None
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"gesture_cam{self.current_camera_idx}_{ts}.jpg"
            filepath = SCREENSHOTS_DIR / filename
            cv2.imwrite(str(filepath), self.latest_frame)
            return {
                "filename": filename,
                "url": f"/screenshots/{filename}",
                "timestamp": ts,
                "camera": self.current_camera_idx
            }


camera_mgr = CameraManager()
app = Flask(__name__)


def gen_frames():
    """Generator for streaming video frames with FPS tracking."""
    prev_time = time.time()
    frame_count = 0

    while True:
        frame_bytes = camera_mgr.get_frame()
        if frame_bytes is None:
            time.sleep(0.04)
            continue

        frame_count += 1
        now = time.time()
        elapsed = now - prev_time
        if elapsed >= 1.0:
            camera_mgr.fps = frame_count / elapsed
            frame_count = 0
            prev_time = now

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.015)


# ─── Flask Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Render main GUI dashboard."""
    return render_template(
        'index.html',
        model_name=MODEL_PATH.name,
        cameras=camera_mgr.available_cameras,
        current_camera=camera_mgr.current_camera_idx,
        confidence=int(camera_mgr.confidence_threshold * 100)
    )


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get list of detected cameras and current active camera."""
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    if refresh:
        camera_mgr.scan_cameras()
    return jsonify({
        "success": True,
        "cameras": camera_mgr.available_cameras,
        "current_camera": camera_mgr.current_camera_idx
    })


@app.route('/api/set_camera', methods=['POST'])
def set_camera():
    """Switch active camera index."""
    data = request.get_json(silent=True) or {}
    cam_idx = data.get('camera_index')
    if cam_idx is None:
        return jsonify({"success": False, "error": "camera_index is required"}), 400

    try:
        cam_idx = int(cam_idx)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid camera index"}), 400

    success = camera_mgr.switch_camera(cam_idx)
    return jsonify({
        "success": success,
        "current_camera": camera_mgr.current_camera_idx
    })


@app.route('/api/set_confidence', methods=['POST'])
def set_confidence():
    """Update detection confidence threshold (0.1 to 0.95)."""
    data = request.get_json(silent=True) or {}
    conf = data.get('confidence')
    if conf is None:
        return jsonify({"success": False, "error": "confidence value required"}), 400

    try:
        conf_val = float(conf)
        if 0.05 <= conf_val <= 0.98:
            camera_mgr.confidence_threshold = conf_val
            return jsonify({"success": True, "confidence": camera_mgr.confidence_threshold})
        return jsonify({"success": False, "error": "Confidence must be between 0.05 and 0.98"}), 400
    except ValueError:
        return jsonify({"success": False, "error": "Invalid number"}), 400


@app.route('/api/toggle_mirror', methods=['POST'])
def toggle_mirror():
    """Toggle horizontal flip (mirror mode)."""
    camera_mgr.mirror = not camera_mgr.mirror
    return jsonify({"success": True, "mirror": camera_mgr.mirror})


@app.route('/api/snapshot', methods=['POST'])
def take_snapshot():
    """Save snapshot from live feed."""
    snapshot_info = camera_mgr.capture_snapshot()
    if snapshot_info:
        return jsonify({"success": True, "snapshot": snapshot_info})
    return jsonify({"success": False, "error": "No frame available to capture"}), 500


@app.route('/api/screenshots', methods=['GET'])
def list_screenshots():
    """Return list of saved screenshot files."""
    files = []
    if SCREENSHOTS_DIR.exists():
        for p in sorted(SCREENSHOTS_DIR.glob("*.jpg"), key=os.path.getmtime, reverse=True):
            files.append({
                "filename": p.name,
                "url": f"/screenshots/{p.name}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
            })
    return jsonify({"success": True, "screenshots": files[:20]})


@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    """Serve saved screenshot image."""
    return send_from_directory(SCREENSHOTS_DIR, filename)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return real-time telemetry (FPS, latency, detections, camera)."""
    return jsonify({
        "fps": round(camera_mgr.fps, 1),
        "inference_ms": round(camera_mgr.inference_time_ms, 1),
        "current_camera": camera_mgr.current_camera_idx,
        "confidence": round(camera_mgr.confidence_threshold, 2),
        "mirror": camera_mgr.mirror,
        "detections": camera_mgr.current_detections,
        "classes": list(CLASS_DISPLAY.values())
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting Hand Gesture Detection Flask Web GUI")
    print(f"👉 Local Web Server: http://127.0.0.1:5000")
    print(f"📷 Active Camera: Index {camera_mgr.current_camera_idx}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
