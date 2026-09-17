"""
hand_detector.py
Hand-Focused Gesture Detection Module.

Solves:
1. Focuses detection strictly to hands (suppresses face, neck, background).
2. Returns 0 active gestures when no hands are in frame.
3. Crops candidate hand regions and classifies them using the trained YOLOv9 model (best.pt).
"""

import cv2
import numpy as np


class HandGestureDetector:
    """Detects and classifies hand gestures strictly focused on hand regions."""

    def __init__(self, model):
        self.model = model
        # Load OpenCV Haar cascade for face suppression
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def find_hand_candidates(self, frame):
        """
        Segment skin regions and isolate hand candidates while suppressing faces.
        Returns a list of bounding boxes [(x1, y1, x2, y2), ...]
        """
        h, w = frame.shape[:2]
        frame_area = w * h

        # 1. Detect faces to prevent faces/heads from being mistaken for hands
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.25,
            minNeighbors=4,
            minSize=(60, 60)
        )

        # 2. Skin color segmentation using dual-color space (YCrCb + HSV)
        # YCrCb skin chrominance
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

        # HSV skin hue & saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, (0, 30, 45), (28, 255, 255))

        # Combine both for robust illumination handling
        skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # 3. Suppress detected face and neck regions
        for (fx, fy, fw, fh) in faces:
            pad_x = int(fw * 0.18)
            pad_y_top = int(fh * 0.15)
            pad_y_bot = int(fh * 0.45)  # covers chin and neck
            x_start = max(0, fx - pad_x)
            y_start = max(0, fy - pad_y_top)
            x_end = min(w, fx + fw + pad_x)
            y_end = min(h, fy + fh + pad_y_bot)
            cv2.rectangle(skin_mask, (x_start, y_start), (x_end, y_end), 0, -1)

        # 4. Morphological clean-up (remove pepper noise, connect fingers to palm)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 5. Extract contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_hand_area = frame_area * 0.0035  # ~3,200 px on 720p
        max_hand_area = frame_area * 0.35    # hands don't cover > 35% of frame

        candidate_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_hand_area or area > max_hand_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / float(bh)
            if aspect_ratio < 0.22 or aspect_ratio > 3.2:
                continue

            # Add gentle padding around the hand crop
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)

            candidate_boxes.append((x1, y1, x2, y2))

        return candidate_boxes

    def detect(self, frame, conf_threshold=0.50):
        """
        Detect and classify hand gestures in the given frame.

        Returns:
            list of dicts: [
                {
                    "bbox": (x1, y1, x2, y2),
                    "cls_id": int,
                    "class_name": str,
                    "confidence": float
                },
                ...
            ]
        If no hands are detected or gesture confidence is below threshold, returns [].
        """
        candidates = self.find_hand_candidates(frame)
        if not candidates:
            # Strictly return 0 when no hands are present
            return []

        detections = []
        h, w = frame.shape[:2]

        for (x1, y1, x2, y2) in candidates:
            hand_crop = frame[y1:y2, x1:x2]
            if hand_crop.size == 0 or hand_crop.shape[0] < 20 or hand_crop.shape[1] < 20:
                continue

            # Run YOLO on the hand crop (exact format of training data)
            results = self.model(hand_crop, conf=conf_threshold, verbose=False)
            if not results or len(results[0].boxes) == 0:
                continue

            # Select the most confident prediction for this hand region
            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            conf = float(best_box.conf[0])
            cls_id = int(best_box.cls[0])

            if conf >= conf_threshold:
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "cls_id": cls_id,
                    "class_name": self.model.names.get(cls_id, f"class_{cls_id}"),
                    "confidence": conf
                })

        return detections
