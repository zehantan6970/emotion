"""
Vision Emotion Analysis Module
- Captures real-time video from camera
- Detects facial emotions using DeepFace / FER
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import threading
import time
import warnings
warnings.filterwarnings("ignore")
from collections import deque
from typing import Optional, Dict, Any

from loguru import logger


class CameraCapture:
    """Camera capture class - reads frames from webcam"""

    def __init__(self, cfg: dict):
        self.device_id = cfg.get("device_id", 0)
        self.width = cfg.get("width", 640)
        self.height = cfg.get("height", 480)
        self.fps = cfg.get("fps", 30)
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            logger.error("Cannot open camera, device ID: {}", self.device_id)
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera started, resolution: {}x{}, FPS: {}", self.width, self.height, self.fps)
        return True

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                logger.warning("Camera read failed, retrying")
                time.sleep(0.01)

    def get_frame(self) -> Optional[Any]:
        """Get latest frame (thread-safe)"""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        logger.info("Camera stopped")


class VisionEmotionAnalyzer:
    """Visual emotion analyzer supporting DeepFace and FER backends"""

    EMOTION_EN = {
        "angry": "angry", "disgust": "disgust", "fear": "fear",
        "happy": "happy", "sad": "sad", "surprise": "surprise", "neutral": "neutral"
    }

    def __init__(self, cfg: dict):
        self.backend = cfg.get("backend", "deepface")
        self.model_name = cfg.get("model_name", "VGG-Face")
        self.detector_backend = cfg.get("detector_backend", "opencv")
        self.confidence_threshold = cfg.get("confidence_threshold", 0.3)
        self.emotion_labels = cfg.get("emotion_labels", self.EMOTION_EN)
        self._analyzer = None
        self._load_model()

    def _load_model(self):
        """Lazy-load emotion analysis model"""
        if self.backend == "fer":
            try:
                from fer import FER
                self._analyzer = FER(mtcnn=False)
                logger.info("FER model loaded successfully")
            except ImportError:
                logger.error("FER not installed, run: pip install fer")
        elif self.backend == "deepface":
            logger.info("DeepFace backend ready (model loads on first analysis)")
        else:
            logger.warning("Unknown backend: {}, falling back to fer", self.backend)
            self.backend = "fer"
            self._load_model()

    def analyze(self, frame) -> Optional[Dict[str, Any]]:
        """
        Analyze emotion from a single frame
        Returns: {"emotion": "happy", "emotion_en": "happy", "confidence": 0.95, "all_scores": {...}}
        """
        try:
            if self.backend == "fer":
                return self._analyze_fer(frame)
            else:
                return self._analyze_deepface(frame)
        except Exception as e:
            logger.debug("Vision emotion analysis error: {}", e)
            return None

    def _analyze_fer(self, frame) -> Optional[Dict[str, Any]]:
        result = self._analyzer.detect_emotions(frame)
        if not result:
            return None
        emotions = result[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        confidence = emotions[top_emotion]
        if confidence < self.confidence_threshold:
            return None
        return {
            "emotion": self.emotion_labels.get(top_emotion, top_emotion),
            "emotion_en": top_emotion,
            "confidence": round(confidence, 3),
            "all_scores": {self.emotion_labels.get(k, k): round(v, 3) for k, v in emotions.items()},
            "box": result[0].get("box"),
        }

    def _analyze_deepface(self, frame) -> Optional[Dict[str, Any]]:
        from deepface import DeepFace

        # Step 1: Use Haar Cascade for quick face detection
        haar_box = self._detect_face_haar(frame)

        # Step 2: Try multiple detectors in order
        detectors = []
        if self.detector_backend not in ("opencv", "ssd", "skip"):
            detectors.append(self.detector_backend)
        detectors += ["ssd", "opencv", "skip"]
        detectors = list(dict.fromkeys(detectors))  # deduplicate preserving order

        tried = set()
        for detector in detectors:
            if detector in tried:
                continue
            tried.add(detector)
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend=detector,
                    enforce_detection=False,
                    silent=True,
                )
                if not results:
                    continue
                r = results[0] if isinstance(results, list) else results
                emotions = r.get("emotion", {})
                top_emotion = r.get("dominant_emotion", "")
                if not top_emotion or not emotions:
                    continue
                confidence = emotions.get(top_emotion, 0) / 100.0

                # skip mode: skip if Haar also found no face (avoid false positives)
                region = r.get("region", {})
                rw = region.get("w", 0)
                rh = region.get("h", 0)
                if detector == "skip" and rw < 20 and rh < 20 and haar_box is None:
                    continue

                # Confidence threshold (relaxed for skip mode)
                threshold = 0.08 if detector == "skip" else self.confidence_threshold
                if confidence < threshold:
                    continue

                # Prefer Haar face box (more accurate) when region is too small
                box = haar_box if (haar_box and (rw < 20 or rh < 20)) else region
                return {
                    "emotion": self.emotion_labels.get(top_emotion, top_emotion),
                    "emotion_en": top_emotion,
                    "confidence": round(confidence, 3),
                    "all_scores": {self.emotion_labels.get(k, k): round(v / 100, 3) for k, v in emotions.items()},
                    "box": box,
                    "detector": detector,
                }
            except Exception:
                continue

        # Step 3: All DeepFace detectors failed - use Haar box + skip emotion
        if haar_box is not None:
            return self._analyze_deepface_with_roi(frame, haar_box)
        return None

    def _detect_face_haar(self, frame) -> Optional[Dict]:
        """Detect face using OpenCV Haar Cascade, return largest face region"""
        try:
            if not hasattr(self, "_haar_cascade"):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self._haar_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(80, 80)
            )
            if len(faces) == 0:
                return None
            # Take the largest face (filter out small noise boxes)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        except Exception:
            return None

    def _analyze_deepface_with_roi(self, frame, haar_box: Dict) -> Optional[Dict[str, Any]]:
        """Run DeepFace in skip mode on Haar-detected ROI"""
        from deepface import DeepFace
        try:
            x, y, w, h = haar_box["x"], haar_box["y"], haar_box["w"], haar_box["h"]
            # Slightly expand ROI
            pad = int(min(w, h) * 0.1)
            fh, fw = frame.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(fw, x + w + pad)
            y2 = min(fh, y + h + pad)
            roi = frame[y1:y2, x1:x2]
            results = DeepFace.analyze(
                roi, actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False, silent=True,
            )
            r = results[0] if isinstance(results, list) else results
            emotions = r.get("emotion", {})
            top_emotion = r.get("dominant_emotion", "")
            if not top_emotion:
                return None
            confidence = emotions.get(top_emotion, 0) / 100.0
            if confidence < 0.08:
                return None
            return {
                "emotion": self.emotion_labels.get(top_emotion, top_emotion),
                "emotion_en": top_emotion,
                "confidence": round(confidence, 3),
                "all_scores": {self.emotion_labels.get(k, k): round(v / 100, 3) for k, v in emotions.items()},
                "box": haar_box,
                "detector": "haar+skip",
            }
        except Exception:
            return None

    def draw_result(self, frame, result: Optional[Dict]) -> Any:
        """Overlay emotion analysis results on the frame"""
        if frame is None:
            return frame
        if result:
            # Draw face bounding box
            box = result.get("box")
            if box:
                if isinstance(box, dict):  # deepface format
                    x, y, w, h = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
                else:  # fer format [x, y, w, h]
                    x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Main emotion label
            emotion_en = result.get("emotion_en", "")
            display_text = f"{emotion_en} {result['confidence']:.0%}"
            cv2.putText(frame, display_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            # All emotion scores (top-right, small font)
            y_offset = 20
            for label, score in result.get("all_scores", {}).items():
                bar_len = int(score * 80)
                cv2.rectangle(frame, (frame.shape[1] - 110, y_offset - 10),
                              (frame.shape[1] - 110 + bar_len, y_offset), (0, 200, 100), -1)
                cv2.putText(frame, f"{label[:3]}{score:.0%}", (frame.shape[1] - 110, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 16
        else:
            cv2.putText(frame, "No face detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # Bottom hint
        h = frame.shape[0]
        cv2.putText(frame, "Press ESC or Q to quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return frame
