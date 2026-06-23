from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
from ultralytics import YOLO


BBox = Tuple[int, int, int, int]


@dataclass
class FaceDetectorConfig:
    model_path: str = "yolov8n-face.pt"
    confidence_threshold: float = 0.35


class FaceDetector:
    """YOLOv8 face detector wrapper."""

    def __init__(self, config: FaceDetectorConfig | None = None) -> None:
        self.config = config or FaceDetectorConfig()
        self.model = None
        self.haar = None

        try:
            model_path = self._resolve_model_path(self.config.model_path)
            self.model = YOLO(str(model_path))
            self.mode = "yolo"
        except Exception as exc:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self.haar = cv2.CascadeClassifier(str(cascade_path))
            if self.haar.empty():
                raise RuntimeError(
                    "YOLO model unavailable and OpenCV Haar cascade could not be loaded."
                ) from exc
            self.mode = "haar"
            print(
                "Warning: YOLO face model unavailable. "
                "Falling back to OpenCV Haar cascade detector."
            )

    def _resolve_model_path(self, configured_path: str) -> Path:
        """Resolve local model path or download default face model on first run."""
        path = Path(configured_path)
        if path.exists():
            return path

        if path.name == "yolov8n-face.pt":
            cache_dir = Path.home() / ".face_blur_models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_model = cache_dir / path.name
            if cached_model.exists():
                return cached_model

            model_urls = [
                "https://github.com/andrisan/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
                "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
            ]

            for url in model_urls:
                try:
                    with urllib.request.urlopen(url, timeout=30) as response:
                        with open(cached_model, "wb") as out_file:
                            shutil.copyfileobj(response, out_file)
                    return cached_model
                except Exception:
                    continue

            raise FileNotFoundError(
                "Could not download 'yolov8n-face.pt' automatically. "
                "Please download it manually and place it in the project directory, "
                "or pass a custom model path in detector configuration."
            )

        raise FileNotFoundError(f"Model file not found: {path}")

    def detect(self, frame) -> List[BBox]:
        """Return detected face boxes as (x, y, w, h)."""
        boxes: List[BBox] = []
        if self.mode == "yolo":
            results = self.model.predict(
                source=frame,
                conf=self.config.confidence_threshold,
                verbose=False,
                device="cpu",
            )

            if not results:
                return boxes

            h, w = frame.shape[:2]
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                if bw > 0 and bh > 0:
                    boxes.append((x1, y1, bw, bh))
            return boxes

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in detected:
            boxes.append((int(x), int(y), int(w), int(h)))

        return boxes