import cv2
from pathlib import Path

from src.utils import clamp_bbox


class MediaPipeFaceDetector:
    """
    Keep class name for compatibility with `src.camera`.

    The environment's `mediapipe` package does not expose `mp.solutions`,
    so we use a stable OpenCV Haar-cascade face detector instead.
    """

    def __init__(
        self,
        min_detection_confidence=0.5,  # kept for signature compatibility
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (40, 40),
    ):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

        # Heuristics for webcam-sized faces
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = (int(min_size[0]), int(min_size[1]))

    def detect_largest_face(self, frame_bgr):
        frame_height, frame_width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )

        if rects is None or len(rects) == 0:
            return None

        best_bbox = None
        best_area = 0

        for (x, y, w, h) in rects:
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.20)

            x -= margin_x
            y -= margin_y
            w += margin_x * 2
            h += margin_y * 2

            x, y, w, h = clamp_bbox(x, y, w, h, frame_width, frame_height)
            area = w * h

            if area > best_area:
                best_area = area
                best_bbox = (x, y, w, h)

        return best_bbox