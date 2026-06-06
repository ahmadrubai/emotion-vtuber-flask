import threading
import time

import cv2

from config import CAMERA_INDEX, EMOTION_TOP_N, VIDEO_PREVIEW_MAX_WIDTH
from src.emotion_model import EmotionModel
from src.utils import draw_label, ranked_emotions


class CameraProcessor:
    def __init__(self):
        self.camera_index = CAMERA_INDEX

        self.emotion_model = EmotionModel()

        self.capture = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_jpeg = None

        self.state = {
            "emotion": "neutral",
            "raw_emotion": "neutral",
            "confidence": 0.0,
            "top_n": [],
            "face_detected": False,
            "used_face_crop": False,
            "fps": 0.0,
            "camera_running": False,
            "stream_ready": False,
            "camera_start_error": None,
            "model_loaded": self.emotion_model.is_loaded,
        }

    def start(self):
        if self.running:
            return

        # Webcam backend/index can differ on Windows. Try a small set of indexes/backends.
        candidate_indexes = [self.camera_index, self.camera_index + 1, self.camera_index - 1, 0, 1, 2, 3]
        candidate_indexes = [i for i in candidate_indexes if i >= 0]
        candidate_backends = [cv2.CAP_ANY]

        self.capture = None
        last_error = None
        for idx in candidate_indexes:
            for backend in candidate_backends:
                cap = cv2.VideoCapture(idx, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                if cap.isOpened():
                    self.capture = cap
                    break
            if self.capture is not None:
                break

        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(
                "Webcam tidak bisa dibuka. Cek CAMERA_INDEX di config.py (atau izin webcam)."
            ) from last_error

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        with self.lock:
            self.state["camera_running"] = True
            self.state["camera_start_error"] = None

        print("[INFO] Camera started.")

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.capture:
            self.capture.release()
        with self.lock:
            self.state["camera_running"] = False
            self.state["stream_ready"] = False

        print("[INFO] Camera stopped.")

    def _loop(self):
        last_time = time.time()

        while self.running:
            success, frame = self.capture.read()

            if not success:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - last_time
            last_time = now

            fps = 1.0 / dt if dt > 0 else 0.0

            processed_frame, new_state = self._process_frame(frame, fps)

            stream_frame = processed_frame
            max_w = int(VIDEO_PREVIEW_MAX_WIDTH)
            if max_w > 0:
                h, w = stream_frame.shape[:2]
                if w > max_w:
                    new_w = max_w
                    new_h = max(1, int(round(h * (new_w / w))))
                    stream_frame = cv2.resize(
                        stream_frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )

            ok, jpeg = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 72])

            with self.lock:
                self.latest_frame = processed_frame
                self.latest_jpeg = jpeg.tobytes() if ok else None
                self.state.update(new_state)
                self.state["camera_running"] = self.running
                self.state["stream_ready"] = self.latest_jpeg is not None

    def _process_frame(self, frame, fps):
        display_frame = frame.copy()

        raw_emotion = "neutral"
        emotion = "neutral"
        confidence = 0.0
        top_n = []
        face_detected = False
        used_face_crop = False
        image_for_prediction = frame

        if image_for_prediction is not None:
            raw_emotion, confidence, probabilities = self.emotion_model.predict(image_for_prediction)
            emotion = raw_emotion
            top_n = ranked_emotions(probabilities, EMOTION_TOP_N)

            top_preview = " | ".join(
                f"{item['emotion']} {item['confidence']:.2f}" for item in top_n[:3]
            )
            label = top_preview or f"{emotion} {confidence:.2f}"
            draw_label(display_frame, label, 20, 80)

        debug_text = f"FPS: {fps:.1f} | Emotion: {emotion}"
        cv2.putText(
            display_frame,
            debug_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        new_state = {
            "emotion": emotion,
            "raw_emotion": raw_emotion,
            "confidence": confidence,
            "top_n": top_n,
            "face_detected": face_detected,
            "used_face_crop": used_face_crop,
            "fps": fps,
            "model_loaded": self.emotion_model.is_loaded,
        }

        return display_frame, new_state

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_state(self):
        with self.lock:
            state = dict(self.state)
            state["camera_running"] = self.running
            state["stream_ready"] = self.latest_jpeg is not None
            return state

    def set_start_error(self, error_message):
        with self.lock:
            self.running = False
            self.state["camera_running"] = False
            self.state["stream_ready"] = False
            self.state["camera_start_error"] = error_message
