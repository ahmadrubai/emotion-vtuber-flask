import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import (
    EMOTION_TOP_N,
    KERAS_DIR,
    MERGED_EMOTION_NAME,
    MODEL_PREPROCESSING_MODE,
    MODEL_USE_TM_PREPROCESSING,
    MODEL_USE_FACE_CROP,
    VIDEO_STREAM_TARGET_FPS,
)
from src.camera import CameraProcessor
from src.emotion_model import EmotionModel
from src.utils import (
    ALLOWED_KERAS_MODEL_EXTENSIONS,
    load_class_names,
    load_labels_from_upload,
    ranked_emotions,
    save_labels_txt,
    set_active_model_path,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

camera = CameraProcessor()


def _json_model_status():
    output_count = camera.emotion_model.get_output_count() if camera.emotion_model.is_loaded else 0
    return {
        "model_loaded": camera.emotion_model.is_loaded,
        "model_path": camera.emotion_model.model_path,
        "class_names": camera.emotion_model.class_names,
        "class_count": len(camera.emotion_model.class_names),
        "output_count": output_count,
        "use_face_crop": MODEL_USE_FACE_CROP,
        "use_tm_preprocessing": MODEL_USE_TM_PREPROCESSING,
        "preprocessing_mode": MODEL_PREPROCESSING_MODE,
    }


def _remove_file_if_exists(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


@app.before_request
def start_camera_once():
    if request.path.startswith("/static"):
        return
    if request.endpoint not in {"avatar", "debug", "emotion_state", "video_feed"}:
        return
    if not camera.running:
        try:
            camera.start()
        except Exception as e:
            # In dev environment webcam may be unavailable; keep server running.
            print(f"[WARNING] Camera start failed: {e}")
            camera.set_start_error(str(e))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/avatar")
def avatar():
    return render_template("avatar.html", merged_emotion=MERGED_EMOTION_NAME)


@app.route("/debug")
def debug():
    return render_template("debug.html")


@app.route("/model_test")
def model_test():
    return render_template("model_test.html")


@app.route("/model_status")
def model_status():
    return jsonify(_json_model_status())


@app.route("/upload_model", methods=["POST"])
def upload_model():
    uploaded_model = request.files.get("model")
    uploaded_labels = request.files.get("labels")

    if uploaded_model is None or uploaded_model.filename == "":
        return jsonify({"error": "Pilih file model Keras dulu."}), 400

    original_name = secure_filename(uploaded_model.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_KERAS_MODEL_EXTENSIONS:
        return jsonify({"error": "Format model harus .h5 atau .keras."}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = KERAS_DIR / f"uploaded_{timestamp}{extension}"
    uploaded_model.save(model_path)

    try:
        candidate_labels = (
            load_labels_from_upload(uploaded_labels)
            if uploaded_labels and uploaded_labels.filename
            else load_class_names()
        )

        if not candidate_labels:
            _remove_file_if_exists(model_path)
            return jsonify({"error": "Label kosong. Upload labels.txt atau siapkan Keras/labels.txt."}), 400

        candidate_model = EmotionModel(model_path=model_path)
        if not candidate_model.is_loaded:
            _remove_file_if_exists(model_path)
            return jsonify({"error": "Model gagal diload. Pastikan file .h5/.keras valid."}), 400

        output_count = candidate_model.get_output_count()
        if output_count != len(candidate_labels):
            _remove_file_if_exists(model_path)
            return jsonify({
                "error": (
                    f"Jumlah output model ({output_count}) tidak sama dengan jumlah label "
                    f"({len(candidate_labels)}). Upload labels.txt yang cocok."
                )
            }), 400

        if uploaded_labels and uploaded_labels.filename:
            save_labels_txt(candidate_labels)

        set_active_model_path(model_path)
        camera.emotion_model.reload(model_path=model_path)

        with camera.lock:
            camera.state["model_loaded"] = camera.emotion_model.is_loaded

        return jsonify({
            "message": "Model berhasil diupload dan langsung aktif.",
            **_json_model_status(),
        })
    except Exception as error:
        _remove_file_if_exists(model_path)
        return jsonify({"error": str(error)}), 400


@app.route("/emotion_state")
def emotion_state():
    return jsonify(camera.get_state())


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "File image tidak ditemukan."}), 400

    uploaded = request.files["image"]
    if uploaded.filename == "":
        return jsonify({"error": "Pilih file gambar dulu."}), 400

    file_bytes = uploaded.read()
    if not file_bytes:
        return jsonify({"error": "File gambar kosong."}), 400

    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Format gambar tidak valid."}), 400

    face_detected = False
    image_for_prediction = image_bgr

    emotion, confidence, probabilities = camera.emotion_model.predict(image_for_prediction)
    sorted_probabilities = dict(
        sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    )

    return jsonify({
        "emotion": emotion,
        "confidence": confidence,
        "top_n": ranked_emotions(probabilities, EMOTION_TOP_N),
        "probabilities": sorted_probabilities,
        "face_detected": face_detected,
        "used_face_crop": False,
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "camera_running": camera.running,
    })


def generate_video_stream():
    min_interval = 1.0 / max(1, int(VIDEO_STREAM_TARGET_FPS))
    last_sent = 0.0
    while True:
        frame = camera.get_jpeg()

        if frame is None:
            time.sleep(0.04)
            continue

        now = time.monotonic()
        elapsed = now - last_sent
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            now = time.monotonic()

        last_sent = now
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    try:
        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True,
        )
    finally:
        camera.stop()
