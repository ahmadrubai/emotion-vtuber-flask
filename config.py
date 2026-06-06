from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_NAME = "deanngkl/raf-db-7emotions"

# For VTuber-style output, we merge visually-similar classes into one label.
# NOTE: after changing this, regenerate training data folders + retrain the model.
MERGED_EMOTION_NAME = "ick"
MERGED_EMOTION_SOURCES = ("fear", "disgust")

DEFAULT_CLASSES = [
    # Fallback if `models/class_names.json` doesn't exist yet.
    "anger",
    MERGED_EMOTION_NAME,
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

IMG_SIZE = 160

# Ukuran input model yang dipakai Flask / inferensi (harus sama dengan model di MODEL_PATH).
MODEL_INPUT_SIZE = 224

# True = preprocessing seperti export Teachable Machine / snippet Keras:
# ImageOps.fit(..., LANCZOS) lalu (pixel.astype(float32) / 127.5) - 1
MODEL_USE_TM_PREPROCESSING = True

# False = inferensi pakai gambar/frame utuh, cocok untuk dataset Teachable Machine yang tidak dicrop wajah.
MODEL_USE_FACE_CROP = False

# Dipakai hanya kalau MODEL_USE_TM_PREPROCESSING = False.
# Pilihan: "rescale_0_1" atau "raw_0_255".
MODEL_PREPROCESSING_MODE = "rescale_0_1"
BATCH_SIZE = 32

EPOCHS_HEAD = 15
EPOCHS_FINE_TUNE = 20
FINE_TUNE_LAST_N = 40

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

# Dataset balancing strategy for training.
# - "oversample": sample classes uniformly (good for minority recall)
# - "downsample": keep only N samples per class (N = min class count) (fast + balanced)
# - "none": use raw distribution
BALANCE_MODE = "downsample"  # "oversample" | "downsample" | "none"
BALANCE_SEED = 42

CONFIDENCE_THRESHOLD = 0.65

# Jumlah ranking emosi (prob tertinggi ke-N) di `/emotion_state` — tanpa jeda/threshold.
EMOTION_TOP_N = 5

# Dinonaktifkan: inferensi realtime pakai argmax per frame (lihat `EMOTION_TOP_N`).
SMOOTHING_WINDOW = 10
MIN_VOTES = 6

CAMERA_INDEX = 0

# MJPEG `/video_feed`: batasi FPS + lebar preview supaya halaman debug tidak ngelag.
VIDEO_STREAM_TARGET_FPS = 15
VIDEO_PREVIEW_MAX_WIDTH = 854

MODEL_DIR = BASE_DIR / "models"
KERAS_DIR = BASE_DIR / "Keras"
MODEL_PATH = KERAS_DIR / "keras_model.h5"
KERAS_LABELS_PATH = KERAS_DIR / "labels.txt"
ACTIVE_MODEL_PATH_FILE = KERAS_DIR / "active_model_path.txt"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

STATIC_DIR = BASE_DIR / "static"
ASSETS_DIR = STATIC_DIR / "assets"

REPORTS_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
KERAS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
