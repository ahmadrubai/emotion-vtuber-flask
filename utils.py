import json
from pathlib import Path

import cv2

from config import (
    ACTIVE_MODEL_PATH_FILE,
    CLASS_NAMES_PATH,
    DEFAULT_CLASSES,
    KERAS_LABELS_PATH,
    MODEL_PATH,
)


ALLOWED_KERAS_MODEL_EXTENSIONS = {".h5", ".keras"}


def _load_labels_txt(path: Path):
    """Parse `labels.txt` lines like `0 Ang` -> ordered class names (lowercase)."""
    text = path.read_text(encoding="utf-8")
    indexed = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        indexed.append((int(parts[0]), parts[1].strip().lower()))
    indexed.sort(key=lambda item: item[0])
    return [name for _, name in indexed]


def save_labels_txt(class_names, path=KERAS_LABELS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"{index} {name}" for index, name in enumerate(class_names)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_labels_from_upload(file_storage):
    text = file_storage.read().decode("utf-8-sig")
    indexed = []
    plain_labels = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            indexed.append((int(parts[0]), parts[1].strip().lower()))
        else:
            plain_labels.append(line.lower())

    if indexed:
        indexed.sort(key=lambda item: item[0])
        return [name for _, name in indexed]

    return plain_labels


def resolve_active_model_path():
    active_path_file = Path(ACTIVE_MODEL_PATH_FILE)
    if active_path_file.exists():
        active_path = Path(active_path_file.read_text(encoding="utf-8").strip())
        if active_path.exists():
            return active_path

    return Path(MODEL_PATH)


def set_active_model_path(model_path):
    model_path = Path(model_path).resolve()
    Path(ACTIVE_MODEL_PATH_FILE).write_text(str(model_path), encoding="utf-8")


def save_class_names(class_names, path=CLASS_NAMES_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=4)


def load_class_names(path=CLASS_NAMES_PATH):
    keras_labels = Path(KERAS_LABELS_PATH)
    if keras_labels.exists():
        return _load_labels_txt(keras_labels)

    path = Path(path)

    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    return DEFAULT_CLASSES


def ranked_emotions(probabilities: dict, top_n: int):
    """Urut probabilitas menurun; `top_n` pertama untuk UI/API."""
    if not probabilities or top_n <= 0:
        return []
    items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [{"emotion": name, "confidence": float(score)} for name, score in items]


def draw_label(frame, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size

    cv2.rectangle(
        frame,
        (x, y - text_h - 12),
        (x + text_w + 10, y),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 5, y - 6),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def clamp_bbox(x, y, w, h, frame_width, frame_height):
    x = max(0, x)
    y = max(0, y)

    w = min(w, frame_width - x)
    h = min(h, frame_height - y)

    return x, y, w, h
