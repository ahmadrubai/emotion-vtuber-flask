import argparse
import json
from pathlib import Path

import tensorflow as tf

from config import CLASS_NAMES_PATH, MODEL_DIR, MODEL_PATH


def _read_labels(labels_path: Path) -> list[str]:
    lines = labels_path.read_text(encoding="utf-8").splitlines()
    labels: list[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Teachable Machine often exports: "0 label"
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels.append(parts[1].strip())
        else:
            labels.append(line)

    if not labels:
        raise RuntimeError(f"labels.txt kosong: {labels_path}")

    return labels


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train v2 (Teachable Machine): import keras_model.h5 + labels.txt "
            "dan simpan jadi models/emotion_mobilenetv2.keras + class_names.json."
        )
    )
    parser.add_argument(
        "--tm_dir",
        type=str,
        default=str(MODEL_DIR / "teachable_machine"),
        help="Folder export Teachable Machine (isi: keras_model.h5 dan labels.txt).",
    )
    parser.add_argument(
        "--h5",
        type=str,
        default="keras_model.h5",
        help="Nama/Path file .h5 (relative terhadap --tm_dir jika bukan absolute).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="labels.txt",
        help="Nama/Path labels.txt (relative terhadap --tm_dir jika bukan absolute).",
    )
    parser.add_argument(
        "--out_model",
        type=str,
        default=str(MODEL_PATH),
        help="Output path model .keras.",
    )
    parser.add_argument(
        "--out_labels",
        type=str,
        default=str(CLASS_NAMES_PATH),
        help="Output path class_names.json.",
    )
    args = parser.parse_args()

    tm_dir = Path(args.tm_dir)
    h5_path = Path(args.h5)
    labels_path = Path(args.labels)

    if not h5_path.is_absolute():
        h5_path = tm_dir / h5_path
    if not labels_path.is_absolute():
        labels_path = tm_dir / labels_path

    out_model = Path(args.out_model)
    out_labels = Path(args.out_labels)

    if not h5_path.exists():
        raise FileNotFoundError(
            "File model Teachable Machine tidak ketemu. "
            f"Harus ada: {h5_path}\n"
            "Download dari Teachable Machine: Export Model → Tensorflow → Keras."
        )

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.txt tidak ketemu: {labels_path}")

    print("[INFO] Teachable Machine dir:", tm_dir)
    print("[INFO] Loading TM model:", h5_path)

    # TM Keras export is typically a plain Keras H5 model.
    model = tf.keras.models.load_model(h5_path, compile=False, safe_mode=False)

    labels = _read_labels(labels_path)
    print("[INFO] Labels:", labels)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_labels.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Saving model to:", out_model)
    model.save(out_model)

    print("[INFO] Saving class names to:", out_labels)
    out_labels.write_text(json.dumps(labels, indent=4), encoding="utf-8")

    print("[OK] Import selesai.")
    print("[NEXT] Jalankan evaluate:")
    print("       python .\\scripts\\evaluate.py")


if __name__ == "__main__":
    main()

