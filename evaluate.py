import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.append(str(ROOT_DIR))
sys.path.append(str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

from config import BATCH_SIZE, IMG_SIZE, MODEL_PATH, REPORTS_DIR
from scripts.train import make_tf_dataset, prepare_hf_dataset
from src.folder_dataset import gather_labeled_image_paths, make_stratified_image_dataset, stratified_train_val_split
from src.utils import load_class_names


def _has_internal_mobilenet_preprocess(model) -> bool:
    try:
        return any(layer.name == "mobilenet_preprocess" for layer in model.layers)
    except Exception:
        return False


def _has_internal_imagenet_norm_m1p1(model) -> bool:
    try:
        return any(getattr(layer, "name", "") == "imagenet_norm_m1p1" for layer in model.layers)
    except Exception:
        return False


def _make_folder_eval_dataset(data_dir: Path, subset: str, val_split: float, seed: int, batch_size: int):
    if subset not in {"training", "validation"}:
        raise ValueError("subset must be 'training' or 'validation'")

    paths, labels, class_names = gather_labeled_image_paths(data_dir)
    train_paths, train_labels, val_paths, val_labels = stratified_train_val_split(
        paths=paths,
        labels=labels,
        val_split=float(val_split),
        seed=int(seed),
    )

    if subset == "training":
        split_paths, split_labels = train_paths, train_labels
    else:
        split_paths, split_labels = val_paths, val_labels

    ds = make_stratified_image_dataset(
        paths=split_paths,
        labels=split_labels,
        image_size=int(IMG_SIZE),
        batch_size=int(batch_size),
        shuffle=False,
        seed=int(seed),
    )

    print("[INFO] Folder dataset:", data_dir)
    print("[INFO] Subset:", subset, f"(val_split={val_split}, seed={seed})")
    print("[INFO] Classes:", class_names)
    print("[INFO] Subset size:", len(split_paths))

    return ds, class_names


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved Keras model.")
    parser.add_argument("--mode", type=str, default="hf", choices=["hf", "folder"])
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path("data_tm") / "rafdb7_train_facecrop_gray_clahe"),
        help="Used when --mode folder",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="validation",
        choices=["training", "validation"],
        help="Used when --mode folder (must match how you split during training).",
    )
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[INFO] Loading model:", MODEL_PATH)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            # Lambda layers often serialize the function as name "preprocess_input"
            "preprocess_input": mobilenet_v2_preprocess_input,
            # Some models serialize the actual symbol name
            "mobilenet_v2_preprocess_input": mobilenet_v2_preprocess_input,
        },
        safe_mode=False,
    )

    internal_preprocess = _has_internal_mobilenet_preprocess(model)
    internal_m1p1 = _has_internal_imagenet_norm_m1p1(model)

    if args.mode == "folder":
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

        val_tf, folder_class_names = _make_folder_eval_dataset(
            data_dir=data_dir,
            subset=str(args.subset),
            val_split=float(args.val_split),
            seed=int(args.seed),
            batch_size=int(BATCH_SIZE),
        )

        disk_class_names = load_class_names()
        if disk_class_names != folder_class_names:
            print("[WARNING] models/class_names.json != folder class order.")
            print("[WARNING] - disk:", disk_class_names)
            print("[WARNING] - folder:", folder_class_names)
            print("[WARNING] Using folder class order for reporting (recommended: retrain or fix json).")

        class_names = folder_class_names

        def _prepare_batch(images, labels):
            # Folder pipeline yields float32 RGB in [0, 255].
            if internal_preprocess:
                images = tf.clip_by_value(images, 0.0, 255.0)
                return images, labels

            if internal_m1p1:
                # Model already contains `imagenet_norm_m1p1` -> do NOT rescale here.
                images = tf.clip_by_value(images, 0.0, 255.0)
                return images, labels

            model_name = str(getattr(model, "name", "")).lower()
            if "mobilenetv3" in model_name or "mobilenet_v3" in model_name:
                images = images / 127.5 - 1.0
                images = tf.clip_by_value(images, -1.0, 1.0)
                return images, labels

            images = mobilenet_v2_preprocess_input(images)
            return images, labels

        val_tf = val_tf.map(_prepare_batch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    else:
        _, val_hf, image_col, label_col, _ = prepare_hf_dataset()
        class_names = load_class_names()
        val_tf = make_tf_dataset(val_hf, image_col, label_col, training=False)

    y_true = []
    y_pred = []

    print("[INFO] Predicting evaluation set...")

    for images, labels in val_tf:
        predictions = model.predict(images, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(pred_labels.tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(report)

    suffix = "_folder" if args.mode == "folder" else ""
    report_path = REPORTS_DIR / f"classification_report{suffix}.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    cm_path = REPORTS_DIR / f"confusion_matrix{suffix}.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)

    print(f"[INFO] Report saved to: {report_path}")
    print(f"[INFO] Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()