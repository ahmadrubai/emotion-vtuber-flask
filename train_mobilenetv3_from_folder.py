import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config import (
    BATCH_SIZE,
    EPOCHS_FINE_TUNE,
    EPOCHS_HEAD,
    FINE_TUNE_LAST_N,
    IMG_SIZE,
    LEARNING_RATE_FINE_TUNE,
    LEARNING_RATE_HEAD,
    MODEL_PATH,
)
from src.folder_dataset import gather_labeled_image_paths, make_stratified_image_dataset, stratified_train_val_split
from src.utils import save_class_names


def print_device_info():
    print("[INFO] TensorFlow:", tf.__version__)
    print("[INFO] GPU:", tf.config.list_physical_devices("GPU"))


def make_datasets(data_dir: Path, val_split: float, seed: int):
    paths, labels, class_names = gather_labeled_image_paths(data_dir)
    train_paths, train_labels, val_paths, val_labels = stratified_train_val_split(
        paths=paths,
        labels=labels,
        val_split=float(val_split),
        seed=int(seed),
    )

    print("[INFO] Classes:", class_names)
    print("[INFO] Total images:", len(paths))
    print("[INFO] Train images:", len(train_paths))
    print("[INFO] Val images:", len(val_paths))

    train_ds = make_stratified_image_dataset(
        paths=train_paths,
        labels=train_labels,
        image_size=int(IMG_SIZE),
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        seed=int(seed),
    )

    val_ds = make_stratified_image_dataset(
        paths=val_paths,
        labels=val_labels,
        image_size=int(IMG_SIZE),
        batch_size=int(BATCH_SIZE),
        shuffle=False,
        seed=int(seed),
    )

    return train_ds, val_ds, class_names


def compute_class_weight_from_dataset(train_ds, num_classes: int):
    labels = []
    for _, y in train_ds:
        labels.append(y.numpy())

    if not labels:
        return None

    y = np.concatenate(labels, axis=0)
    classes = np.arange(num_classes)

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {int(i): float(w) for i, w in zip(classes, weights)}
    print("[INFO] Class weight:", class_weight)
    return class_weight


def build_model(num_classes: int):
    # NOTE: In newer Keras/TF versions, `mobilenet_v3_preprocess_input` is a no-op placeholder.
    # MobileNetV3 expects inputs scaled roughly to [-1, 1] when `include_preprocessing=False`.
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.08),
            layers.RandomTranslation(0.06, 0.06),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="imagenet_norm_m1p1")(x)

    base_model = MobileNetV3Large(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        minimalistic=False,
        include_preprocessing=False,
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.35, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = models.Model(inputs, outputs, name="emotion_mobilenetv3_large")
    return model, base_model


def get_callbacks():
    return [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def freeze_batch_norm_layers(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3Large from a folder dataset (class subfolders).")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path("data_tm") / "rafdb7_train_facecrop_gray_clahe"),
        help="Root folder containing class subfolders.",
    )
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    print_device_info()

    train_ds, val_ds, class_names = make_datasets(data_dir, val_split=float(args.val_split), seed=int(args.seed))
    save_class_names(class_names)

    num_classes = len(class_names)
    print("[INFO] Num classes:", num_classes)

    class_weight = compute_class_weight_from_dataset(train_ds, num_classes=num_classes)

    model, base_model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    print("\n[INFO] Stage 1: Training classifier head...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weight,
        callbacks=get_callbacks(),
    )

    print("\n[INFO] Stage 2: Fine-tuning MobileNetV3Large...")
    base_model.trainable = True

    freeze_until = max(0, len(base_model.layers) - FINE_TUNE_LAST_N)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    freeze_batch_norm_layers(base_model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        class_weight=class_weight,
        callbacks=get_callbacks(),
    )

    model.save(MODEL_PATH)
    print(f"[INFO] Final model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
