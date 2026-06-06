import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
import tensorflow as tf
from datasets import DatasetDict, load_dataset
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import (
    BALANCE_MODE,
    BALANCE_SEED,
    BATCH_SIZE,
    DATASET_NAME,
    EPOCHS_FINE_TUNE,
    EPOCHS_HEAD,
    FINE_TUNE_LAST_N,
    IMG_SIZE,
    LEARNING_RATE_FINE_TUNE,
    LEARNING_RATE_HEAD,
    MODEL_PATH,
)
from src.utils import save_class_names


def downsample_to_min_class(hf_dataset, label_col: str, seed: int):
    """
    Create a balanced subset by taking N=min_count samples per class.
    Deterministic by `seed`. Keeps only the training split balanced; validation stays untouched.
    """
    labels = [int(x) for x in hf_dataset[label_col]]
    num_classes = int(max(labels) + 1) if labels else 0

    indices_by_class: dict[int, list[int]] = {i: [] for i in range(num_classes)}
    for idx, lab in enumerate(labels):
        if 0 <= lab < num_classes:
            indices_by_class[lab].append(idx)

    counts = {k: len(v) for k, v in indices_by_class.items()}
    if not counts:
        return hf_dataset

    min_count = min(counts.values()) if counts else 0
    if min_count <= 0:
        return hf_dataset

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for class_id, idxs in indices_by_class.items():
        if not idxs:
            continue
        idxs = list(idxs)
        rng.shuffle(idxs)
        chosen.extend(idxs[:min_count])

    rng.shuffle(chosen)
    balanced = hf_dataset.select(chosen)

    print("[INFO] Downsample balanced training set:")
    print("[INFO] - per_class:", min_count)
    print("[INFO] - total:", len(balanced), "from", len(hf_dataset))

    return balanced


def print_device_info():
    print("[INFO] TensorFlow:", tf.__version__)
    print("[INFO] GPU:", tf.config.list_physical_devices("GPU"))


def _infer_columns(ds):
    image_col = None
    label_col = None

    for name, feature in ds.features.items():
        ft = str(feature)
        if image_col is None and "Image" in ft:
            image_col = name
        if label_col is None and "ClassLabel" in ft:
            label_col = name

    if image_col is None:
        sample = ds[0]
        for key, value in sample.items():
            if hasattr(value, "convert"):
                image_col = key
                break

    if label_col is None:
        if "label" in ds.column_names:
            label_col = "label"
        elif "labels" in ds.column_names:
            label_col = "labels"

    if image_col is None or label_col is None:
        raise RuntimeError(f"Gagal mendeteksi kolom image/label. Columns: {ds.column_names}")

    return image_col, label_col


def _infer_class_names(ds, label_col):
    feature = ds.features[label_col]
    if hasattr(feature, "names") and feature.names:
        return list(feature.names)

    labels = sorted(set(ds[label_col]))
    return [str(label) for label in labels]


def prepare_hf_dataset():
    print(f"[INFO] Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    splits = list(dataset.keys())
    print("[INFO] Available splits:", splits)

    base_split = "train" if "train" in dataset else splits[0]

    image_col, label_col = _infer_columns(dataset[base_split])
    class_names = _infer_class_names(dataset[base_split], label_col)

    print("[INFO] Image column:", image_col)
    print("[INFO] Label column:", label_col)
    print("[INFO] Class names:", class_names)

    if "train" in dataset and "validation" in dataset:
        train_ds = dataset["train"]
        val_ds = dataset["validation"]
    elif "train" in dataset and "test" in dataset:
        train_ds = dataset["train"]
        val_ds = dataset["test"]
    else:
        split = dataset[base_split].train_test_split(
            test_size=0.2,
            seed=42,
            stratify_by_column=label_col,
        )
        train_ds = split["train"]
        val_ds = split["test"]

    return train_ds, val_ds, image_col, label_col, class_names


def make_tf_dataset(hf_dataset, image_col, label_col, training=False):
    def sequential_generator():
        for example in hf_dataset:
            image = example[image_col].convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))

            image_array = np.asarray(image, dtype=np.float32)
            label = int(example[label_col])

            yield image_array, label

    def balanced_generator(num_classes: int):
        indices_by_class: dict[int, list[int]] = {i: [] for i in range(num_classes)}

        for idx, example in enumerate(hf_dataset):
            label = int(example[label_col])
            if 0 <= label < num_classes:
                indices_by_class[label].append(idx)

        empty = [k for k, v in indices_by_class.items() if not v]
        if empty:
            print("[WARNING] Empty classes (no samples):", empty)

        rng = np.random.default_rng(42)

        # Infinite stream: sample classes uniformly, then sample within class.
        while True:
            class_id = int(rng.integers(0, num_classes))
            pool = indices_by_class.get(class_id) or []
            if not pool:
                continue

            example = hf_dataset[int(rng.choice(pool))]

            image = example[image_col].convert("RGB")
            image = image.resize((IMG_SIZE, IMG_SIZE))

            image_array = np.asarray(image, dtype=np.float32)
            label = int(example[label_col])

            yield image_array, label

    output_signature = (
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    balanced_sampling = bool(training and BALANCE_MODE == "oversample")

    if balanced_sampling:
        # For imbalanced emotion datasets, this helps minority classes a lot.
        num_classes = int(max(hf_dataset[label_col]) + 1)
        ds = tf.data.Dataset.from_generator(
            lambda: balanced_generator(num_classes),
            output_signature=output_signature,
        ).repeat()
    else:
        ds = tf.data.Dataset.from_generator(
            sequential_generator,
            output_signature=output_signature,
        )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    x = data_augmentation(inputs)
    x = layers.Lambda(preprocess_input, name="mobilenet_preprocess")(x)

    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.35, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = models.Model(inputs, outputs, name="emotion_mobilenetv2")
    return model, base_model


def calculate_class_weight(hf_dataset, label_col, num_classes):
    labels = np.array([int(example[label_col]) for example in hf_dataset])

    classes = np.arange(num_classes)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    class_weight = {
        int(class_index): float(weight)
        for class_index, weight in zip(classes, weights)
    }

    print("[INFO] Class weight:", class_weight)

    return class_weight


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
    print_device_info()

    train_hf, val_hf, image_col, label_col, class_names = prepare_hf_dataset()
    save_class_names(class_names)

    num_classes = len(class_names)

    if BALANCE_MODE == "downsample":
        train_hf = downsample_to_min_class(train_hf, label_col=label_col, seed=BALANCE_SEED)

    print("[INFO] Train size:", len(train_hf))
    print("[INFO] Val size:", len(val_hf))
    print("[INFO] Num classes:", num_classes)

    train_tf = make_tf_dataset(train_hf, image_col, label_col, training=True)
    val_tf = make_tf_dataset(val_hf, image_col, label_col, training=False)

    # NOTE: With balanced sampling enabled, class_weight usually over-compensates.
    class_weight = None

    model, base_model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    print("\n[INFO] Stage 1: Training classifier head...")
    model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=EPOCHS_HEAD,
        steps_per_epoch=int(np.ceil(len(train_hf) / BATCH_SIZE)),
        callbacks=get_callbacks(),
    )

    print("\n[INFO] Stage 2: Fine-tuning MobileNetV2...")

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
        train_tf,
        validation_data=val_tf,
        epochs=EPOCHS_FINE_TUNE,
        steps_per_epoch=int(np.ceil(len(train_hf) / BATCH_SIZE)),
        callbacks=get_callbacks(),
    )

    model.save(MODEL_PATH)
    print(f"[INFO] Final model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()