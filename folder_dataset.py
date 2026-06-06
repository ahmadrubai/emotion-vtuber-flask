from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def gather_labeled_image_paths(data_dir: Path) -> tuple[list[str], list[int], list[str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found under: {data_dir}")

    class_names = [p.name for p in class_dirs]
    name_to_index = {name: i for i, name in enumerate(class_names)}

    paths: list[str] = []
    labels: list[int] = []

    for class_dir in class_dirs:
        label = int(name_to_index[class_dir.name])
        for img_path in sorted([p for p in class_dir.iterdir() if p.is_file()]):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            paths.append(str(img_path))
            labels.append(label)

    if not paths:
        raise RuntimeError(f"No images found under: {data_dir}")

    return paths, labels, class_names


def stratified_train_val_split(
    paths: list[str],
    labels: list[int],
    val_split: float,
    seed: int,
) -> tuple[list[str], list[int], list[str], list[int]]:
    y = np.array(labels, dtype=np.int32)
    train_paths, val_paths, train_y, val_y = train_test_split(
        paths,
        y,
        test_size=float(val_split),
        random_state=int(seed),
        stratify=y,
        shuffle=True,
    )

    return train_paths, train_y.tolist(), val_paths, val_y.tolist()


def _read_image_rgb_uint8(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.uint8)


def make_stratified_image_dataset(
    paths: list[str],
    labels: list[int],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(np.array(labels, dtype=np.int32))
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(min(len(paths), 4096), seed=int(seed), reshuffle_each_iteration=True)

    def _load(path_tensor, label_tensor):
        path_str = path_tensor.numpy().decode("utf-8")
        image_rgb = _read_image_rgb_uint8(path_str)
        image = tf.convert_to_tensor(image_rgb)
        image = tf.image.resize(image, (int(image_size), int(image_size)), method="bilinear")
        image = tf.cast(image, tf.float32)
        return image, label_tensor

    def _load_and_set_shape(path_tensor, label_tensor):
        image, label = tf.py_function(func=_load, inp=[path_tensor, label_tensor], Tout=[tf.float32, tf.int32])
        image.set_shape((int(image_size), int(image_size), 3))
        label.set_shape(())
        label = tf.cast(label, tf.int32)
        return image, label

    ds = ds.map(_load_and_set_shape, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(int(batch_size)).prefetch(tf.data.AUTOTUNE)
    return ds
