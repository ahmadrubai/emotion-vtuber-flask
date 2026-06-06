import argparse
from collections import defaultdict
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from datasets import DatasetDict, load_dataset

from config import DATASET_NAME


def _sanitize(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    return "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-", " ")).strip()


def _get_splits(ds):
    if isinstance(ds, DatasetDict):
        return ds
    return DatasetDict({"train": ds})


def _ensure_train_val(dataset: DatasetDict, label_col: str) -> tuple:
    if "train" in dataset and "validation" in dataset:
        return dataset["train"], dataset["validation"]
    if "train" in dataset and "test" in dataset:
        return dataset["train"], dataset["test"]

    base = "train" if "train" in dataset else list(dataset.keys())[0]
    split = dataset[base].train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column=label_col,
    )
    return split["train"], split["test"]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export dataset HuggingFace (mis. parquet/arrow) jadi folder per class "
            "untuk di-upload ke Teachable Machine."
        )
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("data_tm") / "rafdb7"),
        help="Output folder (akan dibuat).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "all"],
        help="Split yang diexport.",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="Batas jumlah gambar per class (0 = semua).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Format output gambar.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading dataset:", DATASET_NAME)
    dataset = _get_splits(load_dataset(DATASET_NAME))

    # RAF-DB 7 emotions biasanya image/label
    # Kita nggak pakai heuristik rumit di sini, cukup default umum.
    base = "train" if "train" in dataset else list(dataset.keys())[0]
    cols = dataset[base].column_names
    image_col = "image" if "image" in cols else cols[0]
    label_col = "label" if "label" in cols else cols[1]

    feature = dataset[base].features.get(label_col)
    class_names = None
    if feature is not None and hasattr(feature, "names") and feature.names:
        class_names = list(feature.names)

    train_ds, val_ds = _ensure_train_val(dataset, label_col=label_col)
    splits_to_export = []
    if args.split == "train":
        splits_to_export = [("train", train_ds)]
    elif args.split == "val":
        splits_to_export = [("val", val_ds)]
    else:
        splits_to_export = [("train", train_ds), ("val", val_ds)]

    for split_name, ds in splits_to_export:
        split_root = out_dir / split_name
        split_root.mkdir(parents=True, exist_ok=True)

        counts = defaultdict(int)
        total_saved = 0

        print(f"[INFO] Exporting split: {split_name} (len={len(ds)})")

        for i, example in enumerate(ds):
            label_id = int(example[label_col])
            label = (
                class_names[label_id]
                if class_names is not None and 0 <= label_id < len(class_names)
                else str(label_id)
            )
            label = _sanitize(label)

            if args.max_per_class and counts[label] >= args.max_per_class:
                continue

            img = example[image_col]
            # HF Image feature returns PIL.Image
            if hasattr(img, "convert"):
                img = img.convert("RGB")

            class_dir = split_root / label
            class_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{label}_{split_name}_{i:06d}.{args.format}"
            out_path = class_dir / filename

            if args.format == "jpg":
                img.save(out_path, format="JPEG", quality=95, optimize=True)
            else:
                img.save(out_path, format="PNG", optimize=True)

            counts[label] += 1
            total_saved += 1

            if total_saved % 500 == 0:
                print(f"[INFO] Saved {total_saved} images so far...")

        print(f"[OK] Split {split_name} saved: {total_saved} images → {split_root}")

    print("[NEXT] Upload ke Teachable Machine:")
    print("       - Buka Image Project")
    print("       - Untuk tiap class, upload folder dari out_dir\\train\\<class>")


if __name__ == "__main__":
    main()

