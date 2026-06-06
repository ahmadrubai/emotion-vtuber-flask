import argparse
import sys
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, Features, Image, Value, load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config import DATASET_NAME


def _ensure_dd(ds):
    if isinstance(ds, DatasetDict):
        return ds
    return DatasetDict({"train": ds})


def _pick_label_col(column_names: list[str]) -> str:
    if "label" in column_names:
        return "label"
    if "labels" in column_names:
        return "labels"
    # Fallback: assume second column is label-ish
    return column_names[1] if len(column_names) > 1 else column_names[0]


def _print_counts(name: str, counts: Counter, total: int, class_names: list[str] | None):
    print(f"\n== {name} ==")
    print(f"total: {total}")

    for class_id in sorted(counts.keys()):
        label = (
            class_names[class_id]
            if class_names is not None and 0 <= class_id < len(class_names)
            else str(class_id)
        )
        n = counts[class_id]
        pct = (n / total * 100.0) if total else 0.0
        print(f"- {class_id:>2} {label:<12} {n:>6}  ({pct:>5.2f}%)")


def _print_split(name, ds, label_col="label"):
    labels = [int(x) for x in ds[label_col]]
    total = len(labels)

    feature = ds.features.get(label_col)
    class_names = None
    if feature is not None and hasattr(feature, "names") and feature.names:
        class_names = list(feature.names)

    counts = Counter(labels)
    _print_counts(name=name, counts=counts, total=total, class_names=class_names)


def _stream_counts(dataset_name: str, split: str, limit: int, label_col: str | None):
    ds = load_dataset(dataset_name, split=split, streaming=True)
    feature_keys = list(ds.features.keys()) if hasattr(ds, "features") else []
    label_col = label_col or _pick_label_col(feature_keys)

    counts: Counter = Counter()
    total = 0

    def _count(iterable):
        nonlocal total
        for example in iterable:
            if limit and total >= limit:
                break

            value = example.get(label_col)
            if value is None:
                continue

            try:
                key = int(value)
            except Exception:
                key = str(value)

            counts[key] += 1
            total += 1

    try:
        _count(ds)
    except ValueError as e:
        # Some zipped datasets declare ClassLabel but store unexpected strings (e.g. "0anger").
        # This causes decoding to fail before iteration. Retry with label forced to string.
        if "Invalid string class label" not in str(e):
            raise

        override = Features({})
        for k, v in ds.features.items():  # type: ignore[attr-defined]
            override[k] = v
        if label_col in override:
            override[label_col] = Value("string")
        if "image" in override and isinstance(override["image"], Image):
            pass

        ds2 = load_dataset(dataset_name, split=split, streaming=True, features=override)
        counts.clear()
        total = 0
        _count(ds2)

    print(f"[INFO] label_col: {label_col}")
    print("[INFO] (streaming) features:", ", ".join(feature_keys))

    print(f"\n== {split} (streaming, limit={limit or 'all'}) ==")
    print(f"total: {total}")
    for key, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        pct = (n / total * 100.0) if total else 0.0
        print(f"- {str(key):<20} {n:>8}  ({pct:>5.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Print label distribution for a HF dataset.")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME, help="HF dataset name.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming=True (no full download). Good for huge datasets.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name for streaming mode (e.g. train/validation/test).",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="",
        help="Optional: force label column name (streaming mode).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
        help="Max rows to count in streaming mode (0 = all, may be slow).",
    )
    args = parser.parse_args()

    print("[INFO] Dataset:", args.dataset)

    if args.streaming:
        _stream_counts(
            dataset_name=args.dataset,
            split=args.split,
            limit=int(args.limit),
            label_col=args.label_col.strip() or None,
        )
        return

    dataset = _ensure_dd(load_dataset(args.dataset))

    base = "train" if "train" in dataset else list(dataset.keys())[0]
    cols = dataset[base].column_names
    label_col = _pick_label_col(cols)

    if "train" in dataset and "validation" in dataset:
        _print_split("train", dataset["train"], label_col=label_col)
        _print_split("validation", dataset["validation"], label_col=label_col)
    elif "train" in dataset and "test" in dataset:
        _print_split("train", dataset["train"], label_col=label_col)
        _print_split("test", dataset["test"], label_col=label_col)
    else:
        for split_name, split_ds in dataset.items():
            _print_split(split_name, split_ds, label_col=label_col)


if __name__ == "__main__":
    main()

