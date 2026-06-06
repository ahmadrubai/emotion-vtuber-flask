import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config import MERGED_EMOTION_NAME, MERGED_EMOTION_SOURCES


def merge_folder(
    root: Path,
    sources: list[str],
    target: str,
    mode: str,
):
    target_dir = root / target
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for source in sources:
        src_dir = root / source
        if not src_dir.exists():
            print(f"[SKIP] missing source folder: {src_dir}")
            continue

        for img_path in sorted([p for p in src_dir.iterdir() if p.is_file()]):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue

            dst_name = f"{source}__{img_path.name}"
            dst_path = target_dir / dst_name

            if mode == "move":
                shutil.move(str(img_path), str(dst_path))
            else:
                shutil.copy2(str(img_path), str(dst_path))

            moved += 1

        if mode == "move":
            shutil.rmtree(src_dir)

    print(f"[OK] merged {sources} -> {target_dir} ({moved} files, mode={mode})")


def main():
    parser = argparse.ArgumentParser(description="Merge emotion class folders (e.g. fear+disgust) into one class folder.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path("data_tm") / "rafdb7_train_facecrop_gray_clahe"),
        help="Root folder containing class subfolders.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(MERGED_EMOTION_SOURCES),
        help="Comma-separated source class folder names.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=MERGED_EMOTION_NAME,
        help="Target class folder name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="move",
        choices=["move", "copy"],
        help="Move (default) or copy files into the merged folder.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    sources = [s.strip() for s in str(args.sources).split(",") if s.strip()]
    if len(sources) < 2:
        raise RuntimeError("Need at least 2 source folders.")

    merge_folder(root=root, sources=sources, target=str(args.target), mode=str(args.mode))


if __name__ == "__main__":
    main()
