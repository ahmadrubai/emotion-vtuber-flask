import argparse
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Siapkan folder upload Teachable Machine dari hasil export (data_tm/...): "
            "copy subset per class ke folder output, siap drag-drop."
        )
    )
    parser.add_argument(
        "--src",
        type=str,
        default=str(Path("data_tm") / "rafdb7" / "train"),
        help="Sumber folder export, berisi subfolder per class.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("tm_upload") / "rafdb7_train"),
        help="Output folder (akan dibuat/diisi ulang).",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=500,
        help="Jumlah file per class (0 = semua).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed supaya sampling konsisten.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="",
        help="Filter extension (mis: .jpg atau .png). Kosong = semua.",
    )
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        raise FileNotFoundError(f"Folder sumber tidak ketemu: {src}")

    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ext = args.ext.lower().strip()

    class_dirs = sorted([p for p in src.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"Tidak ada subfolder class di: {src}")

    print("[INFO] src:", src)
    print("[INFO] out:", out)
    print("[INFO] classes:", [p.name for p in class_dirs])

    for class_dir in class_dirs:
        files = [p for p in class_dir.iterdir() if p.is_file()]
        if ext:
            files = [p for p in files if p.suffix.lower() == ext]

        if not files:
            print("[WARNING] Empty class:", class_dir.name)
            continue

        files = sorted(files)

        if args.max_per_class and len(files) > args.max_per_class:
            idx = rng.choice(len(files), size=args.max_per_class, replace=False)
            chosen = [files[int(i)] for i in sorted(idx)]
        else:
            chosen = files

        dst_class = out / class_dir.name
        if dst_class.exists():
            shutil.rmtree(dst_class)
        dst_class.mkdir(parents=True, exist_ok=True)

        for src_file in chosen:
            shutil.copy2(src_file, dst_class / src_file.name)

        print(f"[OK] {class_dir.name}: {len(chosen)} files")

    print("\n[NEXT] Upload ke Teachable Machine:")
    print("       - Buka Image Project")
    print("       - Klik tiap class → Upload")
    print(f"       - Drag-drop folder: {out}\\<class> (atau semua file di dalamnya)")


if __name__ == "__main__":
    main()

