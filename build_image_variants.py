import argparse
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 1 final image per source image using "
            "grayscale -> blur -> canny edge detection."
        )
    )
    parser.add_argument(
        "--src",
        type=str,
        default=str(Path("data_tm") / "rafdb7" / "train"),
        help="Source dataset folder with class subfolders.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data_tm") / "rafdb7_train_canny"),
        help="Output folder with same class subfolder structure as source.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output folder first if it already exists.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="Gaussian blur kernel size (odd number).",
    )
    parser.add_argument(
        "--canny-th1",
        type=int,
        default=70,
        help="Canny threshold 1.",
    )
    parser.add_argument(
        "--canny-th2",
        type=int,
        default=160,
        help="Canny threshold 2.",
    )
    return parser.parse_args()


def ensure_odd(value: int) -> int:
    if value <= 1:
        return 3
    if value % 2 == 0:
        return value + 1
    return value


def transform_image(image_bgr, blur_kernel: int, canny_th1: int, canny_th2: int):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    canny = cv2.Canny(blur, canny_th1, canny_th2)
    return canny


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def build_dataset(src: Path, out: Path, blur_kernel: int, canny_th1: int, canny_th2: int):
    class_dirs = sorted([p for p in src.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found in: {src}")

    out.mkdir(parents=True, exist_ok=True)

    total_files = 0
    written_files = 0

    for class_dir in class_dirs:
        files = sorted([p for p in class_dir.iterdir() if is_image_file(p)])
        if not files:
            print(f"[WARNING] Empty class or no image files: {class_dir.name}")
            continue

        class_out_dir = out / class_dir.name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        class_written = 0
        for image_path in files:
            total_files += 1
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"[WARNING] Skip unreadable file: {image_path}")
                continue

            canny = transform_image(
                image_bgr=image_bgr,
                blur_kernel=blur_kernel,
                canny_th1=canny_th1,
                canny_th2=canny_th2,
            )

            target_path = class_out_dir / image_path.name

            ok = cv2.imwrite(str(target_path), canny)

            if ok:
                written_files += 1
                class_written += 1
            else:
                print(f"[WARNING] Failed to write output for: {image_path}")

        print(f"[OK] {class_dir.name}: {class_written}/{len(files)} written")

    print("\n[SUMMARY]")
    print(f"Source: {src}")
    print(f"Output: {out}")
    print(f"Total files seen: {total_files}")
    print(f"Files transformed: {written_files}")
    print("Pipeline: grayscale -> blur -> canny")


def main():
    args = parse_args()

    src = Path(args.src)
    out = Path(args.out)
    blur_kernel = ensure_odd(int(args.blur_kernel))

    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")

    if out.exists() and args.overwrite:
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] blur_kernel={blur_kernel}")
    print(f"[INFO] canny_th1={args.canny_th1}, canny_th2={args.canny_th2}")

    build_dataset(
        src=src,
        out=out,
        blur_kernel=blur_kernel,
        canny_th1=int(args.canny_th1),
        canny_th2=int(args.canny_th2),
    )


if __name__ == "__main__":
    main()
