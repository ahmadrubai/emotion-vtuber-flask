import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def _clamp_bbox(x, y, w, h, frame_w, frame_h):
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, int(w))
    h = max(1, int(h))

    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    return x, y, w, h


def _expand_bbox(x, y, w, h, frame_w, frame_h, pad_ratio: float):
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)

    x2 = x + w
    y2 = y + h

    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    return _clamp_bbox(x, y, x2 - x, y2 - y, frame_w, frame_h)


def _apply_clahe_gray(gray_u8: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    return clahe.apply(gray_u8)


def _pick_largest_rect(rects: np.ndarray) -> tuple[int, int, int, int] | None:
    if rects is None or len(rects) == 0:
        return None

    best = None
    best_area = -1

    for (x, y, w, h) in rects:
        area = int(w) * int(h)
        if area > best_area:
            best_area = area
            best = (int(x), int(y), int(w), int(h))

    return best


def process_image(
    image_bgr: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    pad_ratio: float,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    grayscale: bool,
    clahe: bool,
    clahe_clip: float,
    clahe_grid: int,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    gray_for_detect = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_for_detect = cv2.equalizeHist(gray_for_detect)

    rects = face_cascade.detectMultiScale(
        gray_for_detect,
        scaleFactor=float(scale_factor),
        minNeighbors=int(min_neighbors),
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(int(min_size), int(min_size)),
    )

    bbox = _pick_largest_rect(rects)
    if bbox is None:
        cropped = image_bgr
    else:
        x, y, bw, bh = bbox
        x, y, bw, bh = _expand_bbox(x, y, bw, bh, w, h, pad_ratio=pad_ratio)
        cropped = image_bgr[y : y + bh, x : x + bw]

    if not grayscale and not clahe:
        return cropped

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if clahe:
        gray = _apply_clahe_gray(gray, clip_limit=clahe_clip, tile_grid_size=clahe_grid)

    if grayscale:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process exported `data_tm` folders: face crop + grayscale + CLAHE contrast. "
            "Writes to a new output tree by default."
        )
    )
    parser.add_argument(
        "--src",
        type=str,
        default=str(Path("data_tm") / "rafdb7" / "train"),
        help="Source root containing class subfolders.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data_tm") / "rafdb7_train_facecrop_gray_clahe"),
        help="Output root (mirrors class subfolders).",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite files inside --src (dangerous). Prefer --out.",
    )
    parser.add_argument("--pad_ratio", type=float, default=0.18, help="Expand bbox by ratio for context.")
    parser.add_argument(
        "--cascade",
        type=str,
        default=str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"),
        help="OpenCV Haar cascade XML path.",
    )
    parser.add_argument("--scale_factor", type=float, default=1.1)
    parser.add_argument("--min_neighbors", type=int, default=5)
    parser.add_argument("--min_size", type=int, default=40, help="Minimum face size in pixels (w/h).")
    parser.add_argument("--grayscale", action="store_true", default=True, help="Convert to grayscale BGR output.")
    parser.add_argument("--no_grayscale", action="store_true", help="Disable grayscale output.")
    parser.add_argument("--clahe", action="store_true", default=True, help="Apply CLAHE contrast on grayscale.")
    parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE.")
    parser.add_argument("--clahe_clip", type=float, default=2.0)
    parser.add_argument("--clahe_grid", type=int, default=8)
    parser.add_argument(
        "--ext_out",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output extension/format.",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    if not src_root.exists():
        raise FileNotFoundError(f"Source not found: {src_root}")

    grayscale = bool(args.grayscale) and not bool(args.no_grayscale)
    clahe = bool(args.clahe) and not bool(args.no_clahe)

    out_root = src_root if args.in_place else Path(args.out)
    if not args.in_place:
        out_root.mkdir(parents=True, exist_ok=True)

    cascade_path = Path(args.cascade)
    if not cascade_path.exists():
        raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")

    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    processed = 0
    failed = 0

    for class_dir in sorted([p for p in src_root.iterdir() if p.is_dir()]):
        dst_class = out_root / class_dir.name
        if not args.in_place:
            if dst_class.exists():
                shutil.rmtree(dst_class)
            dst_class.mkdir(parents=True, exist_ok=True)

        for img_path in sorted([p for p in class_dir.iterdir() if p.is_file()]):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue

            try:
                image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    failed += 1
                    continue

                out_bgr = process_image(
                    image_bgr=image_bgr,
                    face_cascade=face_cascade,
                    pad_ratio=float(args.pad_ratio),
                    scale_factor=float(args.scale_factor),
                    min_neighbors=int(args.min_neighbors),
                    min_size=int(args.min_size),
                    grayscale=grayscale,
                    clahe=clahe,
                    clahe_clip=float(args.clahe_clip),
                    clahe_grid=int(args.clahe_grid),
                )

                out_name = img_path.stem + ("." + args.ext_out)
                out_path = (class_dir / out_name) if args.in_place else (dst_class / out_name)

                if args.ext_out == "jpg":
                    cv2.imwrite(str(out_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    cv2.imwrite(str(out_path), out_bgr)

                processed += 1
            except Exception:
                failed += 1

            if processed and processed % 500 == 0:
                print(f"[INFO] processed={processed} ...")

    print(f"[OK] done processed={processed} failed={failed}")
    print(f"[OUT] {out_root}")


if __name__ == "__main__":
    main()
