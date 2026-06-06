import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def remove_bg_bgra(image_bgr: np.ndarray, selfie_segmentation) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(image_rgb)
    mask = results.segmentation_mask
    if mask is None:
        # Fallback: keep everything opaque
        alpha = np.full((h, w), 255, dtype=np.uint8)
    else:
        # Conservative threshold: keep person region
        alpha = (mask > 0.5).astype(np.uint8) * 255

    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch remove background from tm_upload images using MediaPipe selfie segmentation. "
            "Outputs transparent PNGs to a separate folder."
        )
    )
    parser.add_argument(
        "--src",
        type=str,
        default=str(Path("tm_upload") / "rafdb7_train"),
        help="Source folder (contains class subfolders).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("tm_upload_nobg") / "rafdb7_train"),
        help="Output folder (transparent PNGs).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold (0-1). Higher keeps less background.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert foreground to grayscale (keeps alpha/transparency).",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)

    if not src_root.exists():
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    mp_selfie = mp.solutions.selfie_segmentation
    processed = 0
    failed = 0

    with mp_selfie.SelfieSegmentation(model_selection=1) as selfie:
        for class_dir in sorted([p for p in src_root.iterdir() if p.is_dir()]):
            dst_class = out_root / class_dir.name
            dst_class.mkdir(parents=True, exist_ok=True)

            for img_path in sorted([p for p in class_dir.iterdir() if p.is_file()]):
                try:
                    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if image_bgr is None:
                        failed += 1
                        continue

                    # Override threshold if requested
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    results = selfie.process(image_rgb)
                    mask = results.segmentation_mask
                    if mask is None:
                        alpha = np.full(image_bgr.shape[:2], 255, dtype=np.uint8)
                    else:
                        alpha = (mask > float(args.threshold)).astype(np.uint8) * 255

                    if args.grayscale:
                        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                        image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
                    bgra[:, :, 3] = alpha

                    out_path = dst_class / (img_path.stem + ".png")
                    cv2.imwrite(str(out_path), bgra)
                    processed += 1
                except Exception:
                    failed += 1

                if processed and processed % 500 == 0:
                    print(f"[INFO] Processed {processed} images...")

    print(f"[OK] Done. processed={processed}, failed={failed}")
    print(f"[OUT] {out_root}")


if __name__ == "__main__":
    main()

