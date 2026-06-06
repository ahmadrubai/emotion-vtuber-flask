import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from PIL import Image, ImageDraw, ImageFont

from config import ASSETS_DIR, DEFAULT_CLASSES, MERGED_EMOTION_NAME


COLORS = {
    "surprise": (168, 85, 247, 255),
    MERGED_EMOTION_NAME: (16, 185, 129, 255),
    "happiness": (234, 179, 8, 255),
    "sadness": (14, 165, 233, 255),
    "anger": (239, 68, 68, 255),
    "neutral": (148, 163, 184, 255),
}


def get_font(size):
    candidates = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
    ]

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass

    return ImageFont.load_default()


def create_placeholder(emotion):
    size = 900
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    color = COLORS.get(emotion, (255, 255, 255, 255))

    cx = size // 2
    cy = size // 2
    radius = 310

    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=color,
        outline=(255, 255, 255, 255),
        width=10,
    )

    eye_y = cy - 95
    eye_radius = 28

    draw.ellipse((cx - 130 - eye_radius, eye_y - eye_radius, cx - 130 + eye_radius, eye_y + eye_radius), fill=(20, 20, 20, 255))
    draw.ellipse((cx + 130 - eye_radius, eye_y - eye_radius, cx + 130 + eye_radius, eye_y + eye_radius), fill=(20, 20, 20, 255))

    mouth_y = cy + 100

    if emotion == "happiness":
        draw.arc((cx - 150, mouth_y - 100, cx + 150, mouth_y + 90), 20, 160, fill=(20, 20, 20, 255), width=16)
    elif emotion == "sadness":
        draw.arc((cx - 150, mouth_y - 10, cx + 150, mouth_y + 170), 200, 340, fill=(20, 20, 20, 255), width=16)
    elif emotion == "anger":
        draw.line((cx - 170, eye_y - 80, cx - 80, eye_y - 50), fill=(20, 20, 20, 255), width=16)
        draw.line((cx + 80, eye_y - 50, cx + 170, eye_y - 80), fill=(20, 20, 20, 255), width=16)
        draw.line((cx - 120, mouth_y + 40, cx + 120, mouth_y + 20), fill=(20, 20, 20, 255), width=16)
    elif emotion == "surprise":
        draw.ellipse((cx - 55, mouth_y - 40, cx + 55, mouth_y + 80), outline=(20, 20, 20, 255), width=16)
    elif emotion == MERGED_EMOTION_NAME:
        draw.arc((cx - 140, mouth_y - 40, cx + 140, mouth_y + 120), 200, 340, fill=(20, 20, 20, 255), width=16)
        draw.rectangle((cx - 90, mouth_y - 10, cx + 90, mouth_y + 60), outline=(20, 20, 20, 255), width=12)
    else:
        draw.line((cx - 130, mouth_y + 30, cx + 130, mouth_y + 30), fill=(20, 20, 20, 255), width=16)

    title_font = get_font(76)
    label = emotion.upper()

    bbox = draw.textbbox((0, 0), label, font=title_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    draw.text(
        (cx - text_w // 2, size - 155),
        label,
        fill=(255, 255, 255, 255),
        font=title_font,
        stroke_width=4,
        stroke_fill=(0, 0, 0, 180),
    )

    return image


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for emotion in DEFAULT_CLASSES:
        image = create_placeholder(emotion)
        output_path = ASSETS_DIR / f"{emotion}.png"
        image.save(output_path)
        print(f"[OK] Created {output_path}")


if __name__ == "__main__":
    main()