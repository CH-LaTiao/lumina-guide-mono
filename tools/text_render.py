"""Unicode text overlays for OpenCV frames, with an ASCII-safe fallback."""
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

FONT_CANDIDATES = (
    os.environ.get("LUMINA_FONT"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font_path():
    return next((str(Path(path)) for path in FONT_CANDIDATES
                 if path and Path(path).is_file()), None)


def render_overlays(image, labels, banner):
    """Draw box labels and a banner; all coordinates use the source frame."""
    if Image is None or _font_path() is None:
        for text, (x, y), size, color in labels:
            safe = text.encode("ascii", "replace").decode("ascii")
            cv2.putText(image, safe, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        max(.4, size / 32), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, safe, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        max(.4, size / 32), color, 1, cv2.LINE_AA)
        safe_banner = banner.encode("ascii", "replace").decode("ascii")
        cv2.putText(image, safe_banner[:72], (16, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, .76, (245, 245, 245), 2, cv2.LINE_AA)
        return image
    path = _font_path()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    for text, (x, y), size, color in labels:
        font = ImageFont.truetype(path, int(size))
        box = draw.textbbox((x, y), text, font=font)
        draw.rounded_rectangle(
            (box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3),
            radius=3, fill=(18, 23, 31),
        )
        draw.text((x, y), text, font=font, fill=tuple(reversed(color)))
    font = ImageFont.truetype(path, 24)
    draw.rectangle((0, 0, image.shape[1], 46), fill=(12, 18, 26))
    draw.text((16, 9), banner, font=font, fill=(242, 247, 250))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
