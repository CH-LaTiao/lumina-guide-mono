"""MiDaS Small relative monocular depth with explicit calibration boundaries."""
from pathlib import Path
import time

import cv2
import numpy as np

REGIONS = ("LEFT", "CENTER", "RIGHT")


def blob_from_image(image, size=256):
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array((0.485, 0.456, 0.406), np.float32)
    std = np.array((0.229, 0.224, 0.225), np.float32)
    normalized = (rgb - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def normalize_inverse_depth(output):
    raw = np.asarray(output, np.float32).squeeze()
    if raw.ndim != 2 or raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("Unexpected MiDaS output contract")
    low, high = np.percentile(raw, (5, 95))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-6:
        raise ValueError("Depth output has no usable range")
    normalized = np.clip((raw - low) / (high - low), 0, 1)
    return raw, normalized.astype(np.float32)


def region_statistics(raw):
    height, width = raw.shape
    # Walls and distant sky should not dominate a lower-frame obstacle cue.
    top = height // 3
    values = []
    regions = []
    for index, name in enumerate(REGIONS):
        x1 = round(index * width / 3)
        x2 = round((index + 1) * width / 3)
        patch = raw[top:, x1:x2]
        finite = patch[np.isfinite(patch)]
        value = float(np.median(finite)) if finite.size else None
        if value is not None:
            values.append(value)
        regions.append({"region": name, "raw_disparity": value})
    if values:
        low = min(values)
        high = max(values)
        span = max(high - low, 1e-6)
        for item in regions:
            value = item["raw_disparity"]
            item["near_score"] = (
                None if value is None else round(float((value - low) / span), 3)
            )
    return regions


def colorize(raw):
    low, high = np.percentile(raw, (5, 95))
    norm = np.clip((raw - low) / max(high - low, 1e-6), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


class DepthEstimator:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Depth model missing: {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self.backend = "MiDaS Small / OpenCV DNN CPU"

    def infer(self, image):
        started = time.monotonic()
        self.net.setInput(blob_from_image(image))
        raw, normalized = normalize_inverse_depth(self.net.forward())
        regions = region_statistics(raw)
        return {
            "state": "READY",
            "model": "MiDaS Small v2.1",
            "backend": self.backend,
            "raw": raw,
            "normalized": normalized,
            "regions": regions,
            "inference_ms": round((time.monotonic() - started) * 1000, 1),
        }

