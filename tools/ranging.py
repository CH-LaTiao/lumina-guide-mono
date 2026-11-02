"""Monocular distance estimates with explicit method and uncertainty labels."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

# Typical full standing height, not bounding-box size under arbitrary poses.
HEIGHT_PRIORS_M = {
    "person": 1.70,
    "bicycle": 1.05,
    "car": 1.50,
    "motorcycle": 1.10,
    "bus": 3.00,
    "truck": 3.00,
    "traffic light": 0.85,
    "stop sign": 0.75,
    "parking meter": 1.20,
    "fire hydrant": 0.80,
    "bench": 0.85,
    "chair": 0.90,
    "couch": 0.85,
    "potted plant": 0.75,
    "bed": 0.80,
    "dining table": 0.75,
    "toilet": 0.75,
    "suitcase": 0.65,
}
CLIPPED_CLASSES = {"bottle", "wine glass", "cup", "cell phone", "laptop"}


def focal_pixels(width, horizontal_fov_deg):
    if not 10 <= horizontal_fov_deg <= 170:
        raise ValueError("Horizontal FOV must be between 10 and 170 degrees")
    return width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))


def clipped_fraction(bbox, width, height):
    x1, y1, x2, y2 = bbox
    edges = sum((
        x1 <= 2,
        y1 <= 2,
        x2 >= width - 2,
        y2 >= height - 2,
    ))
    return edges / 4


def estimate_object_distance(obj, width, height, horizontal_fov_deg):
    x1, y1, x2, y2 = obj["bbox"]
    bbox_height = max(0.0, y2 - y1)
    previous = HEIGHT_PRIORS_M.get(obj["label"])
    result = {
        "method": "class_prior",
        "approx_m": None,
        "range_m": None,
        "confidence": "unknown",
        "reason": None,
    }
    if previous is None:
        result["reason"] = "no_reliable_size_prior"
        return result
    if obj["label"] in CLIPPED_CLASSES or bbox_height < height * 0.04:
        result["reason"] = "target_too_small_or_unsuitable"
        return result
    if clipped_fraction(obj["bbox"], width, height) >= 0.25:
        result["reason"] = "target_clipped_at_frame_edge"
        return result
    distance = previous * focal_pixels(width, horizontal_fov_deg) / bbox_height
    if not 0.4 <= distance <= 15:
        result["reason"] = "outside_class_prior_range"
        return result
    uncertainty = max(0.25, min(0.55, 0.23 + 0.18 * clipped_fraction(
        obj["bbox"], width, height
    )))
    result.update(
        approx_m=round(distance, 2),
        range_m=[round(distance * (1 - uncertainty), 2),
                 round(distance * (1 + uncertainty), 2)],
        confidence="low" if uncertainty > 0.35 else "medium",
    )
    return result


@dataclass
class DepthCalibration:
    scale_distance_times_disparity: float | None = None
    reference_distance_m: float | None = None
    reference_disparity: float | None = None

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        scale = float(data["scale_distance_times_disparity"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid depth calibration")
        return cls(
            scale_distance_times_disparity=scale,
            reference_distance_m=data.get("reference_distance_m"),
            reference_disparity=data.get("reference_disparity"),
        )

    def distances(self, regions, method="calibrated_depth", uncertainty=0.35):
        result = []
        for item in regions:
            raw = item.get("raw_disparity")
            if self.scale_distance_times_disparity is None or raw is None or raw <= 0:
                result.append({"region": item["region"], "approx_m": None,
                               "range_m": None, "method": "uncalibrated_depth"})
                continue
            distance = self.scale_distance_times_disparity / raw
            result.append({
                "region": item["region"],
                "approx_m": round(distance, 2),
                "range_m": [
                    round(distance * (1 - uncertainty), 2),
                    round(distance * (1 + uncertainty), 2),
                ],
                "method": method,
                "valid": 0.4 <= distance <= 12,
            })
        return result


def object_anchor_scale(objects, raw_depth, source_width, source_height):
    """Infer a per-frame scale from detected objects with valid box distances."""
    values = []
    height, width = raw_depth.shape
    for obj in objects:
        distance = obj.get("distance", {}).get("approx_m")
        if distance is None:
            continue
        x1, y1, x2, y2 = obj["bbox"]
        sx1 = max(0, min(width - 1, round(x1 * width / source_width)))
        sx2 = max(sx1 + 1, min(width, round(x2 * width / source_width)))
        sy1 = max(0, min(height - 1, round(y1 * height / source_height)))
        sy2 = max(sy1 + 1, min(height, round(y2 * height / source_height)))
        cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
        rx, ry = max(2, (sx2 - sx1) // 8), max(2, (sy2 - sy1) // 8)
        patch = raw_depth[
            max(0, cy - ry):min(height, cy + ry + 1),
            max(0, cx - rx):min(width, cx + rx + 1),
        ]
        finite = patch[np.isfinite(patch) & (patch > 1e-6)]
        if finite.size < 4:
            continue
        disparity = float(np.median(finite))
        if disparity > 0:
            values.append(float(distance) * disparity)
    if not values:
        return None, 0
    return float(np.median(values)), len(values)


def calibrate(depth_raw, distance_m, output_path):
    if not math.isfinite(distance_m) or not 0.3 <= distance_m <= 10:
        raise ValueError("Calibration distance must be between 0.3 and 10 m")
    height, width = depth_raw.shape
    patch = depth_raw[height // 2:, width // 3:2 * width // 3]
    finite = patch[np.isfinite(patch)]
    if finite.size < 100:
        raise ValueError("Not enough depth evidence for calibration")
    disparity = float(np.median(finite))
    if not math.isfinite(disparity) or disparity <= 0:
        raise ValueError("Invalid central depth")
    scale = distance_m * disparity
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scale_distance_times_disparity": scale,
        "reference_distance_m": distance_m,
        "reference_disparity": disparity,
        "method": "user_measured_central_reference",
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def annotate_objects(objects, width, height, horizontal_fov_deg):
    annotated = []
    for obj in objects:
        item = dict(obj)
        item["distance"] = estimate_object_distance(
            obj, width, height, horizontal_fov_deg
        )
        annotated.append(item)
    return annotated

