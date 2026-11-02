"""Deterministic, concise monocular guidance. This never authorizes movement."""
from dataclasses import dataclass
import time


def _center_overlap(obj, width):
    x1, _, x2, _ = obj["bbox"]
    center_left, center_right = width / 3, 2 * width / 3
    overlap = max(0.0, min(x2, center_right) - max(x1, center_left))
    return overlap / max(1.0, x2 - x1)


def _valid_distance(obj):
    distance = obj.get("distance", {})
    value = distance.get("approx_m")
    return value if distance.get("confidence") in ("low", "medium") else None


def _nearest_object(objects, region, width):
    candidates = []
    for obj in objects:
        x1, _, x2, _ = obj["bbox"]
        if region == "CENTER" and _center_overlap(obj, width) < 0.20:
            continue
        if region == "LEFT" and not (x2 <= 0.55 * width):
            continue
        if region == "RIGHT" and not (x1 >= 0.45 * width):
            continue
        value = _valid_distance(obj)
        if value is not None:
            candidates.append((value, obj))
    return min(candidates, key=lambda item: item[0]) if candidates else None


@dataclass(frozen=True)
class GuidanceResult:
    mode: str
    action: str
    speech: str
    direction: str | None
    priority: int
    movement_allowed: bool = False


def advise(mode, objects, depth_regions, camera_status, width):
    mode = mode.upper()
    if mode == "PAUSED":
        return GuidanceResult(mode, "PAUSED", "已暂停。", None, 0)
    age = camera_status.get("frame_age_ms")
    if camera_status.get("state") != "STREAMING" or age is None or age > 1000:
        return GuidanceResult(mode, "STOP", "停，感知不可用。", None, 100)
    center = _nearest_object(objects, "CENTER", width)
    left = _nearest_object(objects, "LEFT", width)
    right = _nearest_object(objects, "RIGHT", width)
    near = {item["region"]: item.get("near_score") for item in depth_regions}
    center_depth = near.get("CENTER")
    if center and center[0] <= 2.4:
        value, obj = center
        if mode == "WALKING":
            side = None
            if left is None and right is None:
                if (near.get("LEFT") or 1) < (near.get("RIGHT") or 1):
                    side = "左侧"
                elif (near.get("RIGHT") or 1) < (near.get("LEFT") or 1):
                    side = "右侧"
            elif left is None:
                side = "左侧"
            elif right is None:
                side = "右侧"
            suffix = f"，可试{side}" if side else ""
            return GuidanceResult(
                mode, "STOP",
                f"停，前方{obj['label']}约{value:.1f}米{suffix}。",
                side, 95,
            )
        return GuidanceResult(
            mode, "OBSERVE",
            f"前方约{value:.1f}米有{obj['label']}。",
            None, 65,
        )
    side_object = None
    if left and left[0] <= 2.0:
        side_object = ("左侧", left[1]["label"], left[0], "向右")
    if right and right[0] <= 2.0 and (side_object is None or right[0] < side_object[2]):
        side_object = ("右侧", right[1]["label"], right[0], "向左")
    if side_object:
        side, label, value, direction = side_object
        if mode == "WALKING":
            return GuidanceResult(
                mode, "ADJUST", f"{side}{label}约{value:.1f}米，{direction}。",
                direction, 80,
            )
        return GuidanceResult(
            mode, "OBSERVE", f"{side}约{value:.1f}米有{label}。", None, 55
        )
    if center_depth is not None and center_depth >= 0.85:
        if mode == "WALKING":
            left_score = near.get("LEFT")
            right_score = near.get("RIGHT")
            direction = None
            if left_score is not None and right_score is not None:
                direction = "向左" if left_score < right_score else "向右"
            if direction:
                return GuidanceResult(
                    mode, "STOP", f"停，前方近物，{direction}。", direction, 90
                )
            return GuidanceResult(mode, "STOP", "停，前方近物。", None, 95)
        return GuidanceResult(mode, "OBSERVE", "前方检测到较近物体。", None, 60)
    if mode == "WALKING":
        return GuidanceResult(mode, "CAUTION", "前方未发现近物，谨慎通行。", None, 20)
    labels = []
    for obj in objects[:3]:
        if obj["label"] not in labels:
            labels.append(obj["label"])
    description = "、".join(labels) if labels else "暂未识别到明显目标"
    return GuidanceResult(mode, "OBSERVE", f"当前观察到{description}。", None, 10)


def draw_depth_strips(image, regions):
    import cv2
    height, width = image.shape[:2]
    colors = {"LEFT": (255, 170, 40), "CENTER": (80, 230, 110), "RIGHT": (255, 170, 40)}
    for index, item in enumerate(regions):
        score = item.get("near_score")
        if score is None:
            continue
        x1 = round(index * width / 3)
        x2 = round((index + 1) * width / 3)
        thickness = max(3, round(3 + score * 8))
        cv2.rectangle(image, (x1, height - thickness), (x2, height),
                      colors[item["region"]], -1)
    return image

