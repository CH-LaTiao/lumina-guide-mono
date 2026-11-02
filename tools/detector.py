"""Bounded YOLO11 ONNX inference through OpenCV DNN."""
from pathlib import Path
import time

import cv2
import numpy as np

COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def prepare(image, size=640):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + new_height, left:left + new_width] = cv2.resize(
        image, (new_width, new_height)
    )
    return canvas, (scale, left, top, width, height)


def decode(output, transform, threshold=0.35, nms=0.45):
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] < 5:
        raise ValueError("Unexpected YOLO output contract")
    rows = output[0].T
    if not np.isfinite(rows).all():
        raise ValueError("Non-finite detector output")
    class_scores = rows[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(rows)), classes]
    keep = scores >= threshold
    rows, classes, scores = rows[keep], classes[keep], scores[keep]
    scale, left, top, width, height = transform
    objects = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        selected = rows[indices]
        boxes = np.column_stack((
            selected[:, 0] - selected[:, 2] / 2,
            selected[:, 1] - selected[:, 3] / 2,
            selected[:, 2],
            selected[:, 3],
        )).tolist()
        chosen = cv2.dnn.NMSBoxes(
            boxes, scores[indices].tolist(), threshold, nms
        )
        for local_index in np.asarray(chosen).flatten():
            index = indices[int(local_index)]
            cx, cy, box_w, box_h = rows[index, :4]
            x1 = max(0.0, min(width - 1, float((cx - box_w / 2 - left) / scale)))
            y1 = max(0.0, min(height - 1, float((cy - box_h / 2 - top) / scale)))
            x2 = max(0.0, min(width, float((cx + box_w / 2 - left) / scale)))
            y2 = max(0.0, min(height, float((cy + box_h / 2 - top) / scale)))
            if x2 <= x1 or y2 <= y1:
                continue
            objects.append({
                "class_id": int(class_id),
                "label": COCO80[int(class_id)],
                "score": float(scores[index]),
                "bbox": [x1, y1, x2, y2],
            })
    return sorted(objects, key=lambda item: item["score"], reverse=True)[:30]


class Detector:
    def __init__(self, model_path: str, threshold=0.35):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO model missing: {self.model_path}")
        self.threshold = threshold
        self.session = None
        self.net = None
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.backend = f"ONNX Runtime {ort.__version__} / CPU"
        except ImportError:
            # OpenCV remains a fallback for supported ONNX graph exports.
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.backend = f"OpenCV DNN {cv2.__version__} / CPU"

    def infer(self, image):
        started = time.monotonic()
        canvas, transform = prepare(image)
        if self.session is not None:
            blob = np.ascontiguousarray(
                cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None],
                dtype=np.float32,
            ) / 255.0
            output = self.session.run(None, {self.input_name: blob})[0]
        else:
            blob = cv2.dnn.blobFromImage(
                canvas, 1 / 255.0, (640, 640), swapRB=True, crop=False
            )
            self.net.setInput(blob)
            output = self.net.forward()
        objects = decode(output, transform, self.threshold)
        return objects, round((time.monotonic() - started) * 1000, 1)

