"""Single-camera perception, ranging, guidance and LAN dashboard."""
from argparse import ArgumentParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from camera import CameraCapture, CameraConfig
from depth_estimator import DepthEstimator, colorize
from detector import Detector
from guidance import advise, draw_depth_strips
from ranging import DepthCalibration, annotate_objects, calibrate
from text_render import render_overlays

ROOT = Path(__file__).resolve().parents[1]


class AppState:
    def __init__(self, calibration_path):
        self.lock = threading.RLock()
        self.camera = None
        self.camera_source = None
        self.raw = None
        self.annotated_jpeg = None
        self.depth_jpeg = None
        self.objects = []
        self.depth = {"state": "WAITING", "regions": []}
        self.guidance = None
        self.mode = "WALKING"
        self.inference = {
            "state": "STARTING",
            "cycles": 0,
            "last_cycle": None,
            "detection_ms": None,
            "depth_ms": None,
            "annotated_age_ms": None,
        }
        self.calibration_path = calibration_path
        self.calibration = DepthCalibration.load(calibration_path)

    def set_mode(self, mode):
        if mode not in ("standing", "walking", "paused"):
            raise ValueError("无效模式")
        with self.lock:
            self.mode = mode.upper()
        return self.mode

    def calibrate(self, distance):
        with self.lock:
            raw = None if self.depth.get("state") != "READY" else self.depth.get("raw")
            if raw is None:
                raise ValueError("深度图尚未就绪")
            payload = calibrate(raw, float(distance), self.calibration_path)
            self.calibration = DepthCalibration.load(self.calibration_path)
        return payload

    def status(self):
        with self.lock:
            camera = (
                self.camera_source.read_status()
                if self.camera_source is not None
                else dict(self.camera) if self.camera
                else {"state": "STARTING"}
            )
            inference = dict(self.inference)
            objects = [{key: value for key, value in obj.items() if key != "crop"}
                       for obj in self.objects]
            regions = [
                {key: value for key, value in region.items()}
                for region in self.depth.get("regions", [])
            ]
            guidance = self.guidance
            annotated = self.annotated_jpeg
            annotated_at = inference.get("last_cycle")
        if annotated is not None and annotated_at is not None:
            inference["annotated_age_ms"] = round(
                (time.monotonic() - annotated_at) * 1000, 1
            )
        return {
            "version": "0.1.0",
            "mode": self.mode,
            "camera": camera,
            "inference": inference,
            "objects": objects,
            "depth": {
                "state": self.depth.get("state", "WAITING"),
                "model": self.depth.get("model"),
                "backend": self.depth.get("backend"),
                "regions": regions,
                "calibrated": self.calibration.scale_distance_times_disparity is not None,
            },
            "guidance": guidance,
            "real_output": False,
            "movement_allowed": False,
            "limitations": [
                "单目近似测距，不是安全级距离传感器",
                "未验证真实佩戴、全部光照和复杂障碍",
                "不能替代白手杖、导盲犬、陪同人员或交通判断",
            ],
        }


class InferenceRunner:
    def __init__(self, state, camera, detector, depth_estimator, fov):
        self.state = state
        self.camera = camera
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.fov = fov
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=4)

    def _draw(self, frame, objects, depth, guidance):
        image = frame.copy()
        height, width = image.shape[:2]
        labels = []
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            distance = obj["distance"]
            color = (60, 220, 105) if distance["approx_m"] is not None else (50, 190, 240)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            value = distance["approx_m"]
            label = (
                f"{obj['label']} {obj['score']:.2f} ~{value:.1f}m"
                if value is not None
                else f"{obj['label']} {obj['score']:.2f} 距离未知"
            )
            text_y = max(24, y1 - 8)
            labels.append((label, (max(4, x1 + 2), text_y - 18), 18, color))
        image = draw_depth_strips(image, depth.get("regions", []))
        banner = guidance["speech"]
        return render_overlays(image, labels, banner[:72])

    def run(self):
        last_sequence = -1
        while not self.stop_event.is_set():
            snapshot = self.camera.snapshot()
            if snapshot is None or snapshot["sequence"] == last_sequence:
                self.stop_event.wait(0.03)
                continue
            last_sequence = snapshot["sequence"]
            frame = snapshot["frame"]
            captured = snapshot["captured"]
            height, width = frame.shape[:2]
            try:
                objects, detection_ms = self.detector.infer(frame)
                objects = annotate_objects(objects, width, height, self.fov)
                depth = self.depth_estimator.infer(frame)
                depth_ranges = self.state.calibration.distances(depth["regions"])
                ranges_by_region = {
                    item["region"]: item for item in depth_ranges
                }
                for region in depth["regions"]:
                    region.update(ranges_by_region.get(region["region"], {}))
                camera_status = self.camera.read_status()
                result = advise(
                    self.state.mode, objects, depth["regions"], camera_status, width
                )
                guided = {
                    "mode": result.mode,
                    "action": result.action,
                    "speech": result.speech,
                    "direction": result.direction,
                    "priority": result.priority,
                    "movement_allowed": False,
                }
                image = self._draw(frame, objects, depth, guided)
                ok, jpeg = cv2.imencode(
                    ".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 82)
                )
                if not ok:
                    raise RuntimeError("预览编码失败")
                depth_jpeg = cv2.imencode(
                    ".jpg", colorize(depth["raw"]),
                    (cv2.IMWRITE_JPEG_QUALITY, 78)
                )[1].tobytes()
                with self.state.lock:
                    self.state.objects = objects
                    self.state.depth = {
                        key: value for key, value in depth.items() if key not in ("raw", "normalized")
                    }
                    self.state.depth["raw"] = depth["raw"]
                    self.state.guidance = guided
                    self.state.annotated_jpeg = jpeg.tobytes()
                    self.state.depth_jpeg = depth_jpeg
                    self.state.inference.update(
                        state="READY",
                        cycles=self.state.inference["cycles"] + 1,
                        last_cycle=time.monotonic(),
                        detection_ms=detection_ms,
                        detector_backend=self.detector.backend,
                        depth_ms=depth["inference_ms"],
                        captured=round(captured, 3),
                        error=None,
                    )
            except Exception as error:
                with self.state.lock:
                    self.state.inference.update(
                        state="FAULT",
                        error=f"{type(error).__name__}: {str(error)[:160]}",
                    )
                self.stop_event.wait(0.25)


class Handler(BaseHTTPRequestHandler):
    server_version = "LuminaMono/0.1"

    def log_message(self, format_string, *args):
        sys.stdout.write(
            "[HTTP] " + (format_string % args) + "\n"
        )
        sys.stdout.flush()

    def _cors_origin_valid(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = self.headers.get("Host", "")
        return urlsplit(origin).netloc == host

    def _send(self, code, content_type, payload, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "private, max-age=1" if cache else "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, payload, code=200):
        self._send(
            code, "application/json; charset=utf-8",
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8",
                       (ROOT / "tools/index.html").read_bytes())
        elif path == "/static/app.js":
            self._send(200, "text/javascript; charset=utf-8",
                       (ROOT / "tools/static/app.js").read_bytes())
        elif path == "/status":
            self._json(self.server.app_state.status())
        elif path in ("/frame.jpg", "/stream.mjpg"):
            with self.server.app_state.lock:
                payload = self.server.app_state.annotated_jpeg
            if payload is None:
                self._json({"error": "stream-warming"}, 503)
            elif path == "/frame.jpg":
                self._send(200, "image/jpeg", payload, cache=True)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while True:
                    with self.server.app_state.lock:
                        payload = self.server.app_state.annotated_jpeg
                    try:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                            str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n"
                        )
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    time.sleep(0.08)
        elif path == "/depth.jpg":
            with self.server.app_state.lock:
                payload = self.server.app_state.depth_jpeg
            if payload is None:
                self._json({"error": "depth-warming"}, 503)
            else:
                self._send(200, "image/jpeg", payload, cache=True)
        else:
            self._json({"error": "not-found"}, 404)

    def do_POST(self):
        if not self._cors_origin_valid():
            self._json({"error": "cross-origin-request-blocked"}, 403)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 4096:
            self._json({"error": "body-too-large"}, 413)
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if urlsplit(self.path).path == "/api/mode":
                self._json({"mode": self.server.app_state.set_mode(payload.get("mode"))})
            elif urlsplit(self.path).path == "/api/calibrate":
                self._json({
                    "calibration": self.server.app_state.calibrate(
                        payload.get("distance_m")
                    )
                })
            else:
                self._json({"error": "not-found"}, 404)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)


def camera_index(value):
    return int(value)


def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--camera", type=camera_index,
                        default=int(os.environ.get("CAMERA_DEVICE", "0")))
    parser.add_argument("--width", type=int,
                        default=int(os.environ.get("CAMERA_WIDTH", "1280")))
    parser.add_argument("--height", type=int,
                        default=int(os.environ.get("CAMERA_HEIGHT", "720")))
    parser.add_argument("--fps", type=int,
                        default=int(os.environ.get("CAMERA_FPS", "30")))
    parser.add_argument("--fov", type=float,
                        default=float(os.environ.get("CAMERA_HFOV_DEG", "68")))
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--depth-model", required=True)
    parser.add_argument("--runtime", type=Path,
                        default=ROOT / "runtime")
    args = parser.parse_args(argv)
    args.runtime.mkdir(parents=True, exist_ok=True)
    calibration_path = args.runtime / "depth_calibration.json"
    state = AppState(calibration_path)
    camera = CameraCapture(CameraConfig(args.camera, args.width, args.height, args.fps))
    state.camera_source = camera
    detector = Detector(args.yolo)
    depth_estimator = DepthEstimator(args.depth_model)
    inference = InferenceRunner(state, camera, detector, depth_estimator, args.fov)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.app_state = state

    def shutdown(*_):
        # HTTPServer.shutdown must run outside serve_forever's own thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    camera.start()
    inference.start()
    print(
        f"LUMINA Mono ready: http://{args.host}:{args.port}/",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        inference.stop()
        camera.stop()
        server.server_close()


if __name__ == "__main__":
    main()
