"""Latest-frame UVC camera capture with bounded reconnect behavior."""
from dataclasses import dataclass
import threading
import time

import cv2


@dataclass(frozen=True)
class CameraConfig:
    device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


class CameraCapture:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = None
        self.sequence = 0
        self.state = {
            "state": "STARTING",
            "device": config.device,
            "width": config.width,
            "height": config.height,
            "reconnects": 0,
            "read_failures": 0,
            "fps": 0.0,
        }
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def snapshot(self):
        with self.lock:
            item = self.latest
        if item is None:
            return None
        return dict(item)

    def read_status(self):
        with self.lock:
            status = dict(self.state)
            item = self.latest
        if item is None:
            status["frame_age_ms"] = None
        else:
            status["frame_age_ms"] = round((time.monotonic() - item["captured"]) * 1000, 1)
        return status

    def _open(self):
        cap = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        cap = None
        failures = 0
        stamps = []
        while not self.stop_event.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    with self.lock:
                        self.state.update(state="RECONNECTING")
                        self.state["reconnects"] += 1
                    self.stop_event.wait(1.0)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                with self.lock:
                    self.state.update(state="RECONNECTING", read_failures=failures)
                cap.release()
                cap = None
                self.stop_event.wait(min(2.0, 0.15 * failures))
                continue
            now = time.monotonic()
            stamps.append(now)
            stamps = [stamp for stamp in stamps if now - stamp <= 2.0]
            with self.lock:
                self.sequence += 1
                self.latest = {
                    "frame": frame,
                    "sequence": self.sequence,
                    "captured": now,
                }
                self.state.update(
                    state="STREAMING",
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    fps=round((len(stamps) - 1) / max(0.01, stamps[-1] - stamps[0]), 1)
                    if len(stamps) > 1
                    else 0.0,
                )
        if cap is not None:
            cap.release()

