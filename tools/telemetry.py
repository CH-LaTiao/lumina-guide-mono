"""Jetson GPU, thermal and memory telemetry parsed from tegrastats."""
import re
import subprocess
import threading
import time

PATTERNS = {
    "ram_used_mb": re.compile(r"RAM (\d+)/(\d+)MB"),
    "swap_used_mb": re.compile(r"SWAP (\d+)/(\d+)MB"),
    "gpu_util_percent": re.compile(r"GR3D_FREQ (\d+)%"),
    "gpu_temp_c": re.compile(r"GPU@([\d.]+)C"),
    "cpu_temp_c": re.compile(r"CPU@([\d.]+)C"),
    "soc_temp_c": re.compile(r"SOC([0-9]+)@([\d.]+)C"),
}


def parse_tegrastats(line):
    result = {
        "ram_used_mb": None, "ram_total_mb": None,
        "swap_used_mb": None, "swap_total_mb": None,
        "gpu_util_percent": None, "gpu_temp_c": None,
        "cpu_temp_c": None, "soc_temp_c": None,
    }
    match = PATTERNS["ram_used_mb"].search(line)
    if match:
        result.update(ram_used_mb=int(match.group(1)), ram_total_mb=int(match.group(2)))
    match = PATTERNS["swap_used_mb"].search(line)
    if match:
        result.update(swap_used_mb=int(match.group(1)), swap_total_mb=int(match.group(2)))
    match = PATTERNS["gpu_util_percent"].search(line)
    if match:
        result["gpu_util_percent"] = int(match.group(1))
    for key in ("gpu_temp_c", "cpu_temp_c"):
        match = PATTERNS[key].search(line)
        if match:
            result[key] = float(match.group(1))
    match = PATTERNS["soc_temp_c"].search(line)
    if match:
        result["soc_temp_c"] = float(match.group(2))
    return result


class TelemetryMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = {
            "state": "STARTING",
            "ram_used_mb": None,
            "ram_total_mb": None,
            "swap_used_mb": None,
            "swap_total_mb": None,
            "gpu_util_percent": None,
            "gpu_temp_c": None,
            "cpu_temp_c": None,
            "soc_temp_c": None,
        }
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)

    def read(self):
        with self.lock:
            result = dict(self.latest)
        updated = result.pop("_updated", None)
        result["age_ms"] = (
            None if updated is None
            else round((time.monotonic() - updated) * 1000, 1)
        )
        return result

    def run(self):
        process = None
        try:
            process = subprocess.Popen(
                ["tegrastats", "--interval", "1000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                if self.stop_event.is_set():
                    break
                parsed = parse_tegrastats(line)
                with self.lock:
                    self.latest = dict(parsed, state="READY", _updated=time.monotonic())
        except (OSError, ValueError) as error:
            with self.lock:
                self.latest = {
                    "state": "UNAVAILABLE",
                    "error": f"{type(error).__name__}: {error}",
                    "_updated": time.monotonic(),
                }
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
