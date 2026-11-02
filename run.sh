#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi
cd "$ROOT"
exec "$PYTHON" tools/app.py \
  --yolo "${YOLO_MODEL:-$ROOT/models/yolo11n.onnx}" \
  --depth-model "${DEPTH_MODEL:-$ROOT/models/midas-small.onnx}" \
  --detector-engine "${LUMINA_DET_ENGINE:-$ROOT/models/yolo11n_640_fp16.engine}" \
  --depth-engine "${LUMINA_DEPTH_ENGINE:-$ROOT/models/dav2s_518_fp16.engine}" \
  "$@"

