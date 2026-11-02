#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3)"; fi
cd "$ROOT"
exec "$PYTHON" tools/app.py \
  --yolo "${YOLO_MODEL:-$ROOT/models/yolo11n.onnx}" \
  --depth-model "${DEPTH_MODEL:-$ROOT/models/midas-small.onnx}" \
  "$@"

