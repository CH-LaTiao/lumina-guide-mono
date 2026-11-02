#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$ROOT/models"
mkdir -p "$MODEL_DIR"

YOLO_SHA="634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b"
MIDAS_SHA="2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58"

download() {
  local name="$1" url="$2" expected="$3"
  local target="$MODEL_DIR/$name"
  if [[ -f "$target" ]] && [[ "$(sha256sum "$target" | cut -d' ' -f1)" == "$expected" ]]; then
    echo "$name: already verified"
    return
  fi
  echo "Downloading $name"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$target.part" "$url"
  echo "$expected  $target.part" | sha256sum -c -
  mv "$target.part" "$target"
}

download "yolo11n.onnx" \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx" \
  "$YOLO_SHA"
download "midas-small.onnx" \
  "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx" \
  "$MIDAS_SHA"

