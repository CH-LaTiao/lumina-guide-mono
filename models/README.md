# Models

Models and engines are never committed. Prepare these files locally:

- `yolo11n_640_fp16.engine`: YOLO11n COCO detection, TensorRT/CUDA.
- `dav2s_518_fp16.engine`: Depth Anything V2 Small, TensorRT/CUDA.
- `yolo11n.onnx`: portable CPU detection fallback.
- `midas-small.onnx`: portable CPU depth fallback.

Run `../scripts/download_models.sh` for the ONNX fallbacks. TensorRT engines must
be built on a compatible Jetson GPU and TensorRT version. Review each upstream
model licence before redistribution or commercial deployment.

