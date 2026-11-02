"""Small TensorRT 10 session backed by CUDA Runtime through ctypes."""
import ctypes
from pathlib import Path

import numpy as np

CUDART_CANDIDATES = (
    "libcudart.so.12",
    "libcudart.so.12.6",
    "libcudart.so",
    "/usr/local/cuda/lib64/libcudart.so",
)
H2D = 1
D2H = 2


def _load_cudart():
    errors = []
    for candidate in CUDART_CANDIDATES:
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(str(error))
    raise RuntimeError("CUDA runtime library not found") from OSError("; ".join(errors))


class TensorRTSession:
    """Expose synchronous run() for one float32 input and one float32 output."""

    def __init__(self, engine_path):
        import tensorrt as trt

        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(self.engine_path)
        self.trt = trt
        self.cuda = _load_cudart()
        self.cuda.cudaMalloc.restype = ctypes.c_int
        self.cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cuda.cudaFree.restype = ctypes.c_int
        self.cuda.cudaFree.argtypes = [ctypes.c_void_p]
        self.cuda.cudaMemcpy.restype = ctypes.c_int
        self.cuda.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        self.cuda.cudaStreamCreate.restype = ctypes.c_int
        self.cuda.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cuda.cudaStreamDestroy.restype = ctypes.c_int
        self.cuda.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.cuda.cudaDeviceSynchronize.restype = ctypes.c_int

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [
            name for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [
            name for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise RuntimeError(f"Expected one input and one output, got {names}")
        self.input_name = self.inputs[0]
        self.output_name = self.outputs[0]
        for name in names:
            if self.engine.get_tensor_dtype(name) != trt.DataType.FLOAT:
                raise RuntimeError(f"Only host float32 bindings are supported: {name}")
        self.input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        self.input_bytes = int(np.prod(self.input_shape)) * 4
        self.output_bytes = int(np.prod(self.output_shape)) * 4
        self.device_input = ctypes.c_void_p()
        self.device_output = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        if self.cuda.cudaMalloc(ctypes.byref(self.device_input), self.input_bytes):
            raise RuntimeError("cudaMalloc input failed")
        if self.cuda.cudaMalloc(ctypes.byref(self.device_output), self.output_bytes):
            self.close()
            raise RuntimeError("cudaMalloc output failed")
        if self.cuda.cudaStreamCreate(ctypes.byref(self.stream)):
            self.close()
            raise RuntimeError("cudaStreamCreate failed")
        if not self.context.set_tensor_address(
            self.input_name, int(self.device_input.value)
        ):
            self.close()
            raise RuntimeError("TensorRT set_tensor_address input failed")
        if not self.context.set_tensor_address(
            self.output_name, int(self.device_output.value)
        ):
            self.close()
            raise RuntimeError("TensorRT set_tensor_address output failed")
        self.version = trt.__version__

    def run(self, tensor):
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if tensor.shape != self.input_shape:
            raise ValueError(
                f"Input shape {tensor.shape} does not match engine {self.input_shape}"
            )
        if self.cuda.cudaMemcpy(
            self.device_input, tensor.ctypes.data, self.input_bytes, H2D
        ):
            raise RuntimeError("cudaMemcpy host-to-device failed")
        if not self.context.execute_async_v3(int(self.stream.value)):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        if self.cuda.cudaDeviceSynchronize():
            raise RuntimeError("CUDA device synchronize failed")
        output = np.empty(self.output_shape, dtype=np.float32)
        if self.cuda.cudaMemcpy(
            output.ctypes.data, self.device_output, self.output_bytes, D2H
        ):
            raise RuntimeError("cudaMemcpy device-to-host failed")
        return output

    def close(self):
        for attribute in ("device_input", "device_output"):
            pointer = getattr(self, attribute, None)
            if pointer is not None and pointer.value:
                self.cuda.cudaFree(pointer)
                setattr(self, attribute, ctypes.c_void_p())
        if getattr(self, "stream", None) is not None and self.stream.value:
            self.cuda.cudaStreamDestroy(self.stream)
            self.stream = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
