#!/usr/bin/env python3
"""Build and validate FP32/FP16 ONNX graphs with TensorRT 10/11.

Precision follows graph tensor types. INT8 and quantized Q/DQ graphs are not
supported. Benchmarks include allocations and host/device transfers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def cu(result):
    values = result if isinstance(result, tuple) else (result,)
    if values[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {values[0]}")
    return values[1] if len(values) == 2 else values[1:]


def check_graph_precision(onnx_path):
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=False)
    graphs = [model.graph]
    while graphs:
        graph = graphs.pop()
        int8_values = any(value.type.tensor_type.elem_type == onnx.TensorProto.INT8 for value in (*graph.input, *graph.output, *graph.value_info))
        if int8_values or any(t.data_type == onnx.TensorProto.INT8 for t in graph.initializer):
            raise ValueError("INT8 is not supported; supply an FP32 or FP16 ONNX graph")
        for node in graph.node:
            if node.op_type in {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"} or node.op_type.startswith("QLinear"):
                raise ValueError("Quantized Q/DQ graphs are not supported; use FP16 for TensorRT")
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    graphs.append(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    graphs.extend(attribute.graphs)


def build_engine(onnx_path, engine_path, strongly_typed=True, workspace_gib=1.0):
    check_graph_precision(onnx_path)
    destination = Path(engine_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite engine: {destination}")
    if workspace_gib <= 0:
        raise ValueError("workspace_gib must be positive")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    flags = (1 << int(flag)) if strongly_typed and flag is not None else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {onnx_path}\n{errors}")
    dynamic = [network.get_input(i).name for i in range(network.num_inputs)
               if any(d < 0 for d in network.get_input(i).shape)]
    if dynamic:
        raise ValueError(f"Static input shapes are required by this builder; re-export with fixed shapes: {dynamic}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))
    # FP32 reference graphs must not silently use reduced-precision TF32 math.
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    print(f"Building TensorRT {trt.__version__} engine; precision follows ONNX tensor types: {onnx_path}", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        handle.write(memoryview(serialized))
    print(f"ENGINE_SAVED {destination} {destination.stat().st_size}", flush=True)


class EngineReader(trt.IStreamReaderV2 if hasattr(trt, "IStreamReaderV2") else object):
    """Read serialized engines without loading a multi-GB Python bytes object."""
    def __init__(self, path):
        super().__init__()
        self.handle = Path(path).open("rb")

    def read(self, size, cuda_stream_pointer):
        return self.handle.read(size)

    def seek(self, offset, where):
        positions = {trt.SeekPosition.SET: os.SEEK_SET,
                     trt.SeekPosition.CUR: os.SEEK_CUR, trt.SeekPosition.END: os.SEEK_END}
        self.handle.seek(offset, positions[where])


def deserialize_engine(runtime, path):
    if hasattr(trt, "IStreamReaderV2"):
        reader = EngineReader(path)
        try:
            return runtime.deserialize_cuda_engine(reader)
        finally:
            reader.handle.close()
    return runtime.deserialize_cuda_engine(Path(path).read_bytes())


class Runner:
    def __init__(self, engine_path):
        self.stream = None
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = deserialize_engine(self.runtime, engine_path)
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create TensorRT execution context")
        self.stream = cu(cudart.cudaStreamCreate())
        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            target = self.input_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.output_names
            target.append(name)

    def close(self):
        if self.stream is not None:
            cu(cudart.cudaStreamDestroy(self.stream))
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def run(self, sample_inputs, sync=True):
        if self.stream is None:
            raise RuntimeError("Runner is closed")
        if not sync:
            raise ValueError("NumPy output requires synchronization; sync=False is not supported")
        missing = set(self.input_names) - set(sample_inputs)
        if missing:
            raise ValueError(f"Missing engine inputs: {sorted(missing)}")
        device, host_inputs = {}, {}
        try:
            for name in self.input_names:
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                arr = np.asarray(sample_inputs[name], dtype=dtype, order="C")
                expected = tuple(self.engine.get_tensor_shape(name))
                if any(d < 0 for d in expected):
                    if not self.context.set_input_shape(name, arr.shape):
                        raise ValueError(f"Invalid input shape for {name}: {arr.shape}")
                elif arr.shape != expected:
                    raise ValueError(f"Input {name}: expected shape {expected}, got {arr.shape}")
                host_inputs[name] = arr
                ptr = cu(cudart.cudaMalloc(max(1, arr.nbytes)))
                device[name] = ptr
                cu(cudart.cudaMemcpy(ptr, arr, arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            outputs = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(d < 0 for d in shape):
                    raise ValueError(f"Unresolved output shape for {name}: {shape}")
                outputs[name] = np.empty(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
                device[name] = cu(cudart.cudaMalloc(max(1, outputs[name].nbytes)))
            for name, ptr in device.items():
                if not self.context.set_tensor_address(name, ptr):
                    raise RuntimeError(f"Cannot bind engine tensor: {name}")
            if not self.context.execute_async_v3(self.stream):
                raise RuntimeError("TensorRT execution failed")
            cu(cudart.cudaStreamSynchronize(self.stream))
            for name, arr in outputs.items():
                cu(cudart.cudaMemcpy(arr, device[name], arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            return outputs
        finally:
            for ptr in device.values():
                cu(cudart.cudaFree(ptr))


def load_inputs(inputs_dir):
    return {p.stem: np.load(p, allow_pickle=False) for p in sorted(Path(inputs_dir).glob("*.npy"))}


def bench(engine_path, inputs_dir, num_runs=100, warmup=10):
    if num_runs < 1 or warmup < 0:
        raise ValueError("num_runs must be positive and warmup nonnegative")
    inputs = load_inputs(inputs_dir)
    with Runner(engine_path) as runner:
        for _ in range(warmup):
            runner.run(inputs)
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            runner.run(inputs)
            times.append((time.perf_counter() - start) * 1000)
    result = {"engine": str(engine_path), "runs": num_runs,
              "scope": "host_to_host_including_allocation", "mean_ms": float(np.mean(times)),
              "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
              "min_ms": float(np.min(times)), "max_ms": float(np.max(times))}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("onnx")
    build.add_argument("engine")
    build.add_argument("typing", nargs="?", choices=["strongly_typed"])
    build.add_argument("--workspace-gib", type=float, default=1.0)
    benchmark = commands.add_parser("bench")
    benchmark.add_argument("engine")
    benchmark.add_argument("inputs_dir")
    benchmark.add_argument("num_runs", nargs="?", type=int, default=100)
    benchmark.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.command == "build":
        build_engine(args.onnx, args.engine, True, args.workspace_gib)
    else:
        bench(args.engine, args.inputs_dir, args.num_runs, args.warmup)


if __name__ == "__main__":
    main()
