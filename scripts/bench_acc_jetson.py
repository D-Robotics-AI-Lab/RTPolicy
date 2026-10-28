#!/usr/bin/env python3
"""Host-to-host TensorRT timing and engine output comparisons using PyTorch CUDA.

Usage:
  python scripts/bench_acc_jetson.py bench ENGINE INPUTS_DIR [RUNS]
  python scripts/bench_acc_jetson.py accuracy REF_ENGINE TEST_ENGINE SAMPLES_DIR [SAMPLES]

Output error is a numerical check, not a robot task success evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import tensorrt as trt
import torch


def load_inputs(inputs_dir):
    return {p.stem: np.load(p, allow_pickle=False) for p in sorted(Path(inputs_dir).glob("*.npy"))}


class Runner:
    def __init__(self, engine_path):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-enabled PyTorch installation and available GPU are required")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create TensorRT execution context")
        self.stream = torch.cuda.Stream()
        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            target = self.input_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.output_names
            target.append(name)

    def close(self):
        if self.stream is not None:
            self.stream.synchronize()
        self.context = self.engine = self.runtime = self.stream = None

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
        device = {}
        # Upload, allocate, and execute on the same stream: no default-stream race.
        with torch.no_grad(), torch.cuda.stream(self.stream):
            for name in self.input_names:
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                array = np.asarray(sample_inputs[name], dtype=dtype, order="C")
                shape = tuple(self.engine.get_tensor_shape(name))
                if any(d < 0 for d in shape):
                    if not self.context.set_input_shape(name, array.shape):
                        raise ValueError(f"Invalid input shape for {name}: {array.shape}")
                elif array.shape != shape:
                    raise ValueError(f"Input {name}: expected shape {shape}, got {array.shape}")
                device[name] = torch.as_tensor(array, device="cuda")
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(d < 0 for d in shape):
                    raise ValueError(f"Unresolved output shape for {name}: {shape}")
                dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
                device[name] = torch.empty(shape, dtype=dtype, device="cuda")
            for name, tensor in device.items():
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"Cannot bind engine tensor: {name}")
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return {name: device[name].cpu().numpy() for name in self.output_names}


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


def accuracy(ref_engine, test_engine, calib_dir, num_samples=10):
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    sample_dirs = sorted(p / "inputs" for p in Path(calib_dir).glob("sample_*") if (p / "inputs").is_dir())[:num_samples]
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_*/inputs directories under {calib_dir}")
    metrics = {}
    with Runner(ref_engine) as reference, Runner(test_engine) as tested:
        if set(reference.output_names) != set(tested.output_names):
            raise ValueError("Reference and test engines have different output names")
        for directory in sample_dirs:
            inputs = load_inputs(directory)
            ref, result = reference.run(inputs), tested.run(inputs)
            for name in reference.output_names:
                a, b = ref[name].astype(np.float64), result[name].astype(np.float64)
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"Output shape mismatch or nonfinite values: {name}")
                difference = a - b
                values = metrics.setdefault(name, {"count": 0, "abs_sum": 0.0, "sq_sum": 0.0, "ref_sq_sum": 0.0, "max_abs_err": 0.0})
                values["count"] += a.size
                values["abs_sum"] += float(np.abs(difference).sum())
                values["sq_sum"] += float(np.square(difference).sum())
                values["ref_sq_sum"] += float(np.square(a).sum())
                values["max_abs_err"] = max(values["max_abs_err"], float(np.abs(difference).max(initial=0)))
    result = {"samples": len(sample_dirs), "scope": "engine_output_numerics_only", "outputs": {}}
    for name, values in metrics.items():
        result["outputs"][name] = {
            "max_abs_err": values["max_abs_err"],
            "mean_abs_err": values["abs_sum"] / values["count"],
            "rmse": float(np.sqrt(values["sq_sum"] / values["count"])),
            "relative_l2": float(np.sqrt(values["sq_sum"] / max(values["ref_sq_sum"], 1e-30))),
        }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("bench")
    benchmark.add_argument("engine")
    benchmark.add_argument("inputs_dir")
    benchmark.add_argument("num_runs", nargs="?", type=int, default=100)
    benchmark.add_argument("--warmup", type=int, default=10)
    comparison = commands.add_parser("accuracy")
    comparison.add_argument("reference")
    comparison.add_argument("test")
    comparison.add_argument("calib_dir")
    comparison.add_argument("num_samples", nargs="?", type=int, default=10)
    args = parser.parse_args()
    if args.command == "bench":
        bench(args.engine, args.inputs_dir, args.num_runs, args.warmup)
    else:
        accuracy(args.reference, args.test, args.calib_dir, args.num_samples)


if __name__ == "__main__":
    main()
