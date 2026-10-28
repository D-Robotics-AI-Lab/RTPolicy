#!/usr/bin/env python3
"""Convert an FP32 ONNX graph to mixed FP16 while keeping its public I/O types.

Requires onnx and onnxconverter-common. Normalization/reduction/softmax operators
stay FP32. External weights are copied in bounded chunks to a new output sidecar;
the source model is never rewritten. Work files default to the output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from types import FunctionType

import numpy as np
import onnx
from onnx import external_data_helper as edh
from onnxconverter_common import float16


FP32_OPERATORS = ["LayerNormalization", "Softmax", "ReduceMean", "ReduceSum", "Pow"]


def graphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from graphs(child)


def tensors(graph, prefix=()):
    if graph.sparse_initializer:
        raise ValueError("Sparse initializers are not supported by this converter")
    for tensor in graph.initializer:
        yield prefix + ("initializer", tensor.name), tensor
    for node in graph.node:
        key = node.name or (node.output[0] if node.output else node.op_type)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield prefix + (key, attribute.name), attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                for i, tensor in enumerate(attribute.tensors):
                    yield prefix + (key, attribute.name, str(i)), tensor
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from tensors(attribute.g, prefix + (key, attribute.name))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for i, subgraph in enumerate(attribute.graphs):
                    yield from tensors(subgraph, prefix + (key, attribute.name, str(i)))


def copy_external_tensor(tensor, original, source_dir, stream, location):
    metadata, source_dtype, dimensions = original
    source = (source_dir / metadata["location"]).resolve()
    expected = int(np.prod(dimensions, dtype=np.int64)) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(source_dtype)).itemsize
    length = int(metadata.get("length", expected))
    if length != expected:
        raise ValueError(f"External tensor byte count mismatch: {tensor.name}")
    offset = int(metadata.get("offset", 0))
    if source.stat().st_size < offset + length:
        raise ValueError(f"Truncated external tensor: {tensor.name}")
    convert = source_dtype == onnx.TensorProto.FLOAT and tensor.data_type == onnx.TensorProto.FLOAT16
    if tensor.data_type != source_dtype and not convert:
        raise ValueError(f"Unsupported external dtype conversion for {tensor.name}")
    padding = (-stream.tell()) % 16
    stream.write(b"\0" * padding)
    start = stream.tell()
    with source.open("rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining:
            data = handle.read(min(8 << 20, remaining))
            if not data:
                raise ValueError(f"Truncated external tensor: {tensor.name}")
            if convert:
                values = np.frombuffer(data, dtype=np.float32)
                data = np.clip(values, -65504.0, 65504.0).astype(np.float16).tobytes()
                remaining -= len(values) * 4
            else:
                remaining -= len(data)
            stream.write(data)
    tensor.ClearField("raw_data")
    del tensor.external_data[:]
    for key, value in {"location": location, "offset": start, "length": stream.tell() - start}.items():
        entry = tensor.external_data.add()
        entry.key, entry.value = key, str(value)
    tensor.data_location = onnx.TensorProto.EXTERNAL


def convert(onnx_in, onnx_out, work_dir=None):
    source, destination = Path(onnx_in).resolve(), Path(onnx_out).resolve()
    data_destination = destination.with_name(destination.name + ".data")
    if source == destination:
        raise ValueError("Input and output models must be different")
    if destination.exists() or data_destination.exists():
        raise FileExistsError(f"Choose a fresh output model and sidecar: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work_parent = Path(work_dir).resolve() if work_dir else destination.parent
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rtpolicy-fp16-", dir=work_parent, ignore_cleanup_errors=True) as temporary:
        work = Path(temporary)
        inferred = work / "inferred.onnx"
        onnx.shape_inference.infer_shapes_path(str(source), str(inferred))
        model = onnx.load(inferred, load_external_data=False)
        for graph in graphs(model.graph):
            if any(node.op_type in {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}
                   or node.op_type.startswith("QLinear") for node in graph.node):
                raise ValueError("INT8/quantized conversion is not supported; supply an FP32 source")
            if any(tensor.data_type == onnx.TensorProto.INT8 for tensor in graph.initializer):
                raise ValueError("INT8 conversion is not supported; supply an FP32 source")
        # Shape inference needs small integer constants in memory. Preserve
        # large weight tensors as external references throughout conversion.
        for _, tensor in tensors(model.graph):
            if edh.uses_external_data(tensor) and tensor.data_type in {
                onnx.TensorProto.INT64, onnx.TensorProto.INT32, onnx.TensorProto.BOOL
            }:
                size = int(np.prod(tensor.dims, dtype=np.int64)) * np.dtype(
                    onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize
                if size <= 1 << 20:
                    metadata = {entry.key: entry.value for entry in tensor.external_data}
                    with (source.parent / metadata["location"]).open("rb") as handle:
                        handle.seek(int(metadata.get("offset", 0)))
                        tensor.raw_data = handle.read(size)
                    if len(tensor.raw_data) != size:
                        raise ValueError(f"Truncated shape tensor: {tensor.name}")
                    tensor.data_location = onnx.TensorProto.DEFAULT
                    del tensor.external_data[:]
        original_float_casts = {
            node.name for graph in graphs(model.graph) for node in graph.node
            if node.op_type == "Cast" and any(a.name == "to" and a.i == onnx.TensorProto.FLOAT for a in node.attribute)
        }
        original_inputs = {value.name: value.type.tensor_type.elem_type for value in model.graph.input}
        original_outputs = {value.name: value.type.tensor_type.elem_type for value in model.graph.output}
        originals = {
            key: ({entry.key: entry.value for entry in tensor.external_data}, tensor.data_type, tuple(tensor.dims))
            for key, tensor in tensors(model.graph) if edh.uses_external_data(tensor)
        }
        blocked = sorted(set(float16.DEFAULT_OP_BLOCK_LIST) | set(FP32_OPERATORS))
        # The converter's optional cast cleanup crashes on fan-out graphs in
        # released versions. Keep casts for TensorRT to fold. A local function
        # namespace avoids mutating the dependency module or shared environment.
        original_convert = float16.convert_float_to_float16
        safe_convert = FunctionType(
            original_convert.__code__,
            {**original_convert.__globals__, "remove_unnecessary_cast_node": lambda graph: None},
            original_convert.__name__, original_convert.__defaults__, original_convert.__closure__,
        )
        model = safe_convert(
            model, keep_io_types=True, disable_shape_infer=True,
            op_block_list=blocked, min_positive_val=float(np.nextafter(np.float16(0), np.float16(1))),
            max_finite_val=65504.0,
        )
        # Existing Cast(to=FLOAT) nodes must follow the converted graph too;
        # newly inserted casts around the FP32 operator blocklist stay FP32.
        for graph in graphs(model.graph):
            for node in graph.node:
                if node.op_type == "Cast" and node.name in original_float_casts:
                    for attribute in node.attribute:
                        if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                            attribute.i = onnx.TensorProto.FLOAT16
        # Some converter versions alter a blocked operator's graph output even
        # with keep_io_types=True. Restore the public boundary explicitly.
        for value in model.graph.input:
            if value.type.tensor_type.elem_type != original_inputs[value.name]:
                raise ValueError(f"Conversion changed public input dtype: {value.name}")
        occupied = {name for node in model.graph.node for name in (*node.input, *node.output)}
        for value in model.graph.output:
            desired = original_outputs[value.name]
            if value.type.tensor_type.elem_type != desired:
                public = value.name
                internal = public + "__rtpolicy_fp16_output"
                while internal in occupied:
                    internal += "_"
                occupied.add(internal)
                for node in model.graph.node:
                    for names in (node.input, node.output):
                        for i, name in enumerate(names):
                            if name == public:
                                names[i] = internal
                for info in model.graph.value_info:
                    if info.name == public:
                        info.name = internal
                value.type.tensor_type.elem_type = desired
                model.graph.node.append(onnx.helper.make_node(
                    "Cast", [internal], [public], to=desired, name=internal + "_cast"))
        # A public output may also feed later nodes (action_pred -> Slice).
        # Internal consumers need the pre-boundary FP16 value, not its FP32
        # public cast, otherwise Slice/arithmetic inherit the wrong dtype.
        for output in model.graph.output:
            boundary = next((node for node in model.graph.node
                             if node.op_type == "Cast" and list(node.output) == [output.name]
                             and any(a.name == "to" and a.i == output.type.tensor_type.elem_type
                                     for a in node.attribute)), None)
            if boundary is not None:
                for node in model.graph.node:
                    if node is boundary:
                        continue
                    for i, name in enumerate(node.input):
                        if name == output.name:
                            node.input[i] = boundary.input[0]
        # Check graph typing before copying any large external weights.
        model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        staged = work / destination.name
        staged_data = work / data_destination.name
        if originals:
            with staged_data.open("wb") as stream:
                for key, tensor in tensors(model.graph):
                    if edh.uses_external_data(tensor):
                        if key not in originals:
                            raise ValueError(f"Unknown external tensor after conversion: {key}")
                        copy_external_tensor(tensor, originals[key], source.parent, stream, data_destination.name)
        onnx.save_model(
            model, str(staged), save_as_external_data=True,
            all_tensors_to_one_file=True, location=data_destination.name, size_threshold=1024,
        )
        onnx.checker.check_model(str(staged), full_check=True)
        copied_data = False
        try:
            if staged_data.exists():
                shutil.copy2(staged_data, data_destination)
                copied_data = True
            shutil.copy2(staged, destination)
        except BaseException:
            destination.unlink(missing_ok=True)
            if copied_data:
                data_destination.unlink(missing_ok=True)
            raise
    result = {"output": str(destination), "graph_bytes": destination.stat().st_size,
              "external_data_bytes": data_destination.stat().st_size if data_destination.exists() else 0,
              "precision": "mixed_fp16", "keep_io_types": True, "fp32_operators": blocked}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--work-dir", help="Temporary workspace parent; defaults to the output directory")
    args = parser.parse_args()
    convert(args.input, args.output, args.work_dir)


if __name__ == "__main__":
    main()
