"""Run one TensorRT inference using an exported sample's named NumPy inputs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-path', required=True, type=Path)
    parser.add_argument('--inputs-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.build_bench_x86 import Runner, load_inputs

    if not args.engine_path.is_file():
        raise FileNotFoundError(args.engine_path)
    if not args.inputs_dir.is_dir():
        raise NotADirectoryError(args.inputs_dir)
    inputs = load_inputs(args.inputs_dir)
    if not inputs:
        raise ValueError(f'No named .npy inputs found in {args.inputs_dir}')
    with Runner(args.engine_path) as runner:
        targets = {name: args.output_dir / f'{name}.npy' for name in runner.output_names}
        for target in targets.values():
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite output: {target}')
        outputs = runner.run(inputs)
    if any(not np.isfinite(value).all() for value in outputs.values()):
        raise RuntimeError('TensorRT produced nonfinite outputs')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        np.save(targets[name], value)
    print(json.dumps({'engine': str(args.engine_path), 'outputs': {
        name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'file': str(targets[name])}
        for name, value in outputs.items()}}, indent=2))


if __name__ == '__main__':
    main()
