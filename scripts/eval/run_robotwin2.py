"""Launch the original RoboTwin2 evaluator from an external runtime directory."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--runtime-dir', required=True, type=Path)
    parser.add_argument('--simulator-root', type=Path, default=repository/'simulation/robotwin2.0')
    parser.add_argument('--evaluator', default='script/eval_policy_parallel.py')
    parser.add_argument('--bootstrap-path', type=Path, default=repository/'tools/robotwin_bootstrap',
                        help='External directory for dependencies required by the selected evaluator')
    args = parser.parse_args()
    simulator = args.simulator_root.resolve()
    runtime = args.runtime_dir.resolve()
    if runtime == simulator or runtime.is_relative_to(simulator):
        raise ValueError('Use a runtime directory outside the simulator source tree')
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ('envs','assets','policy','description','task_config'):
        target = simulator/name
        if not target.is_dir():
            raise FileNotFoundError(target)
        entry = runtime/name
        if not entry.exists():
            entry.symlink_to(target, target_is_directory=True)
        elif entry.resolve() != target.resolve():
            raise ValueError(f'Runtime resource points to a different source: {entry}')
    pretrained = repository/'pretrained'
    if pretrained.exists() and not (runtime/'pretrained').exists():
        (runtime/'pretrained').symlink_to(pretrained.resolve(), target_is_directory=True)
    environment = os.environ.copy()
    paths = [str(repository)]
    if args.bootstrap_path:
        paths.append(str(args.bootstrap_path.resolve()))
    if environment.get('PYTHONPATH'):
        paths.append(environment['PYTHONPATH'])
    environment['PYTHONPATH'] = os.pathsep.join(paths)
    evaluator = (simulator/args.evaluator).resolve()
    if not evaluator.is_relative_to(simulator):
        raise ValueError('Evaluator must be inside the selected simulator snapshot')
    command = [sys.executable, str(evaluator), '--config', str(args.config.resolve())]
    print(f'Running unchanged evaluator: {evaluator}', flush=True)
    result = subprocess.run(command, cwd=runtime, env=environment)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
