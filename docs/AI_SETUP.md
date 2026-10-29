# AI Setup and Checkpoint Export

Give a coding agent this guide, the **repository URL or checkout**, a **checkpoint
directory**, and a writable **external artifact root** (EFS where available).
The root requirements file installs PyTorch, RTPolicy, RoboTwin2, and ONNX.
TensorRT is an optional extra for the same CUDA 12 environment; install and run it
when requested. Task evaluation remains an explicit step.

## Copy to your agent

```text
Follow https://github.com/D-Robotics-AI-Lab/RTPolicy/blob/main/docs/AI_SETUP.md for https://github.com/D-Robotics-AI-Lab/RTPolicy, using LOCAL_REPO_PATH=/path/to/RTPolicy, CHECKPOINT_DIRECTORY=/path/to/checkpoint, and ARTIFACT_ROOT=/path/to/external/artifacts.
Create a fresh compatible environment under ARTIFACT_ROOT, install the root requirements.txt, route caches and generated files there, and preserve shared environments and the original checkpoint.
Inspect the checkpoint to derive actual cameras, state/action shapes, history, horizon, active NFE, and preprocessing; preserve those settings and check the native RoboTwin2 image convention.
Verify rtpolicy imports and export static FP32 ONNX with --verify, explicitly distinguishing synthetic export checks from representative-input validation.
If I request TensorRT, install the tensorrt extra, build and run mixed-FP16 inference on NVIDIA, and compare it with the FP32 reference. Prepare cuRobo and simulator assets only if I request task evaluation, keeping RoboTwin2 code unchanged.
Validate each requested stage once; repeat an evaluation only if I request it or a new failure requires a targeted check.
Return reproducible commands, resolved versions/configuration, I/O, numerical errors and tolerances, artifact paths, and exact missing-resource blockers, keeping generated results outside Git.
```

## 1. Inspect the checkpoint

Read `config.json` and the safetensors header before loading large weights.
The current loader requires `config.json` and `model.safetensors`; identify missing
files or an incompatible weight layout explicitly. Use the installed `rtpolicy`
classes and `rtpolicy.common.checkpoint.load_checkpoint_config`, which normalizes
supported historical class and backbone metadata in memory. Inference and export
also normalize known historical tensor names and require complete weight coverage.

Record `obs_encoder.rgb_key`, `shape_meta`, `action_key`/`supervision_key`,
`n_obs_steps`, `n_image_obs_steps`, `horizon`, `n_action_steps`, and
`noise_scheduler.num_inference_timesteps` (the active NFE). Resolve the referenced
backbone configuration/processor assets and check policy/normalizer weight coverage.
Derive shapes from this checkpoint rather than a previous experiment.

Default RoboTwin2 D435 source images are **240 × 320 (H × W)**. Preserve the
checkpoint's preprocessing, including internal alignment; changing raw image size
or NFE requires an explicit experiment request. Keep the source checkpoint unchanged.
If the user supplies a benchmark NFE, apply it through the export configuration
and the evaluation override, without editing the checkpoint's `config.json`.

## 2. Prepare an isolated environment

Read the root [`requirements.txt`](../requirements.txt) and
[`pyproject.toml`](../pyproject.toml) for the unified stack and optional dependencies.
Inspect the host and select a
compatible interpreter and matched PyTorch/torchvision build before installing
repository dependencies. Use a fresh venv or conda prefix under the artifact root.
Route package/model caches and temporary files there too (`PIP_CACHE_DIR`, `HF_HOME`,
`TORCH_HOME`, `XDG_CACHE_HOME`, `TMPDIR`; `CUDA_CACHE_PATH` for NVIDIA).
Resolve symlinks to confirm the actual storage location.

Run from the repository root in the new environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import rtpolicy, rtpolicy.models; print(rtpolicy.__file__)"
python -m rtpolicy.trainer.onnx --help
```

Record the ONNX execution provider actually used. Fix dependency or import
failures in the isolated environment. cuRobo compilation and simulator assets
belong to an explicitly requested task-evaluation step.

## 3. Export and verify FP32

Run from the repository root in the new environment. Use the same
[RoboTwin2 export YAML](../configs/rtpolicy_onnx/rtpolicy_onnx_robotwin2.yaml) as the
README: set `checkpoint_dir`, a fresh external `output_dir`, and the selected
`device`. Keep `num_inference_timesteps: null` unless the user supplied a benchmark
override. Static FP32 export, image preprocessing, verification, and external
calibration outputs are configured in that YAML.

```bash
python main.py --config-path=configs/rtpolicy_onnx \
  --config-name=rtpolicy_onnx_robotwin2
```

The configured output directory keeps samples and metrics outside the checkpoint.
Keep the `.onnx`, its external weight file, and
`.onnx.metadata.json` together; verify their actual I/O against the checkpoint.
Read the per-sample `verification_metrics.json`, require finite outputs, and report
both `action` and `action_pred` errors and tolerances.

Without a calibration dataset, the exporter uses synthetic inputs. Label that result **synthetic
export verification**. Replay representative user-provided observations for a
real-input numerical check, using the recorded I/O and the same noise trajectory.
If those observations are absent, report that check as pending; numerical export
verification alone does not establish robot task success. Missing weights or
required backbone assets block export until resolved.

For requested closed-loop validation, use the external
[evaluation configuration](../configs/rtpolicy_eval/robotwin2.yaml) and
[launcher](../scripts/eval/run_robotwin2.py). Keep the simulator's environment,
task, and evaluation code unchanged. The launcher calls its original evaluator
from an external working directory; backend adaptation stays in `rtpolicy`.
Record task settings, seed, NFE, JPEG setting, precision, original result logs,
and actual accepted seeds. Report each backend's measured success count without
altering the evaluator's preflight or case selection to force agreement.

## 4. NVIDIA TensorRT inference

For a requested TensorRT target, append its dependencies to the existing
environment and use an idle NVIDIA GPU. Starting from the verified static FP32
export:

```bash
python -m pip install -e '.[tensorrt]'

python scripts/fp16_convert.py \
  /path/to/external/artifacts/model.onnx /path/to/external/artifacts/model-fp16.onnx
python scripts/build_bench_x86.py build \
  /path/to/external/artifacts/model-fp16.onnx /path/to/external/artifacts/model-fp16.engine \
  strongly_typed

python scripts/infer_tensorrt.py \
  --engine-path /path/to/external/artifacts/model-fp16.engine \
  --inputs-dir /path/to/external/artifacts/onnx_calibration/sample_000/inputs \
  --output-dir /path/to/external/artifacts/tensorrt_outputs
```

Execute the engine on the same saved inputs and compare it with the FP32 reference;
report finite checks, per-output errors, declared tolerances, and the input source.
Sensitive operators remain FP32. **S100/S600 BPU INT16 uses a separate toolchain**
and needs its own target validation.

For requested RoboTwin2 evaluation, select `backend: tensorrt` in the external
evaluation YAML, set `model_path` to the engine and `metadata_path` to the original
FP32 export sidecar, and call the unchanged evaluator through the launcher.

## Deliver the result

Write a short report and machine-readable manifest under the artifact root:
source revision/checkpoint identity, environment creation and replay commands,
resolved package versions, sampling/preprocessing configuration, I/O names/shapes/
dtypes, verification input source and metrics, and any remaining blockers.
List generated files with real storage paths and cleanup conditions. Leave model
artifacts, environments, caches, samples, and reports outside repository commits.
