# AI Setup and Checkpoint Export

Give a coding agent this guide, the **repository URL or checkout**, a **checkpoint
directory**, and a writable **external artifact root** (EFS where available).
ONNX is the default target; select NVIDIA separately for TensorRT validation.

## Copy to your agent

```text
Follow https://github.com/D-Robotics-AI-Lab/RTPolicy/blob/main/docs/AI_SETUP.md for https://github.com/D-Robotics-AI-Lab/RTPolicy, using LOCAL_REPO_PATH=/path/to/RTPolicy, CHECKPOINT_DIRECTORY=/path/to/checkpoint, and ARTIFACT_ROOT=/path/to/external/artifacts.
Create a fresh compatible environment under ARTIFACT_ROOT, route caches and generated files there, and preserve shared environments and the original checkpoint.
Inspect the checkpoint to derive actual cameras, state/action shapes, history, horizon, active NFE, and preprocessing; preserve those settings and check the native RoboTwin2 image convention.
Verify rtpolicy imports and export static FP32 ONNX with --verify, explicitly distinguishing synthetic export checks from representative-input validation.
If I select NVIDIA, also build and validate mixed FP16 separately from BPU INT16; install simulation software or assets only if I request evaluation.
Return reproducible commands, resolved versions/configuration, I/O, numerical errors and tolerances, artifact paths, and exact missing-resource blockers, keeping generated results outside Git.
```

## 1. Inspect the checkpoint

Read `config.json` and the safetensors header before loading large weights.
The current loader requires `config.json` and `model.safetensors`; identify missing
files or an incompatible weight layout explicitly. Use the installed `rtpolicy`
classes and `rtpolicy.common.checkpoint.load_checkpoint_config`, which normalizes
supported historical class references in memory.

Record `obs_encoder.rgb_key`, `shape_meta`, `action_key`/`supervision_key`,
`n_obs_steps`, `n_image_obs_steps`, `horizon`, `n_action_steps`, and
`noise_scheduler.num_inference_timesteps` (the active NFE). Resolve the referenced
backbone configuration/processor assets and check policy/normalizer weight coverage.
Derive shapes from this checkpoint rather than a previous experiment.

Default RoboTwin2 D435 source images are **240 × 320 (H × W)**. Preserve the
checkpoint's preprocessing, including internal alignment; changing raw image size
or NFE requires an explicit experiment request. Keep the source checkpoint unchanged.

## 2. Prepare an isolated environment

Read [`pyproject.toml`](../pyproject.toml) for the runtime and optional dependencies.
Inspect the host and select a
compatible interpreter and matched PyTorch/torchvision build before installing
repository dependencies. Use a fresh venv or conda prefix under the artifact root.
Route package/model caches and temporary files there too (`PIP_CACHE_DIR`, `HF_HOME`,
`TORCH_HOME`, `XDG_CACHE_HOME`, `TMPDIR`; `CUDA_CACHE_PATH` for NVIDIA).
Resolve symlinks to confirm the actual storage location.

For a CPU ONNX environment, after installing the selected PyTorch pair:

```bash
python -m pip install -e /path/to/RTPolicy
python -m pip install onnx onnxruntime
python -m pip check
python -c "import rtpolicy, rtpolicy.models; print(rtpolicy.__file__)"
python -m rtpolicy.trainer.onnx --help
```

Use `onnxruntime-gpu` in place of `onnxruntime` when CUDA verification is selected,
and record the provider actually used. Fix dependency or import failures in the
isolated environment. Simulator installation and assets belong to an explicitly
requested evaluation step.

## 3. Export and verify FP32

Run from the repository root in the new environment. Substitute the supplied paths;
use a fresh artifact directory. This command keeps the checkpoint's sampling settings:

```bash
python -m rtpolicy.trainer.onnx \
  --checkpoint-dir /path/to/checkpoint \
  --onnx-path /path/to/external/artifacts/model.onnx \
  --device cpu --export-dtype fp32 --static-batch \
  --image-preprocess-in-onnx --no-export-modules-as-functions \
  --calibration-data-dir /path/to/external/artifacts/verification \
  --no-save-netron-model --verify
```

The absolute calibration path keeps generated samples and metrics outside the
checkpoint directory. Keep the `.onnx`, its external weight file, and
`.onnx.metadata.json` together; verify their actual I/O against the checkpoint.
Read the per-sample `verification_metrics.json`, require finite outputs, and report
both `action` and `action_pred` errors and tolerances.

The direct CLI uses synthetic inputs by default. Label that result **synthetic
export verification**. Replay representative user-provided observations for a
real-input numerical check, using the recorded I/O and the same noise trajectory.
If those observations are absent, report that check as pending; numerical export
verification alone does not establish robot task success. Missing weights or
required backbone assets block export until resolved.

## 4. Optional NVIDIA mixed FP16

For a selected NVIDIA target, use an idle GPU and an isolated compatible TensorRT
setup with `cuda-python` and `onnxconverter-common`. Starting from the verified
static FP32 export:

```bash
python scripts/fp16_convert.py \
  /path/to/external/artifacts/model.onnx /path/to/external/artifacts/model-fp16.onnx
python scripts/build_bench_x86.py build \
  /path/to/external/artifacts/model-fp16.onnx /path/to/external/artifacts/model-fp16.engine \
  strongly_typed
```

Execute the engine on the same saved inputs and compare it with the FP32 reference;
report finite checks, per-output errors, declared tolerances, and the input source.
Sensitive operators remain FP32. **S100/S600 BPU INT16 uses a separate toolchain**
and needs its own target validation.

## Deliver the result

Write a short report and machine-readable manifest under the artifact root:
source revision/checkpoint identity, environment creation and replay commands,
resolved package versions, sampling/preprocessing configuration, I/O names/shapes/
dtypes, verification input source and metrics, and any remaining blockers.
List generated files with real storage paths and cleanup conditions. Leave model
artifacts, environments, caches, samples, and reports outside repository commits.
