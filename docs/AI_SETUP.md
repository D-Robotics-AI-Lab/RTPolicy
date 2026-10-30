# AI Setup and Model Validation

Give a coding agent this guide and a writable **external artifact root** (EFS where
available). It can clone the repository, install the environment, download the
released models from **Google Drive**, and validate an ONNX export automatically.
The default case is **lift_pot, NFE4, seed0**. A local checkpoint is optional.

## Copy to your agent

```text
Follow https://github.com/D-Robotics-AI-Lab/RTPolicy/blob/main/docs/AI_SETUP.md for https://github.com/D-Robotics-AI-Lab/RTPolicy, using ARTIFACT_ROOT=/path/to/external/artifacts. Use an existing checkout if available; otherwise clone the repository.
Create an isolated compatible environment, install the root requirements.txt, and route downloads, checkpoints, caches and outputs outside Git.
Automatically download the released lift_pot PyTorch model from the Google Drive links in the guide, verify its pinned SHA256 and safely extract it. Reuse verified cached files. A local checkpoint directory is optional.
Inspect the checkpoint and export static FP32 ONNX with verification, using the released lift_pot protocol NFE4 and seed0 through configuration overrides; preserve checkpoint weights and stored config.json. If I provide another checkpoint or protocol, use those settings instead.
Validate the exported policy once and report action/action_pred errors against the same FP32 reference and identical inputs/noise. Label synthetic inputs explicitly.
If I request published ONNX or TensorRT validation, download and verify the matching package too. Check the TensorRT build environment before using its engine; rebuild from FP32 ONNX when the host requires it.
Run RoboTwin2 task evaluation only when I request it, through the external launcher with the simulator code unchanged. Validate each requested stage once.
Return reproducible commands, source/download identities and checksums, resolved versions/settings, real artifact paths and validation results. Report concrete blockers without substituting different models or claiming unrun tests.
```

## 1. Prepare an isolated environment

Inspect the host and choose a compatible interpreter and matched PyTorch/torchvision
build. Create a fresh environment under `ARTIFACT_ROOT`; preserve shared environments.
Read the root [requirements.txt](../requirements.txt) and [pyproject.toml](../pyproject.toml).

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import rtpolicy, rtpolicy.models; print(rtpolicy.__file__)"
```

Create external `downloads`, `checkpoints`, `exports`, and `evaluations` directories.
Route package/model and temporary caches there (`PIP_CACHE_DIR`, `HF_HOME`,
`TORCH_HOME`, `XDG_CACHE_HOME`, `TMPDIR`, and `CUDA_CACHE_PATH` for NVIDIA).
Resolve symlinks to confirm where files will be written.

## 2. Download from Google Drive

[Released checkpoint folder](https://drive.google.com/drive/folders/1UxLqe8mIYPSrPwuCchVBL2StSbZ67_0x).
These public downloads use `gdown`, included in the root requirements. Google Drive
account authorization is not needed for downloading the published files.

| Model | Purpose | Google Drive |
| --- | --- | --- |
| **lift_pot · PyTorch** | **Default: export and numerical validation** | [Download](https://drive.google.com/file/d/1382Y1VLgH7LEyJH49ZuVK5N0vsZo3XP5/view) |
| lift_pot · ONNX FP32 | Validate the published NFE4 export | [Download](https://drive.google.com/file/d/1RbjJ23fG8AjcpAz9O2Qlz75UXzMvb1NT/view) |
| lift_pot · TensorRT mixed FP16 | Validate a compatible prebuilt NFE4 engine | [Download](https://drive.google.com/file/d/1wjgUX8ptoTxo3MZX3gk3ke532PKDmYdK/view) |
| RoboTwin2 pretraining · PyTorch | Initialize fine-tuning | [Download](https://drive.google.com/file/d/1mUBhBGJkg7Pc1zkn1EEEqU6XcfmXxMQp/view) |

Download only the packages needed for the requested route. For the default case,
create the external download directory and run:

```bash
python -m gdown --fuzzy \
  "https://drive.google.com/file/d/1382Y1VLgH7LEyJH49ZuVK5N0vsZo3XP5/view" \
  --no-cookies --continue \
  --output /path/to/external/artifacts/downloads/rtpolicy_lift_pot_pytorch_ft30k.zip
```

Verify the archive SHA256 before extraction, using the pinned values below or the
published [SHA256SUMS](https://drive.google.com/file/d/1_Kynsf4PFK1qMwteFxDhUkGX7tzfeo2M/view).
Reuse matching cached downloads. Extract into the external `checkpoints` directory;
reject absolute paths, parent traversal and symbolic links in archive entries.
Preserve existing unrelated files. If an extracted model already exists, verify
its files against the archive's `manifest.json` before reusing it.

<details>
<summary>Release SHA256 checksums</summary>

```text
770d9f664b0804655c778b72baac9e04880b94d1a8298084464ad13ed13c9037  rtpolicy_robotwin2_pretrain_cnnb_3v_300k.zip
0948278d257585e1510d6b25822fb65359a2638c86ecc490a928a8816b31d046  rtpolicy_lift_pot_pytorch_ft30k.zip
afd393d93e3c1152cdc8209701b2d0eaaed8a578286e85ca7a480b7218c14b26  rtpolicy_lift_pot_onnx_fp32_nfe4.zip
96507bbd2492a747c9263216ddbbd531f1aed0908c8d3b6b14473993ae7af1da  rtpolicy_lift_pot_tensorrt_fp16_nfe4_l4_trt11.1.zip
```

</details>

### Inspect the checkpoint

The default ZIP extracts to `rtpolicy_lift_pot_pytorch_ft30k/`, containing
`config.json`, `model.safetensors`, `evaluation.yaml`, and usage notes. It includes
the visual-backbone weights and configuration. Load through the installed
`rtpolicy` classes and require complete policy/normalizer weight coverage.

Read the config and safetensors header before loading large tensors. Derive camera
keys, shapes, history, horizon, action chunk, preprocessing and sampling settings
from this model. RoboTwin2 source images are **240 × 320 (H × W)**.
The released PyTorch checkpoint stores NFE10; apply the **NFE4** release protocol
through export/evaluation overrides, keeping the stored checkpoint unchanged.
For a user-provided checkpoint, preserve its settings unless a protocol override
was explicitly supplied.

## 3. Export and verify FP32

Use the [RoboTwin2 export YAML](../configs/rtpolicy_onnx/rtpolicy_onnx_robotwin2.yaml).
With the default downloaded checkpoint, run:

```bash
python main.py --config-path=configs/rtpolicy_onnx \
  --config-name=rtpolicy_onnx_robotwin2 \
  checkpoint_dir=/path/to/external/artifacts/checkpoints/rtpolicy_lift_pot_pytorch_ft30k \
  output_dir=/path/to/external/artifacts/exports/lift_pot_nfe4 \
  num_inference_timesteps=4 seed=0
```

Select the export device according to the host. The YAML enables static FP32
export, image preprocessing, numerical verification and one saved calibration
sample. Require finite `action` and `action_pred` outputs and check
`verification_metrics.json` with the configured `rtol/atol: 1e-4`.
Keep the `.onnx`, `.onnx.data`, and `.onnx.metadata.json` together.

The verification compares the same weights, observations and noise trajectory
against a reproducible FP32 mathematical reference. With no supplied dataset,
label the result **synthetic export verification**. When representative inputs
are available, replay those inputs and report their source separately.

If asked to test the published ONNX package, download and verify that ZIP too,
then run its graph on the same saved input tensors using ONNX Runtime. Record the
provider actually used and compare both outputs with the same reference.
Use one numerical check; run latency benchmarking only when requested.

## 4. NVIDIA TensorRT inference

For a requested TensorRT target:

```bash
python -m pip install -e '.[tensorrt]'
```

The published engine was built on **NVIDIA L4 with TensorRT 11.1.0.106**.
Download its ZIP only when needed, verify its SHA256, and inspect
`build_environment.json` before selecting the prebuilt engine. Keep
`model.engine.metadata.json` alongside `model.engine`.

On a compatible host, validate the engine on the same saved export inputs:

```bash
python scripts/infer_tensorrt.py \
  --engine-path /path/to/external/artifacts/checkpoints/rtpolicy_lift_pot_tensorrt_fp16_nfe4_l4_trt11.1/model.engine \
  --inputs-dir /path/to/external/artifacts/exports/lift_pot_nfe4/onnx_calibration/sample_000/inputs \
  --output-dir /path/to/external/artifacts/tensorrt_outputs
```

When the host requires a rebuilt engine, use the verified FP32 ONNX export:

```bash
python scripts/fp16_convert.py \
  /path/to/external/artifacts/exports/lift_pot_nfe4/model.onnx \
  /path/to/external/artifacts/exports/lift_pot_nfe4/model-fp16.onnx
python scripts/build_bench_x86.py build \
  /path/to/external/artifacts/exports/lift_pot_nfe4/model-fp16.onnx \
  /path/to/external/artifacts/exports/lift_pot_nfe4/model-fp16.engine strongly_typed
```

Execute each selected engine once on the saved inputs. Compare finite outputs with
the FP32 reference, reporting errors and declared tolerances (mixed-FP16
`rtol/atol: 1e-2`). Preserve the input shapes and NFE4 across all selected backends.

## 5. Optional RoboTwin2 task evaluation

For a task-success measurement, add this to the agent prompt:

```text
Also evaluate lift_pot once with the requested backend using the released evaluation.yaml, NFE4, seed0, and 100 episodes. Prepare the official simulator assets and dependencies on external storage; keep RoboTwin2 task/evaluator code and case selection unchanged. Preserve logs and actual accepted seeds.
```

For a rebuilt TensorRT engine, set `metadata_path` to the verified FP32 export's
`model.onnx.metadata.json`; the public input/output types are preserved by conversion.

Copy the selected package's `evaluation.yaml` into the external artifact root and
set its absolute `ckpt_setting` (PyTorch) or `model_path` (ONNX/TensorRT). The
included YAML already fixes the released task, NFE and seed. Prepare cuRobo,
SAPIEN/Vulkan and official assets when this evaluation is requested, then use the
[external launcher](../scripts/eval/run_robotwin2.py):

```bash
python scripts/eval/run_robotwin2.py \
  --config /path/to/external/artifacts/evaluation.yaml \
  --runtime-dir /path/to/external/artifacts/evaluations/lift_pot
```

Numerical export checks and robot task-success measurements are distinct results.
Run each requested backend evaluation once; repeat only for an explicit request
or a targeted check required by a new failure.

## Deliver the result

Report download URLs and verified hashes, repository revision, model directory,
resolved environment versions, sampling/preprocessing overrides, commands,
I/O names/shapes/dtypes, numerical errors and input sources. Include task-success
logs only for evaluations actually performed. Record real artifact/cache paths
and cleanup conditions; keep downloads, models, environments and outputs outside Git.
