<div align="center">

<picture>
<source media="(prefers-color-scheme: dark)" srcset="assets/rtpolicy-logo-dark.svg">
<img src="assets/rtpolicy-logo.svg" width="360" alt="RTPolicy">
</picture>

<picture>
<source media="(prefers-color-scheme: dark)" srcset="assets/rtpolicy-paper-title-dark.svg">
<img src="assets/rtpolicy-paper-title.svg" width="100%" alt="Real-Time (32.9 Hz) Multi-View Visuomotor Policy for Robotic Manipulation on Edge Devices">
</picture>

<p>
<b>Xianbao Hou</b><sup>1,3,*</sup> · <b>Zehao Ni</b><sup>2,3,*</sup> · <b>Yonghao He</b><sup>4,†</sup> · <b>Lingfeng Qian</b><sup>3</sup><br>
<b>Xin Shi</b><sup>3</sup> · <b>Hu Su</b><sup>5</sup> · <b>Wei Sui</b><sup>3,✉</sup> · <b>Cong Yang</b><sup>1,✉</sup>
</p>

<p>
<sup>1</sup>School of Future Science and Engineering, Soochow University<br>
<sup>2</sup>Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University<br>
<sup>3</sup>D-Robotics &nbsp; <sup>4</sup>Intime AI<br>
<sup>5</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems,<br>Institute of Automation, Chinese Academy of Sciences
</p>

<p><sup>*</sup>Equal contribution &nbsp; <sup>†</sup>Project lead &nbsp; <sup>✉</sup>Corresponding authors</p>

<p>Wei Sui: <a href="mailto:wei.sui@d-robotics.cc">wei.sui@d-robotics.cc</a><br>
Cong Yang: <a href="mailto:cong.yang@suda.edu.cn">cong.yang@suda.edu.cn</a></p>

<p align="center">
<a href="https://github.com/D-Robotics-AI-Lab/RTPolicy"><img src="assets/badge-code.svg" height="24" alt="Code: GitHub"></a>
<a href="https://github.com/D-Robotics-AI-Lab/RTPolicy.project"><img src="assets/badge-project.svg" height="24" alt="Project page source"></a>
<a href="LICENSE"><img src="assets/badge-license.svg" height="24" alt="License: see LICENSE"></a>
</p>

<p align="center">
<a href="#1-introduction"><picture><source media="(prefers-reduced-motion: reduce)" srcset="assets/navigation/nav-introduction-compact-centered-static.svg"><img src="assets/navigation/nav-introduction-compact-centered.svg" width="165" height="46" alt="Introduction"></picture></a>
<a href="#2-model-overview"><picture><source media="(prefers-reduced-motion: reduce)" srcset="assets/navigation/nav-model-overview-compact-centered-static.svg"><img src="assets/navigation/nav-model-overview-compact-centered.svg" width="165" height="46" alt="Model overview"></picture></a>
<a href="#3-getting-started"><picture><source media="(prefers-reduced-motion: reduce)" srcset="assets/navigation/nav-get-started-compact-centered-static.svg"><img src="assets/navigation/nav-get-started-compact-centered.svg" width="165" height="46" alt="Get started"></picture></a>
<a href="#4-training-export-and-inference"><picture><source media="(prefers-reduced-motion: reduce)" srcset="assets/navigation/nav-train-deploy-compact-centered-static.svg"><img src="assets/navigation/nav-train-deploy-compact-centered.svg" width="165" height="46" alt="Train &amp; deploy"></picture></a>
</p>

https://github.com/user-attachments/assets/bec0058a-a3c2-423e-a642-4f3794a3e0c9

**Multi-view perception. Compact visual tokens. Four-step action generation.**

</div>

RTPolicy turns multiple RGB views and robot state into action chunks with a compact
flow-matching policy. It combines multi-view reasoning with a deployment path for
edge devices, reaching **32.9 Hz on S600 in INT16** in the paper's latency evaluation.

## 🔥 Updates

RTPolicy code and demonstration are available.

<a id="1-introduction"></a>

## 🧭 1. Introduction

### 1.1 Brief Introduction of RTPolicy

<div align="center">
<img src="assets/architecture.webp" width="100%" alt="RTPolicy architecture: frozen DINOv3 ConvNeXt, view-aware feature fusion, visual compression, and a flow-matching DiT conditioned on robot state">
</div>

RTPolicy fuses dense visual features across cameras before compressing them, retaining
cross-view information in a small representation for the action model.

- **Reason across views.** A frozen DINOv3 ConvNeXt backbone and view-aware feature fusion
  combine complementary observations from three RGB cameras.
- **Compress before action generation.** In the paper's 480 × 640 (H × W) setting, visual compression reduces
  **900 visual tokens to 12**, giving the DiT a compact observation representation.
- **Generate action chunks in four steps.** A flow-matching DiT conditions on the visual
  representation and robot state to predict actions with four integration steps.

### 1.2 Repository Structure

| Path                                                                   | What to find                                                        |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [`rtpolicy/models/policy/rtpolicy/`](rtpolicy/models/policy/rtpolicy/) | Visual encoder, multi-view fusion, adapters, DiT, and policy runner |
| [`rtpolicy/datasets/`](rtpolicy/datasets/)                             | Dataset readers and preprocessing interfaces                        |
| [`rtpolicy/trainer/`](rtpolicy/trainer/)                               | Training and ONNX export                                            |
| [`configs/`](configs/)                                                 | Training, observation, robot, and ONNX export configurations        |
| [`simulation/robotwin2.0/`](simulation/robotwin2.0/)                   | RoboTwin2.0 environments and RTPolicy adapter                       |
| [`scripts/`](scripts/) · [`tools/`](tools/)                            | Evaluation, inference, TensorRT tools, and dataset preparation      |
| [`docs/`](docs/)                                                       | [Runtime guide](docs/README.md)                                     |

<a id="2-model-overview"></a>

## 📊 2. Model Overview

### 2.1 RoboTwin2.0 Simulation

The clean-to-clean benchmark uses 50 tasks, 50 clean demonstrations per task,
100 evaluation episodes per task, and task-specific checkpoints.

| Method           | Mean success rate · 50 tasks |
| ---------------- | ---------------------------: |
| Diffusion Policy |                        28.0% |
| ACT              |                        29.7% |
| RDT              |                        34.5% |
| π0               |                        46.4% |
| **RTPolicy**     |                    **52.3%** |

Baseline scores are taken from the official single-task leaderboard.

[Model checkpoints ↓](#24-model-checkpoints)

### 2.2 Latency on GPU

| Platform            | Precision |   Latency | Frequency |
| ------------------- | --------- | --------: | --------: |
| RTX 4090            | FP32      |  23.80 ms |   42.0 Hz |
| RTX 4090            | FP16      |  10.92 ms |   91.6 Hz |

### 2.3 Latency on S600 and S100

| Platform | Precision |      Latency |   Frequency |
| -------- | --------- | -----------: | ----------: |
| **S600** | **INT16** | **30.35 ms** | **32.9 Hz** |
| S100     | INT16     |    149.72 ms |      6.7 Hz |

#### Quantization HTML Reports

| Device | Precision | File                                                                                                                                               |
| ------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| S600   | INT16     | <a href="https://github.com/D-Robotics-AI-Lab/RTPolicy.project/releases/download/project-files/s600.html"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/download-s600-dark.svg"><img src="assets/download-s600-light.svg" width="184" height="32" alt="Download S600 HTML"></picture></a> |
| S100   | INT16     | <a href="https://github.com/D-Robotics-AI-Lab/RTPolicy.project/releases/download/project-files/s100.html"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/download-s100-dark.svg"><img src="assets/download-s100-light.svg" width="184" height="32" alt="Download S100 HTML"></picture></a> |

### 2.4 Model Checkpoints

| Model       | Google Drive |
| ----------- | ------------ |
| RoboTwin2.0 |              |

<a id="3-getting-started"></a>

## 🚀 3. Getting Started

### 3.1 Installation

```bash
git clone https://github.com/D-Robotics-AI-Lab/RTPolicy.git
cd RTPolicy
python -m pip install -r requirements.txt
```

The root [requirements.txt](requirements.txt) covers PyTorch, RTPolicy, RoboTwin2,
and ONNX. TensorRT can be added to the same CUDA 12 environment in Section 4.4.

### 3.2 Prepare Data and Weights

Supply compatible DINOv3 weights and set
`model.obs_encoder.emvis_config.backbone_path` to their directory. Datasets,
checkpoints, simulator assets, and generated outputs are kept outside Git.

Train directly on LeRobot v2.1 datasets with the default backend. Lance is optional; the [Training Guide](docs/README.md) also covers converting LeRobot v3.0 data to JPEG-blob Lance.

For RoboTwin2.0 simulation, prepare SAPIEN, Vulkan, cuRobo, and the official assets:

```bash
cd simulation/robotwin2.0
bash script/_download_assets.sh
cd ../..
```

### 3.3 🤖 AI-assisted Setup and Validation

Give your coding agent the [AI Setup Guide](docs/AI_SETUP.md), a checkpoint,
and an external artifact directory. Replace the paths below and paste this prompt:

```text
Follow https://github.com/D-Robotics-AI-Lab/RTPolicy/blob/main/docs/AI_SETUP.md.
Repository/checkout: /path/to/RTPolicy
Checkpoint: /path/to/checkpoint
External artifacts: /path/to/external/artifacts
Create an isolated compatible environment, install the root requirements.txt,
verify imports, then export and verify ONNX.
If I request TensorRT, install the tensorrt extra and build/run the engine.
Preserve checkpoint settings and any benchmark overrides I provide.
Keep RoboTwin2 code unchanged; run task evaluation only when I request it.
Validate each requested stage once and return the commands, versions, outputs,
and numerical errors, with generated files outside the repository.
```

<a id="4-training-export-and-inference"></a>

## ⚡ 4. Training, Export and Inference

### 4.1 Train a Policy

Follow the [Training Guide](docs/README.md) to train directly with LeRobot or use the optional Lance backend.

The RoboTwin2 pretraining and fine-tuning recipes use DINOv3 ConvNeXt-Base:

| Stage | Configuration | Dataset | Training steps |
| --- | --- | --- | ---: |
| Pretrain | [Three-view Lance](configs/rtpolicy_train/pt_dinov3cnnb_3v_lance_300k.yaml) | 50 tasks, 2500 demonstrations | 300K |
| Fine-tune | [Three-view LeRobot](configs/rtpolicy_train/ft_dinov3cnnb_3v_30k.yaml) | 50 demonstrations per task | 30K |

See [Pretrain and fine-tune](docs/README.md#pretrain-and-fine-tune) for configuration fields and commands.

### 4.2 Evaluate in RoboTwin2.0

Set the checkpoint and backend in the external
[evaluation configuration](configs/rtpolicy_eval/robotwin2.yaml). The adapter
uses RoboTwin2's original policy interface; simulator, task, and evaluation code
remain unchanged.

```bash
python scripts/eval/run_robotwin2.py \
  --config configs/rtpolicy_eval/robotwin2.yaml \
  --runtime-dir /path/to/external/evaluation
```

JPEG input alignment is enabled by default through `RTPOLICY_JPEG_ROUNDTRIP=1`.
Select `backend`, `model_path`, `metadata_path`, and `test_num` in the
[evaluation configuration](configs/rtpolicy_eval/robotwin2.yaml).
The launcher prepares an external working directory and calls the original
evaluator. It does not alter preflight checks or seed selection. Keep checkpoint,
NFE, seed, and task settings identical when comparing backends.

Build cuRobo v0.7.8 in an external directory using the same environment. Set
`WARP_CACHE_PATH` outside the repository. If its filesystem rejects directory
renames with `EXDEV`, run `python tools/patch_warp_cache.py --apply` in this
isolated environment to enable an atomic file-copy fallback for the JIT cache.

```bash
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git /path/to/external/curobo
MAX_JOBS=1 python -m pip install -e /path/to/external/curobo \
  --no-build-isolation -r requirements.txt
```

### 4.3 Export and Run ONNX

Set `checkpoint_dir` and `output_dir` in the
[RoboTwin2 export example](configs/rtpolicy_onnx/rtpolicy_onnx_robotwin2.yaml).

```bash
python main.py --config-path=configs/rtpolicy_onnx \
  --config-name=rtpolicy_onnx_robotwin2

python scripts/infer_onnx.py --help
```

The exported graph follows the checkpoint configuration, including its sampling settings.
For a benchmark with an explicit NFE override, set `num_inference_timesteps` in
the export YAML and use the same `RTPOLICY_NFE` during evaluation.
To generate samples from a converted RoboTwin2 dataset, set `train_dataset.root`
to its LeRobot v2.1 directory and choose `num_calibration_samples`.
The example saves verification inputs and outputs under `output_dir/onnx_calibration`.

### 4.4 TensorRT and Edge Deployment

Use **FP16 for NVIDIA TensorRT** and **INT16 for S600 / S100 BPU deployment**.

Add the optional TensorRT tools to the installed environment:

```bash
python -m pip install -e '.[tensorrt]'
```

Set `RTPOLICY_ARTIFACT_DIR` to the export configuration's `output_dir`:

```bash
export RTPOLICY_ARTIFACT_DIR=/path/to/external/storage/rtpolicy

python scripts/fp16_convert.py \
  "$RTPOLICY_ARTIFACT_DIR/model.onnx" "$RTPOLICY_ARTIFACT_DIR/model-fp16.onnx"
python scripts/build_bench_x86.py build \
  "$RTPOLICY_ARTIFACT_DIR/model-fp16.onnx" "$RTPOLICY_ARTIFACT_DIR/model-fp16.engine" strongly_typed \
  --workspace-gib 1

python scripts/infer_tensorrt.py \
  --engine-path "$RTPOLICY_ARTIFACT_DIR/model-fp16.engine" \
  --inputs-dir "$RTPOLICY_ARTIFACT_DIR/onnx_calibration/sample_000/inputs" \
  --output-dir "$RTPOLICY_ARTIFACT_DIR/tensorrt_outputs"

python scripts/build_bench_x86.py bench \
  "$RTPOLICY_ARTIFACT_DIR/model-fp16.engine" \
  "$RTPOLICY_ARTIFACT_DIR/onnx_calibration/sample_000/inputs" 100
```

Conversion keeps normalization and other sensitive operators in FP32. The engine
follows the ONNX graph's precision and requires static input shapes.
The benchmark includes host transfers and allocation.

For RoboTwin2 inference, set these fields in the external
[evaluation YAML](configs/rtpolicy_eval/robotwin2.yaml), then run the launcher in
Section 4.2. Use the original FP32 export's metadata, whose public I/O is preserved
by mixed-FP16 conversion:

```yaml
backend: tensorrt
model_path: /path/to/artifacts/model-fp16.engine
metadata_path: /path/to/artifacts/model.onnx.metadata.json
```

<a id="acknowledgements"></a>

## 🤝 Acknowledgements

We thank the authors of [DINOv3](https://github.com/facebookresearch/dinov3)
and [VO-DP](https://github.com/D-Robotics-AI-Lab/DRRM) for sharing their research and code.

<a id="license"></a>

## 📄 License

See [LICENSE](LICENSE) for the repository terms. Third-party components retain their
respective licenses.
