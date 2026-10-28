<div align="center">

<picture>
<source media="(prefers-color-scheme: dark)" srcset="assets/rtpolicy-wordmark-dark.svg">
<img src="assets/rtpolicy-wordmark.svg" width="360" alt="RTPolicy">
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

<p align="center">
<a href="https://github.com/D-Robotics-AI-Lab/RTPolicy"><img src="assets/badge-code.svg" height="24" alt="Code: GitHub"></a>
<a href="https://github.com/D-Robotics-AI-Lab/RTPolicy.project"><img src="assets/badge-project.svg" height="24" alt="Project page source"></a>
<a href="LICENSE"><img src="assets/badge-license.svg" height="24" alt="License: see LICENSE"></a>
</p>

<p align="center">
<a href="#1-introduction"><picture><source media="(prefers-reduced-motion: reduce)" srcset="assets/navigation/nav-introduction-compact-static.svg"><img src="assets/navigation/nav-introduction-compact.svg" width="165" height="46" alt="Introduction"></picture></a>
<a href="#2-model-overview"><picture><source media="(prefers-reduced-motion: reduce)" srcset="assets/navigation/nav-model-overview-compact-static.svg"><img src="assets/navigation/nav-model-overview-compact.svg" width="165" height="46" alt="Model overview"></picture></a>
<a href="#3-getting-started"><picture><source media="(prefers-reduced-motion: reduce)" srcset="assets/navigation/nav-get-started-compact-static.svg"><img src="assets/navigation/nav-get-started-compact.svg" width="165" height="46" alt="Get started"></picture></a>
<a href="#4-training-export-and-inference"><picture><source media="(prefers-reduced-motion: reduce)" srcset="assets/navigation/nav-train-deploy-compact-static.svg"><img src="assets/navigation/nav-train-deploy-compact.svg" width="165" height="46" alt="Train &amp; deploy"></picture></a>
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
| [`configs/`](configs/)                                                 | Training, observation, robot, and export configurations             |
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
| S600   | INT16     | [![Download S600 HTML](assets/download-s600.svg)](https://github.com/D-Robotics-AI-Lab/RTPolicy.project/releases/download/project-files/s600.html) |
| S100   | INT16     | [![Download S100 HTML](assets/download-s100.svg)](https://github.com/D-Robotics-AI-Lab/RTPolicy.project/releases/download/project-files/s100.html) |

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
python -m pip install -e .
```

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

<a id="4-training-export-and-inference"></a>

## ⚡ 4. Training, Export and Inference

### 4.1 Train a Policy

Follow the [Training Guide](docs/README.md) to train directly with LeRobot or use the optional Lance backend.

### 4.2 Evaluate in RoboTwin2.0

Pass an absolute checkpoint directory. The example selects four integration steps.

```bash
export CHECKPOINT_DIR=/absolute/path/to/checkpoint
RTPOLICY_NFE=4 bash scripts/eval/eval_robotwin2.0_parallel.sh \
  RTPolicy beat_block_hammer agilex_config "$CHECKPOINT_DIR" 100 2 42
```

JPEG input alignment is enabled by default through `RTPOLICY_JPEG_ROUNDTRIP=1`.

### 4.3 Export and Run ONNX

```bash
export RTPOLICY_ARTIFACT_DIR=/path/to/external/storage/rtpolicy

python -m rtpolicy.trainer.onnx \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --onnx-path "$RTPOLICY_ARTIFACT_DIR/model.onnx" \
  --device cpu --export-dtype fp32 --static-batch --image-preprocess-in-onnx --verify

python scripts/infer_onnx.py --help
python scripts/infer_rtpolicy.py --help
```

Install ONNX, ONNX Runtime, and the export dependencies in your deployment environment.
The exported graph follows the checkpoint configuration, including its sampling settings.

### 4.4 TensorRT and Edge Deployment

Use **FP16 for NVIDIA TensorRT** and **INT16 for S600 / S100 BPU deployment**.

Install compatible `tensorrt`, `cuda-python`, `onnx`, and `onnxconverter-common`
packages in your NVIDIA deployment environment:

```bash
python scripts/fp16_convert.py \
  "$RTPOLICY_ARTIFACT_DIR/model.onnx" "$RTPOLICY_ARTIFACT_DIR/model-fp16.onnx"
python scripts/build_bench_x86.py build \
  "$RTPOLICY_ARTIFACT_DIR/model-fp16.onnx" "$RTPOLICY_ARTIFACT_DIR/model-fp16.engine" strongly_typed \
  --workspace-gib 1
python scripts/build_bench_x86.py bench \
  "$RTPOLICY_ARTIFACT_DIR/model-fp16.engine" /path/to/inputs 100
```

Conversion keeps normalization and other sensitive operators in FP32. The engine
follows the ONNX graph's precision and requires static input shapes.
The benchmark includes host transfers and allocation.

<a id="acknowledgements"></a>

## 🤝 Acknowledgements

We thank the authors of [DINOv3](https://github.com/facebookresearch/dinov3)
and [VO-DP](https://github.com/D-Robotics-AI-Lab/DRRM) for sharing their research and code.

<a id="license"></a>

## 📄 License

See [LICENSE](LICENSE) for the repository terms. Third-party components retain their
respective licenses.
