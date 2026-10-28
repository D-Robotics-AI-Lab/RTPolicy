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

<video src="https://github.com/user-attachments/assets/bec0058a-a3c2-423e-a642-4f3794a3e0c9" controls width="100%"></video>

**Multi-view perception. Compact visual tokens. Four-step action generation.**

</div>

RTPolicy turns multiple RGB views and robot state into action chunks with a compact
flow-matching policy. It combines multi-view reasoning with a deployment path for
edge devices, reaching **32.9 Hz on S600 in INT16** in the paper's latency evaluation.

<p align="center">
<a href="#1-introduction"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/navigation/nav-01-introduction-dark.svg"><img src="assets/navigation/nav-01-introduction-light.svg" width="165" height="44" alt="Introduction"></picture></a>
<a href="#2-model-overview"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/navigation/nav-02-model-overview-dark.svg"><img src="assets/navigation/nav-02-model-overview-light.svg" width="165" height="44" alt="Model overview"></picture></a>
<a href="#3-getting-started"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/navigation/nav-03-get-started-dark.svg"><img src="assets/navigation/nav-03-get-started-light.svg" width="165" height="44" alt="Get started"></picture></a>
<a href="#4-training-export-and-inference"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/navigation/nav-04-train-deploy-dark.svg"><img src="assets/navigation/nav-04-train-deploy-light.svg" width="165" height="44" alt="Train & deploy"></picture></a>
</p>

## 🔥 Updates

RTPolicy code and demonstration are available.

## 1. Introduction

### 1.1 Brief Introduction of RTPolicy

<div align="center">
<img src="assets/architecture.webp" width="100%" alt="RTPolicy architecture: frozen DINOv3 ConvNeXt, view-aware feature fusion, visual compression, and a flow-matching DiT conditioned on robot state">
</div>

RTPolicy fuses dense visual features across cameras before compressing them, retaining
cross-view information in a small representation for the action model.

- **Reason across views.** A frozen DINOv3 ConvNeXt backbone and view-aware feature fusion
  combine complementary observations from three RGB cameras.
- **Compress before action generation.** At 480 × 640 inputs, visual compression reduces
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

## 2. Model Overview

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

### 2.2 Latency on GPU and Jetson

| Platform            | Precision |   Latency | Frequency |
| ------------------- | --------- | --------: | --------: |
| RTX 4090            | FP32      |  23.80 ms |   42.0 Hz |
| RTX 4090            | FP16      |  10.92 ms |   91.6 Hz |
| Jetson Orin NX 16GB | FP32      | 266.30 ms |    3.8 Hz |
| Jetson Orin NX 16GB | FP16      | 143.20 ms |    7.0 Hz |

### 2.3 Latency on S600 and S100

| Platform | Precision |      Latency |   Frequency |
| -------- | --------- | -----------: | ----------: |
| **S600** | **INT16** | **30.35 ms** | **32.9 Hz** |
| S600     | INT8      |     18.59 ms |     53.8 Hz |
| S100     | INT16     |    149.72 ms |      6.7 Hz |
| S100     | INT8      |     86.56 ms |     11.6 Hz |

#### Quantization HTML Reports

| Device | Precision | File                                                                                                                                               |
| ------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| S600   | INT16     | [![Download S600 HTML](assets/download-s600.svg)](https://github.com/D-Robotics-AI-Lab/RTPolicy.project/releases/download/project-files/s600.html) |
| S100   | INT16     | [![Download S100 HTML](assets/download-s100.svg)](https://github.com/D-Robotics-AI-Lab/RTPolicy.project/releases/download/project-files/s100.html) |

### 2.4 Model Checkpoints

| Model       | Google Drive |
| ----------- | ------------ |
| RoboTwin2.0 |              |

## 3. Getting Started

### 3.1 Installation

Use Python 3.10 and a hardware-compatible PyTorch installation. The reference
environment uses PyTorch 2.7.1, torchvision 0.22.1, LeRobot 0.3.3, and transformers 4.56.0.

```bash
git clone https://github.com/D-Robotics-AI-Lab/RTPolicy.git
cd RTPolicy
python -m pip install -e .
```

For an existing compatible environment, append `--no-deps` to the editable install.
The locked cluster dependencies are provided in [`requirements_cluster.txt`](requirements_cluster.txt).

### 3.2 Prepare Data and Weights

Supply compatible DINOv3 weights and set
`model.obs_encoder.emvis_config.backbone_path` to their directory. Datasets,
checkpoints, simulator assets, and generated outputs are kept outside Git.

The [training guide](docs/README.md) covers LeRobot v3 input preparation, JPEG-blob conversion, and Lance training.

For RoboTwin2.0 simulation, prepare SAPIEN, Vulkan, cuRobo, and the official assets:

```bash
cd simulation/robotwin2.0
bash script/_download_assets.sh
cd ../..
```

## 4. Training, Export and Inference

### 4.1 Train a Policy

Follow the [Lance training guide](docs/README.md#4-train-rtpolicy-on-lance) for dataset preparation and the complete training command.

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
python -m rtpolicy.trainer.onnx \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --onnx-path "$RTPOLICY_ARTIFACT_DIR/model.onnx" \
  --device cpu --export-dtype fp32 --image-preprocess-in-onnx --verify

python scripts/infer_onnx.py --help
python scripts/infer_rtpolicy.py --help
```

Install ONNX, ONNX Runtime, and the export dependencies in your deployment environment.
The exported graph follows the checkpoint configuration, including its sampling settings.

### 4.4 TensorRT and Edge Deployment

For compatible NVIDIA TensorRT / CUDA environments:

```bash
python scripts/build_bench_x86.py build \
  "$RTPOLICY_ARTIFACT_DIR/model.onnx" "$RTPOLICY_ARTIFACT_DIR/model.engine"
python scripts/build_bench_x86.py bench \
  "$RTPOLICY_ARTIFACT_DIR/model.engine" /path/to/inputs
```

Jetson conversion and benchmarking helpers are provided in
[`scripts/fp16_convert.py`](scripts/fp16_convert.py),
[`scripts/int8_convert.py`](scripts/int8_convert.py), and
[`scripts/bench_acc_jetson.py`](scripts/bench_acc_jetson.py).
S600 / S100 toolchain instructions and deployment packages will be added separately.

## Acknowledgements

We thank the authors of [DINOv3](https://github.com/facebookresearch/dinov3)
and [VO-DP](https://github.com/D-Robotics-AI-Lab/DRRM) for sharing their research and code.

## Citation

RTPolicy BibTeX and publication details will be added with the paper release.

## License

See [LICENSE](LICENSE) for the repository terms. Third-party components retain their
respective licenses.
