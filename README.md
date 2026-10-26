# RTPolicy

RTPolicy 正式运行代码：训练 / RobotWin 仿真评测 / ONNX 导出与推理 / TensorRT 编译测速。

独立项目主页源码：[RTPolicy.project](https://github.com/D-Robotics-AI-Lab/RTPolicy.project)。当前为私有草稿，可按页面仓库说明在本地预览；主页与训练代码分开维护。

Python 包名仍是 `drrm`；策略类/目录已全部改名 RTPolicy（原 VODPP/VODPPlus）。原论文仓库：[DRRM-VODPP](https://github.com/D-Robotics-AI-Lab/DRRM-VODPP)。

**旧 checkpoint 兼容**：类名从 `VODPPlus*` 改成 `RTPolicy*` 后，存量 ckpt 的
`config.json`（pkg_map/auto_map）需要跑一次
`python tools/rename_ckpt_pkg_map.py --root <ckpt父目录>`（只改 config 字符串，不动权重）。

## 目录

| 路径 | 做什么 |
|---|---|
| `drrm/` | 策略、数据集、训练、ONNX 导出（`drrm.trainer.onnx`） |
| `configs/` | 训练 yaml + `configs/rtpolicy_onnx/` |
| `simulation/robotwin2.0/` | RobotWin 2.0 仿真与 `policy/RTPolicy/`（方法适配层仅保留 RTPolicy） |
| `scripts/eval/` | 仿真评测入口 |
| `scripts/infer_onnx.py` `scripts/infer_rtpolicy.py` | ONNX Runtime 推理 |
| `scripts/build_bench_x86.py` `scripts/bench_acc_jetson.py` | TensorRT 建引擎 + 测速 |
| `scripts/fp16_convert.py` `scripts/int8_convert.py` | ONNX 图精度转换 |
| `docs/` | 正式使用说明及性能优化笔记 |
| `pretrained/` | 软链到 EFS 上的 DINOv3 / VGGT 权重 |

仿真物体网格 **没有拷进来**（太大），沿用原仓库的软链：

- `simulation/robotwin2.0/assets/objects` → `/data/efs/users/ruilin.wang/RoboTwinAssets/objects`
- `simulation/robotwin2.0/assets/background_texture` → 同目录 `background_texture`

本机 embodiment（aloha-agilex / piper / …）已拷进来。缺 objects 时再跑一次 `simulation/robotwin2.0/script/_download_assets.sh`。

## 安装

```bash
git clone git@github.com:D-Robotics-AI-Lab/RTPolicy.git
cd RTPolicy
# 在已有 RobotWin/RTPolicy 环境中登记当前工作树，不重新解析 Torch/LeRobot 依赖
conda activate RTPolicy
python -m pip install -e '.[lance]' --no-deps
```

全新集群环境先按 `requirements_cluster.txt` 安装锁定依赖，再执行上述 editable 安装。Lance extra 固定 lancedb 0.33.0 / pylance 7.0.0，并要求 pyarrow 23–24。

仿真评测还需要 RobotWin 那套（Sapien / Vulkan / curobo），和原来 `RTPolicy` 相同。

## 仿真评测

```bash
cd /path/to/RTPolicy
./scripts/eval/eval_robotwin2.0_parallel.sh \
  RTPolicy <task_name> agilex_config <checkpoint_dir> 50 8
```

`checkpoint_dir` 可以是绝对路径。JPEG 对齐默认开（`RTPOLICY_JPEG_ROUNDTRIP=1`），关：`RTPOLICY_JPEG_ROUNDTRIP=0`。

## ONNX

导出（图内预处理 + verify）：

```bash
python -m drrm.trainer.onnx \
  --checkpoint-dir <ckpt，含 config.json + model.safetensors> \
  --onnx-path <out>/model.onnx \
  --device cpu --export-dtype fp32 \
  --image-preprocess-in-onnx --verify
```

单帧推理：`python scripts/infer_onnx.py --onnx-path model.onnx ...`

历史导出验证记录见 [EFS实验归档](/data/efs/datasets/xianbao01.hou/rtpolicy_archives/20261016_071211/README.md)。

## TensorRT

x86（4090 / L4，TRT 11）：

```bash
python scripts/build_bench_x86.py build <model.onnx> <model.engine>
python scripts/build_bench_x86.py bench <model.engine> <inputs_dir>
```

Jetson 精度/测速：`scripts/fp16_convert.py`、`scripts/int8_convert.py`、`scripts/bench_acc_jetson.py`。

其他模型及PTQ历史实验已统一移至 [EFS实验归档](/data/efs/datasets/xianbao01.hou/rtpolicy_archives/20261016_071211/README.md)。

## 训练（可选）

Lance/EFS 训练的优化思路、验证结果及原生640×480配置见[性能优化笔记](docs/performance_optimization.md)。

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
  main.py --config-path=configs/rtpolicy_train --config-name=<yaml>
```

`WANDB_API_KEY` 从环境变量读，脚本里不再写死。

## 结果与实验存储

实验代码、单测、日志和历史结果存放在 [EFS实验归档](/data/efs/datasets/xianbao01.hou/rtpolicy_archives/20261016_071211/README.md)。本目录的 `outputs/`、`checkpoints/`、`results/`、`wandb/` 及仿真数据/评测结果目录均软链到 `/data/efs/datasets/xianbao01.hou/rtpolicy_runs/runtime/`，运行产物写入EFS。
