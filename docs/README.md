# Training Guide

**Train directly on LeRobot datasets by default.** Lance is an optional storage
backend, covered separately below. Both examples use RTPolicy with three RoboTwin2
RGB cameras and 14-dimensional joint control.

## 1. Environment and weights

Use Python 3.10 and a CUDA-compatible PyTorch installation. The reference environment
uses PyTorch 2.7.1, torchvision 0.22.1, LeRobot 0.3.3, and transformers 4.56.0.

Set these paths for your machine. `/mnt/efs` represents your external storage
mount; keep datasets, weights, and training outputs outside the repository.

```bash
export RTPOLICY_ROOT=/path/to/RTPolicy
export RTPOLICY_STORAGE_ROOT=/mnt/efs/rtpolicy
export TASK=beat_block_hammer
export LEROBOT_ROOT="$RTPOLICY_STORAGE_ROOT/datasets/lerobot_v21/$TASK"
export DINOV3_ROOT="$RTPOLICY_STORAGE_ROOT/pretrained/dinov3-convnext-large-pretrain-lvd1689m"
export RUN_DIR="$RTPOLICY_STORAGE_ROOT/runs/${TASK}_rtpolicy_3v"

cd "$RTPOLICY_ROOT"
python -m pip install -e .
```

Download the complete [DINOv3 ConvNeXt-Large pretrained directory](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m)
to `$DINOV3_ROOT`, including the model configuration, image processor configuration,
and weights. The examples explicitly select this directory.

## 2. Prepare a LeRobot dataset

The pinned **LeRobot 0.3.3** reader targets the **v2.1 dataset layout**. Place a
complete compatible dataset at `$LEROBOT_ROOT`:

```text
$LEROBOT_ROOT/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
├── data/chunk-000/episode_000000.parquet
└── any referenced camera files
```

Preserve the paths declared in `meta/info.json`, all episodes and their statistics,
and any referenced camera files. The three-view robot configuration expects:

| Feature                             | Content                                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| `left_cam`, `right_cam`, `head_cam` | RGB camera observations                                                                  |
| `state`                             | 14 float32 joint values per frame, used for observations and the joint prediction target |
| Frame identifiers                   | `episode_index`, `frame_index`, `index`, `timestamp`, `task_index`                       |

Camera inputs are resized to 240 × 320. Update the robot and observation
configurations if your dataset uses a different feature layout.

Native LeRobot v3 datasets use a different metadata layout and cannot be read
directly by this pinned reader. If your input is v3.0, the optional Lance branch
in section 4 provides a supported route without changing the training environment.

## 3. Train directly on LeRobot

Run this single-GPU starter example; **no Lance installation or conversion is
required**:

```bash
python main.py \
  --config-path=configs/rtpolicy_train \
  --config-name=dit_fm_dinov3cnnl_3v \
  train_dataset.backend=lerobot \
  train_dataset.root="$LEROBOT_ROOT" \
  train_dataset.task="$TASK" \
  train_dataset.demo=null \
  model.obs_encoder.emvis_config.backbone_path="$DINOV3_ROOT" \
  train_batch_size=8 val_batch_size=8 dataloader_num_workers=4 \
  report_to=null \
  output_dir="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR/hydra"
```

Keep the configuration's `train_dataset.dataset_metadata` entry: this backend
requires its `LeRobotDatasetMetadata` instance. `train_dataset.demo=null` uses all
available training episodes after the validation split; use an integer to cap that
split. The Python package is `rtpolicy`, with `RTPolicy*` policy classes.

To inspect either training configuration without starting training, append
`--cfg job --resolve` to its command. For multiple GPUs, launch `main.py` with
Accelerate and a configuration matching your GPU count; the supplied
[`configs/accelerate_config.yaml`](../configs/accelerate_config.yaml) targets eight GPUs.

## 4. Optional: train on Lance

Skip this section when using LeRobot directly. The optional converter accepts a
complete **LeRobot v3.0** source, with `meta/info.json`, `meta/stats.json`,
`meta/tasks.parquet`, `meta/episodes/**/*.parquet`, data Parquet shards, and all
referenced camera files. Older sources must first be upgraded to a separate v3.0
copy using upstream data-preparation tools; this converter does not upgrade v2.1.

```bash
export LEROBOT_V3_ROOT="$RTPOLICY_STORAGE_ROOT/datasets/lerobot_v3/$TASK"
export LANCE_ROOT="$RTPOLICY_STORAGE_ROOT/datasets/lance/$TASK"
export LANCE_RUN_DIR="$RTPOLICY_STORAGE_ROOT/runs/${TASK}_rtpolicy_3v_lance"
export STAGING_DIR=/path/to/local/scratch/rtpolicy-lance

python -m pip install -e '.[lance]'
python tools/lerobot_v30_to_lance.py \
  --root "$LEROBOT_V3_ROOT" \
  --output "$LANCE_ROOT" \
  --table-name frames \
  --mode jpeg --jpeg-storage blob \
  --staging-dir "$STAGING_DIR"
```

Use a fresh output directory. The converter preserves the source and writes
`frames.lance`, the copied `meta/` directory, and `lance_format.json` under
`$LANCE_ROOT`. Append `--dry-run` to validate metadata and paths without converting.
`$STAGING_DIR` must be outside the source and output directories, support atomic
rename, and hold the complete converted dataset; use local scratch if required
by your external mount.

Then train with the Lance backend:

```bash
python main.py \
  --config-path=configs/rtpolicy_train \
  --config-name=dit_fm_dinov3cnnl_3v \
  train_dataset.backend=lance \
  train_dataset.dataset_metadata=null \
  train_dataset.root="$LANCE_ROOT" \
  train_dataset.task="$TASK" \
  train_dataset.demo=null \
  model.obs_encoder.emvis_config.backbone_path="$DINOV3_ROOT" \
  train_batch_size=8 val_batch_size=8 dataloader_num_workers=4 \
  report_to=null \
  output_dir="$LANCE_RUN_DIR" \
  hydra.run.dir="$LANCE_RUN_DIR/hydra"
```

Only this backend uses `dataset_metadata=null`: it reads the converted metadata
itself, without constructing the pinned LeRobot metadata reader.

## 5. Outputs and resume

The selected output directory (`$RUN_DIR` or `$LANCE_RUN_DIR`) contains the final
`config.json` and `model.safetensors`, an `ema/` weights directory, and periodic
`checkpoint-<step>/` training states. The default `checkpointing_period` is 10,000
updates. Choose a fresh output directory for each new experiment.

To continue an interrupted run, repeat its training command with the same dataset,
model settings, and output directory, appending:

```bash
resume_from_checkpoint=latest
```

This restores the latest periodic checkpoint in that output directory. Keep its
complete training state for resuming; final model weights alone are intended for
inference. If no periodic checkpoint exists, `latest` starts a new run.
