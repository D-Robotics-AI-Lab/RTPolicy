# Data Preparation and Lance Training

This guide uses one workflow: **LeRobot v3 → JPEG-blob Lance → RTPolicy training**.
The example is a single RoboTwin2 task with three RGB cameras and 14-dimensional
joint control. Run the commands from the repository root.

## 1. Set up the environment and weights

Use Python 3.10 and a CUDA-compatible PyTorch installation. The reference environment
uses PyTorch 2.7.1, torchvision 0.22.1, and transformers 4.56.0.

Set the paths below for your machine. `/mnt/efs` represents your external storage
mount; datasets, weights, and training outputs stay outside the repository.

```bash
export RTPOLICY_ROOT=/path/to/RTPolicy
export RTPOLICY_STORAGE_ROOT=/mnt/efs/rtpolicy
export TASK=beat_block_hammer
export LEROBOT_ROOT="$RTPOLICY_STORAGE_ROOT/datasets/lerobot_v3/$TASK"
export LANCE_ROOT="$RTPOLICY_STORAGE_ROOT/datasets/lance/$TASK"
export DINOV3_ROOT="$RTPOLICY_STORAGE_ROOT/pretrained/dinov3-convnext-large-pretrain-lvd1689m"
export RUN_DIR="$RTPOLICY_STORAGE_ROOT/runs/${TASK}_rtpolicy_3v"
export STAGING_DIR=/path/to/local/scratch/rtpolicy-lance

cd "$RTPOLICY_ROOT"
python -m pip install -e '.[lance]'
```

Download the complete [DINOv3 ConvNeXt-Large pretrained directory](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m)
to `$DINOV3_ROOT`, including its model configuration, image processor configuration,
and weights. The training command below points to this directory explicitly.

## 2. Prepare the LeRobot v3 input

Place a complete, local **LeRobot v3.0** dataset at `$LEROBOT_ROOT`:

```text
$LEROBOT_ROOT/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/…/*.parquet
├── data/…/*.parquet
└── referenced image or video files, if stored separately
```

`meta/info.json` must declare `codebase_version: "v3.0"`, a positive frame rate,
and the correct frame and episode counts. Episode metadata must cover all frames
in contiguous, nonempty ranges. Preserve every file referenced by the metadata.
Convert or upgrade older datasets to v3.0 before continuing.

The example expects these feature names:

| Feature                             | Content                                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| `left_cam`, `right_cam`, `head_cam` | RGB images, declared as LeRobot `image` or `video` features                              |
| `state`                             | 14 float32 joint values per frame, used for observations and the joint prediction target |
| Frame identifiers                   | `episode_index`, `frame_index`, `index`, `timestamp`, `task_index`                       |

Provide matching feature statistics in `meta/stats.json`. The selected robot
configuration uses 14-dimensional `state` actions and resizes camera inputs to
240 × 320. For a different robot or feature layout, update the robot and observation
configurations before training.

## 3. Convert to JPEG-blob Lance

```bash
python tools/lerobot_v30_to_lance.py \
  --root "$LEROBOT_ROOT" \
  --output "$LANCE_ROOT" \
  --table-name frames \
  --mode jpeg --jpeg-storage blob \
  --staging-dir "$STAGING_DIR"
```

`$STAGING_DIR` must be outside the source and output directories, support atomic
rename, and have room for the complete converted dataset; use local scratch when
your external mount cannot provide that operation.

The converter preserves the source and creates `$LANCE_ROOT/frames.lance`, a
copied `meta/` directory, and `lance_format.json`. It verifies the converted frame
count before copying the result to the output directory. Choose a fresh output
directory for each conversion. To validate metadata and paths first, append
`--dry-run` to the same command.

## 4. Train RTPolicy on Lance

Run this single-GPU starter example after conversion:

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
  output_dir="$RUN_DIR" \
  hydra.run.dir="$RUN_DIR/hydra"
```

The `lance` backend reads the converted directory directly;
`dataset_metadata=null` prevents Hydra from constructing LeRobot metadata for it.
`train_dataset.demo=null` uses all available training episodes after the validation
split; set it to an integer to cap that split.

To check the configuration without starting training, append `--cfg job --resolve`
to the same command. For multiple GPUs, launch `main.py` with Accelerate and use
a configuration matching your GPU count; the supplied
[`configs/accelerate_config.yaml`](../configs/accelerate_config.yaml) targets eight GPUs.

## 5. Find outputs and resume

`$RUN_DIR` contains the final `config.json` and `model.safetensors`, an `ema/`
weights directory, and periodic `checkpoint-<step>/` training states. Periodic
checkpoints are saved every `checkpointing_period` updates; the default is 10,000.
Choose a new `$RUN_DIR` for a new experiment.

To continue an interrupted run, repeat the training command with the same dataset,
model settings, and `$RUN_DIR`, appending:

```bash
resume_from_checkpoint=latest
```

This restores the most recent periodic checkpoint under `$RUN_DIR`. Keep its
complete training state for resuming; the final model weights alone are intended
for inference. If no periodic checkpoint exists, `latest` starts a new run.
