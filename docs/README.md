# Training Guide

RTPolicy trains directly on LeRobot datasets by default. Lance is optional.
Run commands from the repository root.

```bash
python -m pip install -e .
```

## Configure

Set the experiment fields in the existing configuration files:

- [Dataset](../configs/rtpolicy_train/dataset_config/demo_separated_dataset.yaml): `train_dataset.path` (dataset parent) and `train_dataset.task`.
- [Observation encoder](../configs/rtpolicy_train/observation_config/dinov3cnnl_3v.yaml): `model.obs_encoder.emvis_config.backbone_path` (pretrained weight directory).
- [Training recipe](../configs/rtpolicy_train/dit_fm_dinov3cnnl_3v.yaml): `output_dir` (a new experiment directory).

The dataset root is `train_dataset.path/train_dataset.task`.
Keep datasets, weights, and outputs on external storage.

The [three-view recipe](../configs/rtpolicy_train/dit_fm_dinov3cnnl_3v.yaml) uses
`left_cam`, `right_cam`, and `head_cam`, with 14-dimensional joint states and targets.
RoboTwin2 source images are **240 × 320 (H × W)**; preprocessing preserves their dimensions.

The default LeRobot reader uses the **v2.1 layout**: JSONL metadata and per-episode
Parquet files. Keep `train_dataset.dataset_metadata` as configured.

For raw RoboTwin2 HDF5 recordings, install the data tools and convert task folders:

```bash
python -m pip install -e '.[data]'
bash scripts/dataset/robotwin2lerobot.sh \
  /path/to/raw/tasks /path/to/external/lerobot beat_block_hammer
```

## Train

```bash
python main.py --config-path=configs/rtpolicy_train \
  --config-name=dit_fm_dinov3cnnl_3v
```

Keep training settings in the selected configuration. Append `--cfg job --resolve`
to inspect it without starting training.

## Optional: Lance

The converter reads complete **LeRobot v3.0** datasets. Prepare a separate v3.0
copy with upstream tools if your source uses an older layout.

```bash
python -m pip install -e '.[lance]'
python tools/lerobot_v30_to_lance.py \
  --root /path/to/lerobot_v3/task \
  --output /path/to/lance/task \
  --mode jpeg --jpeg-storage blob \
  --staging-dir /path/to/scratch
```

Use a fresh output directory. Staging needs atomic rename and enough space for
the complete dataset; use local scratch when required by the mount.

In the dataset configuration, set `train_dataset.backend: lance`, point
`train_dataset.path` to the converted dataset's parent, and set
`train_dataset.dataset_metadata: null`. Choose a new `output_dir`, then use the
same training command above.

## Outputs and resume

Checkpoints and final policy files are written to `output_dir`. To resume, keep
the same configuration and output directory and append
`resume_from_checkpoint=latest` to the training command.

For agent-managed setup and checkpoint export, see the [AI Setup Guide](AI_SETUP.md).
