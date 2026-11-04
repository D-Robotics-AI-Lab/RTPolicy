# Training and Inference Guide

RTPolicy trains directly on LeRobot datasets by default. Lance is optional.
Run commands from the repository root.

```bash
python -m pip install -r requirements.txt
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

For raw RoboTwin2 HDF5 recordings, convert task folders:

```bash
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

## Pretrain and fine-tune

These RoboTwin2 recipes use DINOv3 ConvNeXt-Base with three cameras, a batch size
of 256, and seed 0. Both use `horizon: 32`, `n_obs_steps: 3`, and
`n_image_obs_steps: 1`: three state observations and the latest image from each
camera. The action and state dimensions are 14; source images remain 240 × 320.

| Stage | Recipe | Data | Updates | Learning rate |
| --- | --- | --- | ---: | ---: |
| Pretrain | [pt_dinov3cnnb_3v_lance_300k](../configs/rtpolicy_train/pt_dinov3cnnb_3v_lance_300k.yaml) | Joint-task Lance, 2500 demonstrations | 300K | 2e-4 |
| Fine-tune | [ft_dinov3cnnb_3v_30k](../configs/rtpolicy_train/ft_dinov3cnnb_3v_30k.yaml) | Single-task LeRobot v2.1, 50 demonstrations | 30K | 2e-5 |

In each recipe, set `train_dataset.path` to the dataset parent,
`model.obs_encoder.emvis_config.backbone_path` to the external DINOv3
ConvNeXt-Base weights, and `output_dir` to a new external run directory.

For pretraining, prepare the converted Lance dataset at
`train_dataset.path/joint_finetune_2500`; see the Lance conversion instructions
below. The recipe uses all 2500 demonstrations without a validation split.

```bash
python -m pip install -e '.[lance]'
python main.py --config-path=configs/rtpolicy_train \
  --config-name=pt_dinov3cnnb_3v_lance_300k
```

For fine-tuning, set `train_dataset.task` (for example, `finetune_lift_pot`) and
`checkpoint_dir` to the final pretrained policy directory containing
`model.safetensors`. The recipe initializes policy weights and keeps the
pretraining normalizer while starting a new optimizer and learning-rate schedule.
Leave `resume_from_checkpoint: null` when starting fine-tuning.

```bash
python main.py --config-path=configs/rtpolicy_train \
  --config-name=ft_dinov3cnnb_3v_30k
```

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

## Streaming inference

The deployment pipeline separates camera acquisition, policy prediction,
trajectory stitching and robot control. Prediction consumes the latest complete
observation window at 10 Hz while execution continues along the active reference.
Each predicted chunk is timestamped with its observation capture time, with
waypoints spaced at 1/30 s. The low-level robot controller samples the continuous
reference at its own servo frequency.

1. **Observe.** Keep timestamped RGB and joint-state observations. History lengths
   come from the checkpoint; the released lift_pot policy uses three states and
   the latest image from each camera.
2. **Predict.** One inference worker reads a snapshot of the newest window.
   It never queues old windows or runs concurrent model forwards.
3. **Stitch.** Discard the elapsed chunk prefix and choose a switch time 30 ms
   ahead. A piecewise-constant-jerk QP tracks the remaining targets, penalizes
   normalized acceleration and jerk, and bounds velocity, acceleration and jerk.
   Its initial position, velocity and acceleration match the active reference at
   the switch, preserving C2 continuity when the trajectories overlap. Arm and
   gripper channels use separate limits. The old prefix remains active until
   the switch; expired updates are discarded. A failed optimization keeps the
   active reference, and the runner holds measured joints if no reference is valid.
4. **Execute.** Sample the stitched reference while the next prediction runs.
   Episode reset drains the worker and clears observation and trajectory history.

[`StreamingPolicyRunner`](../rtpolicy/deployment/streaming.py) wraps the existing
PyTorch, ONNX or TensorRT runner. For a real-robot integration, use monotonic
capture timestamps from the same clock as the runner. Feed encoded observations
through `update_obs(observation, timestamp=capture_time)` and call `get_action()`
to start and drive streaming. An independently scheduled servo can sample
`get_reference()` for joint position, velocity and acceleration. The application
retains responsibility for its robot transport and controller lifecycle.

### RoboTwin2 example

Set `ckpt_setting` to the checkpoint directory in
[`robotwin2_streaming.yaml`](../configs/rtpolicy_eval/robotwin2_streaming.yaml).
The default example uses lift_pot, NFE4, seed0, 100 episodes and one worker.
Use the same task settings and checkpoint when comparing with blocking inference.

```bash
python scripts/eval/run_robotwin2.py \
  --config configs/rtpolicy_eval/robotwin2_streaming.yaml \
  --runtime-dir /path/to/external/streaming-evaluation
```

Only the policy namespace changes to `rtpolicy.deployment.robotwin_streaming`.
The adapter delegates to the existing RoboTwin2 policy hook and emits one joint
target per call. Observation encoding, JPEG alignment, `take_action`, expert
preflight, seed selection and task success checks are reused unchanged.

RoboTwin2 executes a complete, variable-duration TOPP segment per `take_action`.
The default `timebase: control_steps` therefore advances the policy trajectory
by 1/30 s after each executed action, instead of aging it while simulation is
paused. Prediction is scheduled at up to 10 Hz in wall time and command delivery
at up to 30 Hz. This is a simulation adapter for the streaming design; the
real-robot interface uses continuous wall time and an independent servo loop.
The QP knot interval is 20 ms in both interfaces.

At each completed episode the adapter prints `[RTPolicy streaming]` JSON with
prediction, accepted/stale update and hold counts, plus inference and stitching
latencies. The original evaluator writes the task success results under the
external runtime directory.

### Single-task validation

One streaming run used the released lift_pot checkpoint, PyTorch BF16, NFE4,
seed0, JPEG q95 alignment, agilex_config, 100 episodes and one evaluator worker
on NVIDIA L4. The original checkpoint files and RoboTwin2 source were unchanged.

| Inference mode | lift_pot successes |
| --- | ---: |
| Blocking PyTorch BF16, previously measured | 93/100 |
| Streaming PyTorch BF16, one new run | 94/100 |

The original expert preflight and seed scan selected the scenes; the two runs
share 99 accepted seeds. Wall-time scheduling can affect asynchronous rollouts.
These single-task checks are separate from the paper's 50-task benchmark.
The L4 simulation run averaged about three policy commands per second with
the original TOPP execution and rendering; its reference grid remains 30 Hz.

The run completed with no inference-thread exceptions. It recorded 9,788
asynchronous predictions and 123 rejected trajectory optimizations, which
retained the active reference. The tested inference implementation is
[`91c7e0b`](https://github.com/D-Robotics-AI-Lab/RTPolicy/commit/91c7e0baa95753be3db45916a29b22baad05cac6).
