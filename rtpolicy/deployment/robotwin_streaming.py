"""Opt-in inference adapter for RoboTwin2's original evaluator and policy hooks."""
import atexit
import json

from . import robotwin
from .streaming import StreamingPolicyRunner


class _ControlStepRunner(StreamingPolicyRunner):
    """Age references by executed policy actions while the simulator is paused."""

    def __init__(self, runner, **options):
        self._ticks = 0
        frequency = float(options.get('control_hz', 30.0))
        super().__init__(runner, clock=lambda: self._ticks / frequency, **options)

    def update_obs(self, observation, timestamp=None):
        self._ticks += 1
        super().update_obs(observation, timestamp)

    def reset_obs(self):
        super().reset_obs()
        self._ticks = 0


def get_model(args):
    options = dict(args.get('streaming', {}))
    left, right = args.get('left_arm_dim', 6), args.get('right_arm_dim', 6)
    options.setdefault('gripper_indices', (left, left + 1 + right))
    timebase = options.pop('timebase', 'control_steps')
    if timebase not in ('control_steps', 'wall'):
        raise ValueError('Streaming timebase must be control_steps or wall')
    cls = _ControlStepRunner if timebase == 'control_steps' else StreamingPolicyRunner
    model = cls(robotwin.get_model(args), **options)
    atexit.register(model.close)
    return model


def eval(task_env, model, observation):
    try:
        # Reuse the original encoder, take_action and post-action observation update.
        result = robotwin.eval(task_env, model, observation)
    except BaseException:
        model.pause()
        raise
    if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
        model.pause()  # Drain inference before the evaluator closes/reseeds an episode.
        print('[RTPolicy streaming] ' + json.dumps(model.statistics()), flush=True)
    return result


def reset_model(model):
    model.reset_obs()
