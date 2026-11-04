"""Inference-side streaming over an existing NumPy-observation policy runner."""
from collections import deque
from copy import deepcopy
import math
import threading
import time

import numpy as np

from .trajectory import PiecewiseJerkStitcher, SplicedTrajectory


class StreamingPolicyRunner:
    """One inference worker, latest observation window and a time-indexed chunk.

    ``get_action(obs)`` returns one action for an existing execution loop.
    ``update_obs(obs)`` advances history after that action. A following
    ``get_action(obs)`` refreshes that same frame rather than duplicating it.
    All model calls belong to the worker after a synchronous first prediction.
    Timestamps must use the same clock as ``clock`` (monotonic by default).
    """

    def __init__(self, runner, inference_hz=10.0, control_hz=30.0,
                 switch_margin_s=0.03, trajectory_dt=0.02, gripper_indices=(),
                 *, clock=time.monotonic, sleep=time.sleep):
        values = (inference_hz, control_hz, switch_margin_s)
        if not all(math.isfinite(v) for v in values) or min(values[:2]) <= 0 or values[2] < 0:
            raise ValueError('Frequencies must be positive and switch margin nonnegative')
        self.runner, self.clock, self.sleep = runner, clock, sleep
        self._rate_clock = time.monotonic
        self.inference_period = 1.0 / inference_hz
        self.dt, self.margin = 1.0 / control_hz, switch_margin_s
        depth = max(runner.n_obs_steps, getattr(runner, 'n_image_obs_steps', 0),
                    getattr(runner, 'input_time_steps', 0))
        self.obs = deque(maxlen=depth)
        self._cv = threading.Condition()
        self._worker = None
        self._stitcher = PiecewiseJerkStitcher(trajectory_dt, gripper_indices)
        self.reset_obs()

    def _predict(self, history):
        # The control thread never mutates the backend runner's own deque.
        self.runner.reset_obs()
        for observation in history:
            self.runner.update_obs(observation)
        actions, action_type = self.runner.get_action()
        actions = np.asarray(actions)
        if actions.ndim != 2 or min(actions.shape) <= 0 or not np.isfinite(actions).all():
            raise ValueError('A prediction must be a finite, nonempty [time, action] array')
        return actions.copy(), action_type

    def _observe(self, observation, timestamp, replace=False):
        timestamp = self.clock() if timestamp is None else float(timestamp)
        if not math.isfinite(timestamp) or timestamp > self.clock() + self.dt:
            raise ValueError('Observation timestamp must use the runner clock')
        frame = (timestamp, deepcopy(observation))
        with self._cv:
            if self.obs and timestamp < self.obs[-1][0]:
                raise ValueError('Observation timestamps must be nondecreasing')
            if replace and self.obs:
                self.obs[-1] = frame
            else:
                self.obs.append(frame)
            self._version += 1
            self._cv.notify_all()

    def update_obs(self, observation, timestamp=None):
        self._observe(observation, timestamp)
        self._feedback = True

    def get_last_obs(self):
        with self._cv:
            return deepcopy(self.obs[-1][1])

    def _promote(self, now):
        while isinstance(self._active, SplicedTrajectory) and now >= self._active.tail.starts_at:
            self._active = self._active.tail

    def _stitch(self, actions, observed_at, finished):
        with self._cv:
            self._promote(finished)
            active = self._active
            measured = np.asarray(self.obs[-1][1]['state']).copy()
        switch = finished + self.margin
        started = self._rate_clock()
        try:
            plan = self._stitcher.stitch(actions, observed_at, switch, self.dt, active, measured)
        except RuntimeError:
            self._qp_rejected += 1
            return
        self._qp_latencies.append(self._rate_clock() - started)
        with self._cv:
            if self._stop:
                return
            if plan is None:
                self._stale += 1
            else:
                self._promote(self.clock())
                # Keep the old reference as the head until the future switch.
                self._active = SplicedTrajectory(self._active, plan)
                self._updates += 1

    def _run(self):
        while True:
            with self._cv:
                while not self._stop:
                    delay = self._next_inference - self._rate_clock()
                    if self._version != self._seen and delay <= 0:
                        history = tuple(frame[1] for frame in self.obs)
                        observed_at = self.obs[-1][0]
                        self._seen = self._version
                        self._next_inference = self._rate_clock() + self.inference_period
                        break
                    self._cv.wait(timeout=max(0.001, delay) if delay > 0 else None)
                if self._stop:
                    return
            started = self._rate_clock()
            try:
                actions, kind = self._predict(history)
                if kind != self._kind or actions.shape[1] != self._action_dim:
                    raise ValueError('Action type and dimension changed during streaming')
            except Exception as error:
                with self._cv:
                    self._error = error
                    self._cv.notify_all()
                return
            finished = self.clock()
            with self._cv:
                if self._stop:
                    return
                self._latencies.append(self._rate_clock() - started)
            try:
                self._stitch(actions, observed_at, finished)
            except Exception as error:
                with self._cv:
                    self._error = error
                    self._cv.notify_all()
                return
            with self._cv:
                self._cv.notify_all()

    def get_action(self, observation=None, timestamp=None):
        if observation is not None:
            self._observe(observation, timestamp, replace=self._feedback)
            self._feedback = False
        if not self.obs:
            raise ValueError('Submit an observation before requesting an action')
        if self._worker is None:
            started = self._rate_clock()
            actions, self._kind = self._predict(tuple(frame[1] for frame in self.obs))
            if self._kind not in (None, 'qpos'):
                raise ValueError('Piecewise-jerk streaming expects joint-position actions')
            self._action_dim = actions.shape[1]
            if any(i < 0 or i >= self._action_dim for i in self._stitcher.grippers):
                raise ValueError('Gripper index is outside the action vector')
            self._stitch(actions, self.obs[-1][0], self.clock())
            now = self.clock()
            self._started = self._rate_clock()
            self._next_tick = self._started + self.dt
            self._next_inference = started + self.inference_period
            self._seen = self._version
            self._worker = threading.Thread(target=self._run, name='rtpolicy-inference', daemon=True)
            self._worker.start()
        else:
            remaining = self._next_tick - self._rate_clock()
            if remaining > 0:
                self.sleep(remaining)
            now = self.clock()
            missed = max(0, int((self._rate_clock() - self._next_tick) / self.dt))
            self._late_ticks += missed
            self._next_tick += (missed + 1) * self.dt
        with self._cv:
            if self._error is not None:
                raise RuntimeError('Streaming inference failed') from self._error
            self._promote(now)
            state = self._active.sample(now) if self._active is not None else None
            if state is None:
                action = np.asarray(self.obs[-1][1]['state']).copy()
                self._holds += 1
            else:
                action = state[0].copy()
            for i in self._stitcher.grippers:
                action[i] = np.clip(action[i], 0.0, 1.0)
            self._commands += 1
            return action[None], self._kind

    def get_reference(self, timestamp=None):
        """Sample planned p/v/a for an independently scheduled robot controller."""
        now = self.clock() if timestamp is None else float(timestamp)
        with self._cv:
            if self._error is not None:
                raise RuntimeError('Streaming inference failed') from self._error
            self._promote(now)
            state = self._active.sample(now) if self._active is not None else None
            if state is None:
                position = np.asarray(self.obs[-1][1]['state']).copy()
                return position, np.zeros_like(position), np.zeros_like(position)
            return tuple(value.copy() for value in state)

    def pause(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._worker is not None:
            self._worker.join()

    def reset_obs(self):
        self.pause()
        with self._cv:
            self.obs.clear()
            self._worker = self._active = self._error = None
            self._stop = self._feedback = False
            self._version = self._seen = 0
            self._commands = self._holds = self._stale = self._updates = self._late_ticks = 0
            self._latencies = []
            self._qp_latencies = []
            self._qp_rejected = 0
            self._started = None
        self._stitcher.reset()
        self.runner.reset_obs()

    def statistics(self):
        elapsed = max(self._rate_clock() - self._started, 1e-9) if self._started is not None else 0.0
        return {'commands': self._commands, 'async_predictions': len(self._latencies),
                'accepted_updates': self._updates, 'stale_predictions': self._stale,
                'hold_commands': self._holds, 'missed_control_ticks': self._late_ticks,
                'mean_async_inference_ms': float(np.mean(self._latencies) * 1000) if self._latencies else None,
                'mean_stitch_ms': float(np.mean(self._qp_latencies) * 1000) if self._qp_latencies else None,
                'rejected_stitches': self._qp_rejected,
                'async_error': type(self._error).__name__ if self._error is not None else None,
                'observed_control_hz': max(0, self._commands - 1) / elapsed if elapsed else 0.0}

    def close(self):
        self.pause()
        if hasattr(self.runner, 'close'):
            self.runner.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
