"""Run exported policies with the same observation history and CUDA noise stream."""
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from rtpolicy.models.policy.rtpolicy.env_runner.base_runner import RTPolicyRunner
from scripts.infer_onnx import _create_session, _prepare_session_inputs, build_inputs


class CompiledPolicyRunner:
    stack_last_n_obs = RTPolicyRunner.stack_last_n_obs

    def __init__(self, model_path, backend, metadata_path=None, device='cuda'):
        self.backend = backend
        self.device = torch.device(device)
        with Path(metadata_path or f'{model_path}.metadata.json').open() as handle:
            self.metadata = json.load(handle)
        self.n_obs_steps = int(self.metadata['n_obs_steps'])
        self.input_time_steps = int(self.metadata['input_time_steps'])
        self.n_action_steps = int(self.metadata['n_action_steps'])
        self.action_key = self.metadata['action_key']
        self.state_key = self.metadata['state_key']
        self.num_inference_timesteps = int(self.metadata['num_inference_timesteps'])
        self.obs = deque(maxlen=self.input_time_steps + 1)
        if backend == 'onnx':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
            self.session = _create_session(str(model_path), self.metadata, providers=providers)
            if self.device.type == 'cuda' and 'CUDAExecutionProvider' not in self.session.get_providers():
                raise RuntimeError('CUDA ONNX evaluation requires an active CUDAExecutionProvider')
            print(f'[RTPolicy] ONNX providers: {self.session.get_providers()}, options: {self.session.get_provider_options()}')
        elif backend == 'tensorrt':
            from scripts.build_bench_x86 import Runner
            self.engine_runner = Runner(str(model_path))
        else:
            raise ValueError(f'Unknown compiled policy backend: {backend}')

    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, observation):
        self.obs.append(observation)

    def get_last_obs(self):
        return self.obs[-1]

    def get_action(self, observation=None):
        if observation is not None:
            self.update_obs(observation)
        if not self.obs:
            raise ValueError('No observation is recorded')
        images = {key: self.stack_last_n_obs([obs[key] for obs in self.obs], self.input_time_steps)
                  for key in self.metadata['rgb_keys']}
        source_key = 'state' if self.state_key == 'delta_state' else self.state_key
        state = self.stack_last_n_obs([obs[source_key] for obs in self.obs], self.input_time_steps)
        if self.state_key == 'delta_state':
            absolute_first = state[0:1].copy()
            state = state - absolute_first
            state[0:1] = absolute_first
        # The native policy samples FP32 noise on CUDA. NumPy's RNG would use
        # a different trajectory even with the same episode seed.
        noise = torch.randn(tuple(self.metadata['noise_trajectory_shape']),
                            dtype=torch.float32, device=self.device).cpu().numpy()
        if self.backend == 'onnx':
            inputs = _prepare_session_inputs(self.session, images, state, noise, self.metadata)
            values = self.session.run(None, inputs)
            outputs = {item.name: value for item, value in zip(self.session.get_outputs(), values)}
        else:
            outputs = self.engine_runner.run(build_inputs(images, state, noise, self.metadata))
        actions = np.asarray(outputs['action'])[0, :self.n_action_steps]
        if not np.isfinite(actions).all():
            raise RuntimeError(f'{self.backend} generated nonfinite actions')
        return actions, 'ee' if 'endpose' in self.action_key else None

    def close(self):
        if hasattr(self, 'engine_runner'):
            self.engine_runner.close()
