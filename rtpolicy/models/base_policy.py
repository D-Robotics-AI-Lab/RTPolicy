from typing import Dict
import inspect

import torch
import torch.nn as nn

from rtpolicy.common.hub_mixin import CompatiblePyTorchModelHubMixin


class BasePolicy(
        nn.Module, 
        CompatiblePyTorchModelHubMixin
    ):

    n_action_steps: int
    n_obs_steps: int
    
    # ========= Train  ============
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("compute_loss is not implemented")
    
    # ========= Inference  ============
    def predict_action(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("predict_action is not implemented")
    
    def forward(self, obs: Dict[str, torch.Tensor], action: torch.Tensor = None, **kwargs) -> torch.Tensor:
        pretraining = getattr(self, 'pretraining', False)
        
        if self.training or pretraining:
            if 'batch' in inspect.signature(self.compute_loss).parameters:
                batch = {'obs': obs, 'action': action}
                return self.compute_loss(batch, **kwargs)
            return self.compute_loss(obs, action, **kwargs)
        return self.predict_action(obs, gt=action, **kwargs)
