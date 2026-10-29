import unittest
import torch

from rtpolicy.models import RTPolicyDitFlowMatching, RTPolicyDitFlowMatchingConfig
from rtpolicy.trainer.onnx import (PolicyExportWrapper, _disable_export_only_features,
                                   _disable_scene_encoder_preprocess)


class ExportInputTests(unittest.TestCase):
    def test_conditioning_preserves_caller_owned_inputs(self):
        torch.set_num_threads(2)
        config = RTPolicyDitFlowMatchingConfig(
            shape_meta={'obs': {'head_cam': {'shape': [3,64,64], 'type':'rgb'},
                                'state': {'shape':[2], 'type':'low_dim'}},
                        'action': {'state': {'shape':[2]}}},
            noise_scheduler={'num_train_timesteps':1000, 'num_inference_timesteps':2,
                             'noise_beta_alpha':1.5, 'noise_beta_beta':1.0, 'noise_s':0.999},
            obs_encoder={'rgb_key':['head_cam'], 'emvis_config': {
                'dim_2d':64, 'backbone_target':'dinov3cnn', 'load_backbone_pretrain':False,
                'backbone_config':{'depths':[1,1,1,1], 'hidden_sizes':[8,16,32,64]},
                'processor_config':{'load_pretrain':False, 'do_resize':False, 'do_rescale':False},
                'only_2d':True, 'model_adapter_config':{'target':'ResNetAdapter',
                    'shape_out':[2,2], 'dim_out':64, 'conv_num':2}, 'mv_fuser_config':None}},
            hidden_size=64, depth=1, num_heads=4, horizon=6, n_obs_steps=3,
            n_image_obs_steps=1, n_action_steps=3, action_key='state',
            scene_adaptor='mlp2x_gelu', state_adaptor='mlp3x_gelu')
        policy = RTPolicyDitFlowMatching(config).to(dtype=torch.float32).eval()
        policy.normalizer.fit({'state':torch.randn(16,2)}, last_n_dims=1)
        _disable_export_only_features(policy, export_dtype=torch.float32)
        _disable_scene_encoder_preprocess(policy)
        wrapper = PolicyExportWrapper(policy, image_preprocess_in_onnx=True).eval()
        inputs = {'head_cam':torch.rand(1,3,3,64,64), 'state':torch.randn(1,3,2),
                  'noise_trajectory':torch.randn(1,6,2)}
        snapshots = {key:value.clone() for key,value in inputs.items()}
        with torch.no_grad():
            action, full = wrapper(**inputs)
        self.assertEqual(tuple(action.shape), (1,3,2))
        self.assertTrue(torch.isfinite(full).all())
        for key, value in inputs.items():
            self.assertTrue(torch.equal(value, snapshots[key]), key)


if __name__ == '__main__':
    unittest.main()
