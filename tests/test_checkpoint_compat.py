import copy
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtpolicy.common.checkpoint import normalize_checkpoint_metadata, load_checkpoint_weights


class CheckpointCompatibilityTests(unittest.TestCase):
    def model(self):
        model = torch.nn.Module()
        model.obs_encoder = torch.nn.Module()
        model.obs_encoder.scene_encoder = torch.nn.Module()
        model.obs_encoder.scene_encoder.backbone = torch.nn.Linear(2, 3)
        model.obs_encoder.scene_encoder.model_adapter = torch.nn.Linear(3, 3)
        model.head = torch.nn.Linear(3, 1)
        return model

    def test_legacy_metadata_is_normalized_without_mutating_source(self):
        original = {
            'pkg_map': {'AutoModel': 'drrm.models.VODPPlusDitFlowMatching'},
            'dataset_id': 'drrm_robotwin2',
            'obs_encoder': {'emvis_config': {
                'vggt_target': 'dinov3cnn', 'vggt_config': {'hidden_sizes': [8, 16]},
                'vggt_model_path': '/external/pretrained/dinov3-cnn-base',
                'load_vggt_pretrain': False,
            }},
        }
        snapshot = copy.deepcopy(original)
        result = normalize_checkpoint_metadata(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(result['dataset_id'], original['dataset_id'])
        self.assertEqual(result['pkg_map']['AutoModel'], 'rtpolicy.models.RTPolicyDitFlowMatching')
        emvis = result['obs_encoder']['emvis_config']
        self.assertEqual(emvis['backbone_config'], {'hidden_sizes': [8, 16]})
        self.assertEqual(emvis['backbone_path'], '/external/pretrained/dinov3-cnn-base')
        self.assertFalse(emvis['load_backbone_pretrain'])
        self.assertNotIn('vggt_target', emvis)
        self.assertEqual(normalize_checkpoint_metadata(result), result)

    def test_conflicting_backbone_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting checkpoint fields'):
            normalize_checkpoint_metadata({'obs_encoder': {'emvis_config': {
                'vggt_target': 'dinov3cnn', 'backbone_target': 'different',
            }}})

    def test_current_and_legacy_weights_load_completely(self):
        source = self.model()
        with tempfile.TemporaryDirectory(prefix='rtpolicy_checkpoint_compat_') as tmp:
            for legacy in [False, True]:
                weights = {(key.replace('.backbone.', '.vggt_encoder.')
                            .replace('.model_adapter.', '.model_adapter.adapter.')) if legacy else key: value
                           for key, value in source.state_dict().items()}
                path = Path(tmp) / ('legacy.safetensors' if legacy else 'current.safetensors')
                save_file(weights, str(path))
                before = path.read_bytes()
                target = self.model()
                missing, unexpected = load_checkpoint_weights(target, path, device='cpu')
                self.assertFalse(missing)
                self.assertFalse(unexpected)
                for key, value in source.state_dict().items():
                    self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
                self.assertEqual(target._checkpoint_load_report['renamed_backbone_tensors'], 2 if legacy else 0)
                self.assertEqual(target._checkpoint_load_report['renamed_adapter_tensors'], 2 if legacy else 0)
                self.assertEqual(path.read_bytes(), before)

    def test_incomplete_weights_are_rejected(self):
        weights = self.model().state_dict()
        weights.pop('head.weight')
        with tempfile.TemporaryDirectory(prefix='rtpolicy_checkpoint_missing_') as tmp:
            path = Path(tmp) / 'incomplete.safetensors'
            save_file(weights, str(path))
            with self.assertRaises(RuntimeError):
                load_checkpoint_weights(self.model(), path)

    def test_colliding_backbone_prefixes_are_rejected(self):
        weights = dict(self.model().state_dict())
        weights['obs_encoder.scene_encoder.vggt_encoder.weight'] = weights['obs_encoder.scene_encoder.backbone.weight'].clone()
        with tempfile.TemporaryDirectory(prefix='rtpolicy_checkpoint_collision_') as tmp:
            path = Path(tmp) / 'collision.safetensors'
            save_file(weights, str(path))
            with self.assertRaisesRegex(ValueError, 'contains both'):
                load_checkpoint_weights(self.model(), path)


if __name__ == '__main__':
    unittest.main()
