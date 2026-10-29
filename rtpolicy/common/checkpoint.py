"""Load historical checkpoints through the installed RTPolicy package.

Class references and the former vision-backbone field/tensor names are converted
in memory. Dataset IDs, path values and source checkpoint files stay unchanged.
"""

from copy import deepcopy
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_LEGACY_BACKBONE_FIELDS = {
    "vggt_target": "backbone_target",
    "vggt_config": "backbone_config",
    "vggt_model_path": "backbone_path",
    "load_vggt_pretrain": "load_backbone_pretrain",
}
_LEGACY_BACKBONE_PREFIX = "obs_encoder.scene_encoder.vggt_encoder."
_BACKBONE_PREFIX = "obs_encoder.scene_encoder.backbone."
_LEGACY_ADAPTER_PREFIX = "obs_encoder.scene_encoder.model_adapter.adapter."
_ADAPTER_PREFIX = "obs_encoder.scene_encoder.model_adapter."


_LEGACY_CLASS_NAMES = {
    "VODPPlusDitFlowMatching": "RTPolicyDitFlowMatching",
    "VODPPlusDitFlowMatchingConfig": "RTPolicyDitFlowMatchingConfig",
    "DRRMDataset": "RTPolicyDataset",
}


def canonical_class_path(path):
    """Return the current spelling of a serialized local class reference."""
    if not isinstance(path, str):
        return path
    if path.startswith("drrm."):
        path = "rtpolicy." + path[len("drrm."):]
    if path.startswith("rtpolicy."):
        path = path.replace(".policy.vodpp.", ".policy.rtpolicy.")
        path = path.replace(".datasets.drrm_dataset", ".datasets.rtpolicy_dataset")
    module, separator, name = path.rpartition(".")
    if path.startswith("rtpolicy.") or module in ("", "dit_fm_policy"):
        name = _LEGACY_CLASS_NAMES.get(name, name)
    return module + separator + name


def _normalize_reference(value):
    if isinstance(value, str):
        return canonical_class_path(value)
    if isinstance(value, list):
        return [_normalize_reference(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_reference(item) for item in value)
    return value


def normalize_checkpoint_metadata(config):
    """Copy a config and normalize supported historical class/backbone names."""
    normalized = deepcopy(config)
    for field in ("pkg_map", "auto_map"):
        references = normalized.get(field)
        if isinstance(references, dict):
            normalized[field] = {
                name: _normalize_reference(value)
                for name, value in references.items()
            }
    if isinstance(normalized.get("architectures"), list):
        normalized["architectures"] = [
            _LEGACY_CLASS_NAMES.get(name, name)
            for name in normalized["architectures"]
        ]
    emvis = (normalized.get("obs_encoder") or {}).get("emvis_config")
    if isinstance(emvis, dict):
        for old, current in _LEGACY_BACKBONE_FIELDS.items():
            if old not in emvis:
                continue
            value = emvis.pop(old)
            if current in emvis and emvis[current] is not None and emvis[current] != value:
                raise ValueError(f"Conflicting checkpoint fields: {old} and {current}")
            emvis[current] = value
    return normalized


def load_checkpoint_config(checkpoint_dir):
    """Read current checkpoint metadata without rewriting the source JSON."""
    path = Path(checkpoint_dir) / "config.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return normalize_checkpoint_metadata(config)


def load_checkpoint_weights(model, filename, device=None, strict=True):
    """Load weights with complete coverage, including the old backbone prefix.

    Match the destination model before renaming so explicitly loaded historical
    implementations can still consume their original tensor names.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, load_model

    target_keys = set(model.state_dict())
    with safe_open(str(filename), framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
    renames = {}
    for key in source_keys:
        for old, new in ((_LEGACY_BACKBONE_PREFIX, _BACKBONE_PREFIX),
                         (_LEGACY_ADAPTER_PREFIX, _ADAPTER_PREFIX)):
            if not key.startswith(old):
                continue
            current = new + key[len(old):]
            if current in target_keys:
                if current in source_keys:
                    raise ValueError(f"Checkpoint contains both {key} and {current}")
                renames[key] = current

    if renames:
        weights = load_file(str(filename), device=device or "cpu")
        weights = {renames.get(key, key): value for key, value in weights.items()}
        missing, unexpected = model.load_state_dict(weights, strict=strict)
    else:
        missing, unexpected = load_model(model, str(filename), strict=strict, device=device or "cpu")

    # Normalizer loading creates its ParameterDict dynamically from checkpoint
    # tensors. Keep it on the requested device after that reconstruction.
    if device is not None and getattr(model, "normalizer", None) is not None:
        model.normalizer.to(device=device)
    model._checkpoint_load_report = {
        "source_tensors": len(source_keys), "renamed_tensors": len(renames),
        "renamed_backbone_tensors": sum(key.startswith(_LEGACY_BACKBONE_PREFIX) for key in renames),
        "renamed_adapter_tensors": sum(key.startswith(_LEGACY_ADAPTER_PREFIX) for key in renames),
        "missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected),
    }
    logger.info("Checkpoint weight coverage: %s", model._checkpoint_load_report)
    return missing, unexpected
