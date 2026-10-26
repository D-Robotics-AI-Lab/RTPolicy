#!/usr/bin/env python3
"""Fix checkpoint config.json files from the old VODPPlus class names to RTPolicy naming.

Existing checkpoints have pkg_map / auto_map pointing at drrm.models.VODPPlus*
(classes renamed to RTPolicy*); without this fix load_policy fails to resolve.
Only rewrites a few string fields in config.json; weights are untouched except
for the explicit prefix remap below.

Usage:
    python tools/rename_ckpt_pkg_map.py <ckpt_dir>...      # one by one
    python tools/rename_ckpt_pkg_map.py --root <dir>       # recursively scan config.json
    Add --dry-run to print only, no writes
"""
from __future__ import annotations

import argparse
import json
import os

MAP = [
    ("drrm.models.VODPPlusDitFlowMatchingConfig", "drrm.models.RTPolicyDitFlowMatchingConfig"),
    ("drrm.models.VODPPlusDitFlowMatching", "drrm.models.RTPolicyDitFlowMatching"),
    # emvis_config key renames
    ("vggt_target", "backbone_target"),
    ("vggt_model_path", "backbone_path"),
    ("load_vggt_pretrain", "load_backbone_pretrain"),
    ("vggt_config", "backbone_config"),
    ("vggt_train_backbone", "backbone_trainable"),
]
UNSUPPORTED_POLICY_CLASSES = (
    "drrm.models.VODPPlusDitFlowMatchingNoAdapter",
    "drrm.models.VODPPlusDitFlowMatchingNoAdapterConfig",
    "drrm.models.RTPolicyDitFlowMatchingNoAdapter",
    "drrm.models.RTPolicyDitFlowMatchingNoAdapterConfig",
)


def patch_config(path: str, dry: bool) -> bool:
    with open(path) as f:
        raw = f.read()
    if any(class_name in raw for class_name in UNSUPPORTED_POLICY_CLASSES):
        print(f"skip unsupported checkpoint policy removed from RTPolicy: {path}")
        return False
    s = raw
    for old, new in MAP:
        s = s.replace(old, new)
    if s == raw:
        return False
    if not dry:
        with open(path, "w") as f:
            f.write(s)
    print(("would patch " if dry else "patched ") + path)
    return True


WEIGHT_PREFIX_MAP = [
    ("vggt_encoder.", "backbone."),
]


def patch_weights(ckpt_dir: str, dry: bool) -> bool:
    """state_dict prefix remap (vggt_encoder. -> backbone.): rewrites model.safetensors in place."""
    import shutil
    from safetensors.torch import load_file, save_file
    wt = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.isfile(wt):
        return False
    sd = load_file(wt)
    sd2 = {}
    changed = False
    for k, v in sd.items():
        k2 = k
        for old, new in WEIGHT_PREFIX_MAP:
            k2 = k2.replace(old, new)
        if k2 != k:
            changed = True
        sd2[k2] = v
    if not changed:
        return False
    if not dry:
        shutil.copy(wt, wt + ".bak_vodpp_rename")
        save_file(sd2, wt)
    print(("would rewrite " if dry else "rewrote ") + wt)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="checkpoint dirs (must contain config.json)")
    ap.add_argument("--root", help="recursively find config.json under this dir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = []
    for d in args.dirs:
        cfg = os.path.join(d, "config.json")
        if os.path.isfile(cfg):
            targets.append(cfg)
        else:
            print(f"skip (no config.json): {d}")
    if args.root:
        for cur, _, files in os.walk(args.root):
            if "config.json" in files:
                targets.append(os.path.join(cur, "config.json"))

    n = sum(patch_config(t, args.dry_run) for t in targets)
    print(f"\n{n}/{len(targets)} configs patched")
    if not args.dry_run:
        dirs = {os.path.dirname(t) for t in targets}
        m = sum(patch_weights(d, args.dry_run) for d in dirs)
        print(f"{m}/{len(dirs)} weight files rewritten")


if __name__ == "__main__":
    main()
