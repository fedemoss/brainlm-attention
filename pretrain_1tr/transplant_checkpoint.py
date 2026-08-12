"""
Warm-start a 1-TR-patch BrainLM from the pretrained `old_13M` checkpoint.

Only two parameter tensors in the whole architecture depend on
`timepoint_patching_size` (see brainlm_mae/modeling_brainlm.py):
    vit.embeddings.signal_embedding_projection.{weight,bias}   Linear(patch_size -> hidden)
    decoder.decoder_pred2.{weight,bias}                        Linear(dec_hidden//2 -> patch_size)
Everything else -- both Nystromformer layer stacks, both xyz projections, cls_token,
mask_token, decoder_embed, decoder_norm, the (unused, inherited) ViTMAEDecoder.decoder_pred
-- is shape-identical between old_13M (patch=20) and a patch=1 model with the same
num_brain_voxels / hidden sizes, and gets copied verbatim.

This is a WARM START, not a drop-in checkpoint: raw single-TR tokens are a different
input distribution than 20-TR-averaged patches, and the sinusoidal position encoding
will be evaluated over a much longer within-region range than old_13M ever saw
(num_timepoints_per_voxel positions instead of 10). Continued MAE pretraining on the
new tokenization (train.py) is still required -- this just avoids starting the
transformer body from random init.

Usage:
    python transplant_checkpoint.py \
        --out_dir "$DATA_DIR/pretrain_1tr/checkpoints/warm_start_init" \
        --num_timepoints_per_voxel 64
"""
import argparse
import io
import json
import logging
import os
import sys
from copy import deepcopy

import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from brainlm_mae.configuration_brainlm import BrainLMConfig
from brainlm_mae.modeling_brainlm import BrainLMForPretraining

# Params that legitimately change shape with timepoint_patching_size and MUST be
# reinitialized. If the copy loop ever finds anything else left uncopied, that's a
# bug (an unexpected architecture mismatch) and the script hard-fails rather than
# silently shipping a partially-random model.
#
# Note: signal_embedding_projection.bias is NOT in this set. nn.Linear's bias has
# shape [out_features] = [hidden_size], which does not depend on patch size (only
# .weight, shape [hidden_size, patch_size], does) -- so the bias transfers cleanly
# and is expected to be copied, verified directly against old_13M below.
EXPECTED_REINIT = {
    "vit.embeddings.signal_embedding_projection.weight",
    "decoder.decoder_pred2.weight",
    "decoder.decoder_pred2.bias",
}


def load_reference(hf_repo, subfolder):
    """Load old_13M and verify it loaded cleanly (no silently-random weights) --
    same check run_model.py / blm_common.py use before trusting a checkpoint."""
    cfg = BrainLMConfig.from_pretrained(hf_repo, subfolder=subfolder)
    buf = io.StringIO()

    class _Handler(logging.Handler):
        def emit(self, record):
            buf.write(self.format(record) + "\n")

    transformers.logging.set_verbosity_info()
    transformers.logging.get_logger("transformers.modeling_utils").addHandler(_Handler())
    model = BrainLMForPretraining.from_pretrained(hf_repo, config=cfg, subfolder=subfolder)
    log = buf.getvalue()
    clean = ("newly initialized" not in log) and ("were not used" not in log)
    return model, cfg, clean


def transplant(ref_model, new_model):
    ref_sd = ref_model.state_dict()
    new_sd = new_model.state_dict()

    copied, reinitialized, shape_mismatch = [], [], []
    for name, new_tensor in new_sd.items():
        if name in ref_sd and ref_sd[name].shape == new_tensor.shape:
            new_sd[name] = ref_sd[name].clone()
            copied.append(name)
        elif name in ref_sd:
            shape_mismatch.append((name, tuple(ref_sd[name].shape), tuple(new_tensor.shape)))
            reinitialized.append(name)
        else:
            reinitialized.append(name)

    new_model.load_state_dict(new_sd)
    return copied, reinitialized, shape_mismatch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_repo", default="vandijklab/brainlm", help="Hub repo id or local snapshot path")
    ap.add_argument("--subfolder", default="old_13M")
    ap.add_argument("--num_timepoints_per_voxel", type=int, required=True, help="Window length in TRs (patch=1, so this is also the per-region token count)")
    ap.add_argument("--mask_ratio", type=float, default=None, help="Override mask_ratio; defaults to old_13M's value")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    print(f"[load] reference checkpoint {args.hf_repo}/{args.subfolder} ...")
    ref_model, ref_cfg, clean = load_reference(args.hf_repo, args.subfolder)
    print(f"[load] clean weight load = {clean}")
    if not clean:
        raise RuntimeError(
            "Reference checkpoint did not load cleanly (transformers reported "
            "newly-initialized or unused weights) -- refusing to transplant from a "
            "mismatched/random checkpoint. Check --hf_repo/--subfolder."
        )
    print(
        f"[ref config] num_brain_voxels={ref_cfg.num_brain_voxels} "
        f"timepoint_patching_size={ref_cfg.timepoint_patching_size} "
        f"num_timepoints_per_voxel={ref_cfg.num_timepoints_per_voxel} "
        f"hidden_size={ref_cfg.hidden_size} num_hidden_layers={ref_cfg.num_hidden_layers}"
    )

    new_cfg = deepcopy(ref_cfg)
    new_cfg.timepoint_patching_size = 1
    new_cfg.num_timepoints_per_voxel = args.num_timepoints_per_voxel
    if args.mask_ratio is not None:
        new_cfg.mask_ratio = args.mask_ratio
    assert new_cfg.num_timepoints_per_voxel % new_cfg.timepoint_patching_size == 0

    print(
        f"[new config] num_brain_voxels={new_cfg.num_brain_voxels} "
        f"timepoint_patching_size={new_cfg.timepoint_patching_size} "
        f"num_timepoints_per_voxel={new_cfg.num_timepoints_per_voxel} "
        f"-> sequence length = {new_cfg.num_brain_voxels * new_cfg.num_timepoints_per_voxel + 1} tokens (incl. CLS)"
    )

    print("[build] instantiating new model (random init) ...")
    new_model = BrainLMForPretraining(new_cfg)

    print("[transplant] copying shape-matching parameters ...")
    copied, reinitialized, shape_mismatch = transplant(ref_model, new_model)

    unexpected = set(reinitialized) - EXPECTED_REINIT
    print(f"[transplant] copied {len(copied)} tensors, reinitialized {len(reinitialized)} tensors")
    for name in sorted(reinitialized):
        print(f"    reinit: {name}  shape={tuple(new_model.state_dict()[name].shape)}")
    if shape_mismatch:
        print("[transplant] shape mismatches (reinitialized instead of copied):")
        for name, ref_shape, new_shape in shape_mismatch:
            print(f"    {name}: ref={ref_shape} new={new_shape}")

    if unexpected:
        raise RuntimeError(
            f"Unexpected reinitialized parameters beyond the known patch-size-dependent "
            f"set: {sorted(unexpected)}. This means the architectures differ in some way "
            f"other than timepoint_patching_size -- inspect before trusting this transplant."
        )
    missing_expected = EXPECTED_REINIT - set(reinitialized)
    if missing_expected:
        print(
            f"[warn] expected these to be reinitialized but they were copied instead "
            f"(shapes happened to match): {sorted(missing_expected)}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    new_model.save_pretrained(args.out_dir)
    report = {
        "source_checkpoint": f"{args.hf_repo}/{args.subfolder}",
        "source_clean_load": clean,
        "new_config": new_cfg.to_dict(),
        "n_copied": len(copied),
        "n_reinitialized": len(reinitialized),
        "reinitialized_params": sorted(reinitialized),
        "shape_mismatches": shape_mismatch,
    }
    with open(os.path.join(args.out_dir, "transplant_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] warm-started model + transplant_report.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
