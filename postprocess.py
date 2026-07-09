"""
Post-processing for the raw (no-CLS) last-layer attention matrix: time-lagged
region x region matrices, similar to the lagged functional-connectivity F(tau)
matrices used to study lead-lag / directed structure in resting-state fMRI, plus
a scalar asymmetry score for each.

IMPORTANT -- token order: `attn_no_cls` must use BrainLM's native raster order,
region-major / time-minor (token i = region * num_patches + time_patch), i.e. the
same order run_model.py's identity-noise guard protects and reshapes with
`.view(num_regions, num_patches, num_regions, num_patches)`. This is NOT the
order used by the `attn_4d = full_attention.view(num_timepoints, num_regions, ...)`
line in model_testing.ipynb (that reshape is time-major and does not match the
model's actual token order -- see `attn_4d_alt` in the same notebook, which uses
the correct region-major convention).
"""
import numpy as np


def compute_ftau(attn_no_cls, num_regions, num_patches, tau=1):
    """F_tau[i, j] = attention from region i at time patch t to region j at time
    patch t+tau, averaged over all valid t. tau=0 gives the same-time (instantaneous)
    region x region matrix.

    attn_no_cls: [num_regions*num_patches, num_regions*num_patches], CLS already dropped.
    Returns: [num_regions, num_regions] numpy array.
    """
    if hasattr(attn_no_cls, "detach"):
        attn_no_cls = attn_no_cls.detach().cpu().numpy()
    if not (0 <= tau < num_patches):
        raise ValueError(f"tau={tau} must be in [0, {num_patches - 1}]")

    attn_4d = attn_no_cls.reshape(num_regions, num_patches, num_regions, num_patches)
    slabs = np.stack([attn_4d[:, t, :, t + tau] for t in range(num_patches - tau)], axis=0)
    return slabs.mean(axis=0)


def ftau_asymmetry(ftau_matrix):
    """Scalar asymmetry score for a region x region matrix: mean(|(A - A.T) / 2|).
    Zero for a symmetric matrix; larger values indicate more net directed
    (source -> target) structure."""
    antisymmetric = (ftau_matrix - ftau_matrix.T) / 2.0
    return float(np.mean(np.abs(antisymmetric)))
