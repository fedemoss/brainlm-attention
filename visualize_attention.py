#!/usr/bin/env python3
"""
Render heatmaps for one or more per-subject attention matrices saved by run_model.py
(`<subject_id>_attention.npy`, each a [424, 424] float32 region x region matrix).

Examples
--------
# single subject
python visualize_attention.py --input outputs/attention_matrices/Sub_0000_attention.npy

# every subject in a directory -> outputs/attention_matrices/figures/*.png
python visualize_attention.py --input outputs/attention_matrices
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_matrix(matrix, title, out_path, xlabel="target", ylabel="source", cmap="magma", figsize=(6.5, 5.5)):
    """Save a percentile-clipped heatmap of a 2D matrix. Reusable for any [N, N] attention
    matrix -- region x region, token x token, F_tau, etc."""
    v_lo, v_hi = np.percentile(matrix, 1), np.percentile(matrix, 99)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, vmin=v_lo, vmax=v_hi)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_one(npy_path, out_path):
    a = np.load(npy_path)
    subject = os.path.basename(npy_path).replace("_attention.npy", "")
    plot_matrix(
        a,
        f"{subject} -- last-layer region x region attention\n(avg heads / windows / time patches)",
        out_path,
        xlabel="target region",
        ylabel="source region",
    )
    print(f"[{subject}] shape={a.shape} min={a.min():.2e} max={a.max():.2e} mean={a.mean():.2e} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="a single <subject_id>_attention.npy file, or a directory of them")
    ap.add_argument("--outdir", default=None,
                    help="where to save PNGs (default: alongside the input, in a 'figures/' subfolder)")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        npy_files = sorted(
            os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith("_attention.npy")
        )
        outdir = args.outdir or os.path.join(args.input, "figures")
    else:
        npy_files = [args.input]
        outdir = args.outdir or os.path.join(os.path.dirname(args.input) or ".", "figures")

    os.makedirs(outdir, exist_ok=True)
    for npy_path in npy_files:
        subject = os.path.basename(npy_path).replace("_attention.npy", "")
        plot_one(npy_path, os.path.join(outdir, f"{subject}.png"))


if __name__ == "__main__":
    main()
