# BrainLM 1-TR-Patch Pretraining

Continued MAE pretraining of BrainLM with **`timepoint_patching_size=1`** — one token per
region per TR, instead of `old_13M`'s 20-TR-averaged patches.

## Motivation

`../validation/` (see those notebooks for the full analysis) established that `old_13M`'s
last-layer, region x region attention:

- **same-time-patch (diagonal) blocks correlate with real functional connectivity (FC)**
  — r≈0.12 patch-matched, r≈0.43 whole-series-averaged (`fc_attention_correlation.ipynb`,
  `fc_substituted_attention_reconstruction.ipynb`);
- **lagged (off-diagonal) blocks do not correlate with real delay connectivity (FDC)** —
  r≈-0.02 to -0.03 (std≈0.04) across lags of 1-4 patches, i.e. noise
  (`ftau_fdc_asymmetry_check.ipynb`).

Working hypothesis: `old_13M` averages every 20 raw TRs into a single token before the
encoder ever sees them, which may be washing out exactly the fine-grained, lagged
temporal structure that FDC is sensitive to — leaving same-time (diagonal) attention free
to track FC, but giving lagged (off-diagonal) attention nothing left to correlate with.
This pipeline tests that hypothesis directly: pretrain BrainLM at `timepoint_patching_size=1`
(no time-binning at all) and re-run the same FC/FDC attention-correlation analysis against
this model to see whether the off-diagonal/FDC correlation appears once the coarse temporal
binning is removed.

> `fmri_timeseries.h5` (424-region, A424 atlas) → arrow datasets → BrainLM
> (`timepoint_patching_size=1`) → MAE pretraining, from scratch or warm-started →
> (back to `../validation/`-style FC/FDC attention analysis, not yet written for this model)

**Cluster only.** Every script here is sized for SLURM GPU/CPU allocations (see
`#SBATCH --mem` in each `slurm/*.sh`), not a laptop — `train.py` in particular holds a
full arrow dataset plus a batch of `[batch, 424 voxels, window] `signal tensors in RAM
on top of the model; don't smoke-test the training scripts locally, only
`bench_seq_len.py --device cpu` on tiny configs is safe to poke at outside the cluster.

---

## Pipeline

Run in this order (`slurm/00`–`04`), each a submittable `sbatch slurm/NN_*.sh`:

| Step | Script | What it does | GPU? |
|---|---|---|---|
| 00 | `bench_seq_len.py` | Forward+backward on synthetic data for a few candidate `num_timepoints_per_voxel` windows; reports step time + peak GPU memory. Run **first** to pick a window size before committing to 01–04. | yes |
| 01 | `prepare_dataset.py` | Builds `train`/`val`/`test`/`coords` arrow datasets (80/10/10 split) from `fmri_timeseries.h5` + `A424_Coordinates.dat`. | no |
| 02 | `transplant_checkpoint.py` | Warm-starts a `timepoint_patching_size=1` model from `old_13M`: copies every shape-matching weight, reinitializes only the two patch-size-dependent projections. Only needed for the warm-start branch (04). | no |
| 03 | `train.py` (`slurm/03_train_from_scratch.sh`) | Random-init control run: same architecture as `old_13M` (`hidden_size=512`, 4 encoder / 2 decoder layers, Nystromformer attention), trained from scratch. | yes |
| 04 | `train.py` (`slurm/04_train_warm_start.sh`) | Continues MAE pretraining from the checkpoint 02 produced. Compare against 03 to see whether transplanting `old_13M`'s weights actually helps. | yes |

`WINDOW` (`num_timepoints_per_voxel`) must be the **same value** across 00/02/03/04 —
it's an env var each `slurm/*.sh` reads (default `64`), not hardcoded, precisely so a
mismatch doesn't quietly happen.

## Technical notes

With `timepoint_patching_size=1`, sequence length is
`num_brain_voxels * num_timepoints_per_voxel + 1` (CLS) — e.g. `424 * 64 + 1 = 27,137`
tokens for the default window. The encoder is Nystromformer (linear attention via
landmarks — `num_landmarks=64`, `segment_means_seq_len=64`, `conv_kernel_size=65` in
`BrainLMConfig`), which scales sub-quadratically with sequence length, but the
embedding/FFN/decoder cost is still `O(seq_len × hidden_size)` per layer — that's what
`bench_seq_len.py` measures directly rather than assuming.

`transplant_checkpoint.py` documents exactly which two parameter tensors depend on
`timepoint_patching_size` and must be reinitialized
(`vit.embeddings.signal_embedding_projection.weight`, `decoder.decoder_pred2.{weight,bias}`);
everything else — both Nystromformer stacks, xyz projections, `cls_token`,
`mask_token`, `decoder_embed`, `decoder_norm` — is shape-identical to `old_13M` and
copied verbatim. It's a **warm start, not a drop-in checkpoint**: raw single-TR tokens
are a different input distribution than 20-TR-averaged patches, and the sinusoidal
position encoding is evaluated over a much longer within-region range than `old_13M`
ever saw — continued pretraining (04) is still required.

## Data

Same input convention as the parent pipeline (`../README.md`): an HDF5 file with
`parcel_ts` `[N subjects, T, 424 regions]` (+ optional `subject_ids`) and the
`A424_Coordinates.dat` MNI coordinates file. Copy both to
`$DATA_DIR/input/` on the cluster yourself (scp/rsync/Globus) — real subject data,
never goes through git.

Normalization matches `run_model.py`'s `preprocess_subject_data`: per-region robust
scaling, `(x - median) / IQR`, computed independently per subject. This has to match
because 04 warm-starts from `old_13M`, which expects inputs on this scale.

Output layout (under `$DATA_DIR/pretrain_1tr/`):

```
arrow/
  train/ val/ test/          # one row per subject, full [T, 424] recording
  coords/                    # one row per region: X, Y, Z
checkpoints/
  warm_start_init/           # output of 02 (transplant_checkpoint.py)
  from_scratch/<run_name>/   # output of 03
  warm_start/<run_name>/     # output of 04
```

`train.py` doesn't pre-window: each epoch it slices a random `num_timepoints_per_voxel`-
length window from the full recording per sample (`preprocess_fmri`) — the model's
existing augmentation strategy, unchanged from the parent pipeline.

## Environment variables

Read by `slurm/*.sh`, all overridable when you `sbatch`:

- `PROJ_DIR` — repo root (default `$HOME/projects/brainlm-attention`)
- `DATA_DIR` — data root (default `$HOME/data/brainlm`)
- `CONDA_PREFIX_BASE` — where the `brainlm` conda env lives (default `$HOME/miniconda3`)
- `WINDOW` — `num_timepoints_per_voxel`, must match across 00/02/03/04 (default `64`)
- `RUN_NAME` — subdirectory name for a training run (default: timestamp)

```bash
sbatch slurm/00_bench_seq_len.sh
sbatch slurm/01_prepare_dataset.sh
WINDOW=64 sbatch slurm/02_transplant_checkpoint.sh
WINDOW=64 RUN_NAME=my_run sbatch slurm/04_train_warm_start.sh
```

`02_transplant_checkpoint.sh` needs network access to pull `old_13M` from the HF Hub
the first time — point `--hf_repo` at a local snapshot directory instead if compute
nodes have no internet.

## Files

```
train.py                     # HF Trainer-based MAE pretraining script (from scratch or warm-started)
prepare_dataset.py           # fmri_timeseries.h5 -> train/val/test/coords arrow datasets
bench_seq_len.py             # synthetic forward+backward benchmark for a candidate window size
transplant_checkpoint.py     # old_13M -> timepoint_patching_size=1 warm-start checkpoint
utils/brainlm_trainer.py     # Trainer subclass: wandb logging, periodic prediction-trend plots
utils/metrics.py             # eval metrics (MSE, MAE, R2, Pearson r on masked tokens) + eval plots
utils/plots.py                # wandb plotting helpers (histograms, scatter, UMAP of CLS tokens, trend lines)
slurm/00_bench_seq_len.sh
slurm/01_prepare_dataset.sh
slurm/02_transplant_checkpoint.sh
slurm/03_train_from_scratch.sh
slurm/04_train_warm_start.sh
```

BrainLM model/code (c) Yale van Dijk Lab, license CC BY-NC-ND 4.0 (non-commercial).
