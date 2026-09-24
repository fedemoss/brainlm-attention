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

**03 is currently skipped.** The dataset available today (`../playground/fmri/fmri_timeseries.h5`,
1082 subjects → ~866/108/108 train/val/test after the 80/10/10 split, see "Data" below) is
~2 orders of magnitude smaller than the ~70k-recording corpus `old_13M` itself was pretrained
on. A from-scratch run has to learn the entire Nystromformer attention geometry, spatial
embedding, and reconstruction task from that alone, against a ~27k-token sequence length —
not expected to converge to anything useful at this scale, so it's not part of the default
submission (see `slurm/submit_pipeline.sh` below). It's still there to run by hand later if
the dataset grows.

## Technical notes

With `timepoint_patching_size=1`, sequence length is
`num_brain_voxels * num_timepoints_per_voxel + 1` (CLS) — e.g. `424 * 64 + 1 = 27,137`
tokens for the default window.

**Attention here is exact O(seq_len²) per layer, not the sub-quadratic Nystrom
approximation.** `NystromformerSelfAttention.forward` (installed `transformers` package)
has a branch, `if self.num_landmarks == self.seq_len` — where `self.seq_len` is actually
`config.segment_means_seq_len`, a config value, not the real runtime sequence length —
that falls back to exact full softmax attention when the two are equal. Every config
used in this project (`old_13M`, `bench_seq_len.py`'s `BASE_CONFIG`, the transplanted
checkpoint, 03/04) sets `num_landmarks=64` and `segment_means_seq_len=64`, so this branch
always fires. Memory is `O((num_brain_voxels * window)² * batch * num_attention_heads)`
per layer as a result.

Confirmed empirically (`--batch_size 4`): `window=16` fits on T4/L4/A100 alike (12.94GB
peak, identical across GPU types since it's just tensor size, not GPU-dependent).
`window=32/64/128` OOM'd everywhere tested — T4 (`c2`/`c3`, ~14.56GiB), L4 (`c6`,
22.04GiB), *and* A100 (`c4`, 39.49GiB). **The A100 result is not trustworthy as-is**,
though: at the moment `window=32` was attempted, the process already had 30.60GB "in
use" left over from the `window=16` run that had just completed in the same Python
process — `del model, out; torch.cuda.empty_cache()` didn't reclaim it (likely
torch/dynamo checkpoint-machinery caching). `00_bench_seq_len.sh` now runs each window as
a **separate process** (clean CUDA context per window) and at `--batch_size 2` (matching
`03`/`04`'s actual `--per_device_train_batch_size 2`, not the 2x-larger `4` originally
benched) — re-run it before trusting any `window>=32` conclusion; the contaminated/2x
numbers above may have been meaningfully pessimistic. `c4` (2x A100) remains the
strongest candidate for `window>=32` if it does fit; hence `--nodelist=c4` in
`00_bench_seq_len.sh`/`03`/`04` (see "Cluster hardware" below). **Always run
`00_bench_seq_len.sh` via `sbatch` on an actual GPU node before picking `WINDOW`** —
running `bench_seq_len.py` directly on a login node silently falls back to CPU
(`torch.cuda.is_available()` is `False` there) and is not representative (and is how the
RAM-exhaustion crash that started this pipeline happened).

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

Currently available: `../playground/fmri/fmri_timeseries.h5` — `parcel_ts` shape
`(1082, 1200, 424)` — i.e. **1082 subjects**, 1200 TRs each. After `prepare_dataset.py`'s
80/10/10 split that's ~866 train / ~108 val / ~108 test subjects; this is the number behind
the decision to skip 03 above.

Normalization matches `run_model.py`'s `preprocess_subject_data`: per-region robust
scaling, `(x - median) / IQR`, computed independently per subject. This has to match
because 04 warm-starts from `old_13M`, which expects inputs on this scale.

Output layout (under `$DATA_DIR/pretrain_1tr/`):

```
arrow/
  train/ val/ test/            # one row per subject, full [T, 424] recording
  coords/                      # one row per region: X, Y, Z
checkpoints/
  warm_start_init_w<WINDOW>/   # output of 02 (transplant_checkpoint.py), one per window
  from_scratch/<run_name>/     # output of 03
  warm_start/<run_name>/       # output of 04
```

`warm_start_init_w<WINDOW>/` is suffixed by window on purpose: 02 is cheap/CPU-only, so
it's fine to run for several candidate windows ahead of time (in parallel, even) without
them racing on the same output directory. 04 reads `warm_start_init_w${WINDOW}` — the
`WINDOW` it's given must match whichever one 02 was run with, or it fails fast (directory
not found) rather than silently loading the wrong window's checkpoint.

`train.py` doesn't pre-window: each epoch it slices a random `num_timepoints_per_voxel`-
length window from the full recording per sample (`preprocess_fmri`) — the model's
existing augmentation strategy, unchanged from the parent pipeline.

## Cluster hardware

5 nodes, all Intel Xeon Gold 6230 / 192GB RAM (`c5` a bit more):

| Node | GPUs | Role |
|---|---|---|
| `pinky` | none | login/master — never run training or bench here, silently falls back to CPU |
| `c2` | 2x T4 | fits `window<=16` only (confirmed) |
| `c3` | 2x T4 | same as `c2` |
| `c4` | 2x A100 | the only node with a real chance at `window>=32` — GPU scripts default here |
| `c5` | none | storage node |
| `c6` | 2x L4 | in between T4 and A100; not yet benched here |

Storage: per-disk mounts at `/share/data<N>/<user>` (not a parallel filesystem — each
`data<N>` is a separate disk) plus `/share/raid<N>` on RAID1. **The `data<N>` disks are
RAID0 — no redundancy.** `DATA_DIR` (below) points at `/share/data1/mossf/data/brainlm` by
default, which means checkpoints and prepared datasets there have no fault tolerance if
that disk fails; worth keeping in mind for anything you'd be unhappy to lose.

`sinfo`'s own node/partition listing on this cluster has been unreliable in practice (it
reported `MEMORY=1` for every node, and GPU-type strings that didn't match the above) —
this table is from direct confirmation, trust it over `sinfo` output.

## Environment variables

Read by `slurm/*.sh`, all overridable when you `sbatch`:

- `PROJ_DIR` — repo root (default `$HOME/projects/brainlm-attention`) — code lives under home
- `DATA_DIR` — data root (default `/share/data1/mossf/data/brainlm`) — data/checkpoints live on
  fast shared storage, not home
- `CONDA_PREFIX_BASE` — where the `brainlm` conda env lives (default `/share/data1/mossf/miniconda3`,
  same fast-storage mount, the cluster's fast-storage conda install)
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

## Submitting the whole (warm-start) pipeline at once

`slurm/submit_pipeline.sh` submits 00, 01, 02 immediately and 04 with
`--dependency=afterok:<01 jobid>:<02 jobid>`, so it won't start until both the arrow
datasets and the transplanted checkpoint exist. It does **not** submit 03 (see "03 is
currently skipped" above).

```bash
./slurm/submit_pipeline.sh                       # WINDOW=64
WINDOW=32 RUN_NAME=my_run ./slurm/submit_pipeline.sh
```

If 01 or 02 fails, 04 will sit in the queue as `DependencyNeverSatisfied` and never run —
`squeue -u $USER` to check, `scancel <04 jobid>` and resubmit once the failure's fixed.
00 doesn't block anything (it's informational, for picking `WINDOW`); if its numbers
suggest a different window than the default, cancel/ignore the run in progress and
resubmit with `WINDOW=<value>`.

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
slurm/03_train_from_scratch.sh          # from-scratch control, not currently submitted (see above)
slurm/04_train_warm_start.sh
slurm/submit_pipeline.sh                # chains 00+01+02 -> 04 via sbatch --dependency
```

BrainLM model/code (c) Yale van Dijk Lab, license CC BY-NC-ND 4.0 (non-commercial).
