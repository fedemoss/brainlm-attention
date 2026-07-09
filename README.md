# BrainLM Attention Pipeline

Batch pipeline that runs a cohort of fMRI recordings through the **BrainLM** foundation
model (native, region x time tokenization) and saves each subject's **last-layer,
region x region attention matrix** (averaged over attention heads, time patches, and a
few sliding windows per subject).

> cohort `.h5` (`[N subjects, T, 424 regions]`) → BrainLM `old_13M` → one `[424, 424]`
> attention matrix per subject

---

## 1. Setup

Requires Python 3.10+, ~1 GB free (`old_13M` checkpoint is small; weights are cached
on first run).

```bash
conda create -n brainlm python=3.10 -y && conda activate brainlm
pip install -r requirements.txt
```

The pinned versions matter: `transformers 4.28` needs the older `huggingface_hub 0.18`,
and `accelerate` must stay compatible with it. `brainlm_mae/` (the model code, vendored
from `vandijklab/BrainLM`, CC BY-NC-ND 4.0) is already included in this repo — no
separate clone step needed. Model **weights** download automatically from the
`vandijklab/brainlm` Hub repo on first run and are cached (`~/.cache/huggingface`).

## 2. Data

- **Input**: an HDF5 file with dataset `parcel_ts` of shape `[N subjects, T timepoints,
  424 regions]` (A424 atlas) and optionally `subject_ids` (`[N]`, used to name output
  files; falls back to `subject_i` if absent). `T` must be at least `--window_size`
  (200 by default).
- **Coordinates**: `A424_Coordinates.dat` — the MNI coordinates file that ships with
  the upstream `vandijklab/BrainLM` repo's `toolkit/atlases/`.

For local development, `data/input/` mirrors the cluster's `${DATA_DIR}/input/` layout
(`fmri_timeseries_subset_100.h5`, `toolkit/atlases/A424_Coordinates.dat`) so `run_model.sh`'s
paths resolve the same way here and on the cluster. `data/` is git-ignored — it holds real
subject fMRI data and must never be committed or pushed.

## 3. Run

```bash
python run_model.py \
    --input_h5  /path/to/cohort.h5 \
    --output_dir /path/to/outputs/attention_matrices \
    --coords    /path/to/toolkit/atlases/A424_Coordinates.dat \
    --hf_repo   vandijklab/brainlm \
    --subfolder old_13M
```

`--hf_repo` can be a Hub repo id (weights download automatically, as above) or a path
to a local snapshot directory. For SLURM clusters, `run_model.sh` is a template you can
submit with `sbatch run_model.sh` after pointing `PROJ_DIR` / `DATA_DIR` (env vars or
edit the defaults in the script) at your cluster paths.

**Output**: one `<subject_id>_attention.npy` per subject in `--output_dir`, each a
`[424, 424]` `float32` array — the last transformer layer's attention, averaged over
heads, sliding windows, and time patches, with the CLS token dropped. Runs are
auto-resumable: existing output files are skipped.

Useful flags: `--window_size` (TRs per window, default 200 — must be a multiple of the
checkpoint's `timepoint_patching_size`, which is 20 for `old_13M`), `--num_windows`
(sliding windows averaged per subject, default 3).

## 4. Preprocessing

Each region (row) is robust-scaled across the full recording — `(x - median) / IQR`,
per the paper's normalization — before being cut into `--num_windows` overlapping
windows of `--window_size` TRs each. Attention is computed per window and averaged.

## 5. Correctness guarantees

Two checks are enforced and will hard-fail the run (not silently produce garbage) if
violated — both were previously silent bugs in this pipeline that produced attention
matrices from an untrained, token-shuffled model without any error or warning
surfacing in the SLURM logs:

1. **Clean weight load.** `old_13M` is the only Hub subfolder whose parameter names
   match the `BrainLMForPretraining` class this script loads (the alternative
   subfolders, `vitmae_111M` / `vitmae_650M`, hold a different architecture,
   `ViTMAEForPreTraining`, with different parameter names — `attention.attention.*` vs
   `attention.self.*`, `patch_embeddings` vs `signal_embedding_projection`, etc.).
   Loading a mismatched subfolder makes `transformers` silently re-initialize every
   encoder/embedding weight at random while still reporting "success". We parse the
   loader's log for `"newly initialized"` / `"were not used"` and raise instead of
   continuing if either appears — see `load_brainlm_model` in `run_model.py`. Every run
   should log `Clean weight load = True`.
2. **Identity token order.** BrainLM's ViT-MAE-style masking permutes tokens via
   `argsort(noise)` even at `mask_ratio=0.0` (nothing is dropped, but the surviving
   order is a random permutation unless `noise` is supplied explicitly). The
   region x time reshape (`region_mean = attn_reshaped.mean(dim=(1, 3))`) assumes token
   `i` corresponds to `(region, timepatch) = divmod(i, num_patches)` in raster order,
   which only holds if the token order is untouched. We feed an explicit identity
   `noise` and assert `ids_restore` equals the identity permutation before reshaping;
   if it doesn't, the subject is skipped with a logged error instead of silently
   producing a scrambled matrix.

`brainlm_output_33782.out` is a real SLURM log from before this fix — its "were not
used" / "newly initialized" warnings are the loader mismatch described above, showing
up in practice.

## 6. Files

```
run_model.py       # cohort .h5 -> per-subject last-layer region x region attention .npy
run_model.sh        # SLURM submission template
brainlm_mae/         # vendored BrainLM model code (from vandijklab/BrainLM)
requirements.txt    # pinned deps
checks.ipynb, model_testing.ipynb, output.ipynb   # exploratory notebooks, not part of the pipeline
```

BrainLM model/code (c) Yale van Dijk Lab, license CC BY-NC-ND 4.0 (non-commercial).
