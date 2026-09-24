#!/usr/bin/env python3
"""
Production BrainLM inference pipeline: extract the LAST-LAYER region x region
attention matrix for every subject in an HDF5 cohort, using a local
`timepoint_patching_size=1` BrainLM checkpoint (BrainLMForPretraining, region x
time tokenization).

Adapted from `../../run_model.py` (which loads `old_13M` from the HuggingFace
Hub, `timepoint_patching_size=20`, `window_size=200`) for a locally-saved
checkpoint trained with `timepoint_patching_size=1` (see `../README.md`).
Only the model-loading path and the default `--window_size` differ; the
preprocessing/reshape/ftau/asymmetry logic is unchanged and still generic
over `config.timepoint_patching_size` / `config.num_brain_voxels`.

Same correctness guarantee as the original: identity token order. BrainLM's
ViT-MAE-style masking permutes tokens via `argsort(noise)` even when
mask_ratio=0.0. The region x time reshape below assumes token i corresponds
to (region, timepatch) = divmod(i, num_patches) in raster order, which is
only valid if the token order is untouched. We feed an explicit identity
noise and assert `ids_restore` is the identity permutation before reshaping.
"""
import os
import sys
import io
import csv
import argparse
import logging
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from postprocess import compute_ftau, ftau_asymmetry
from visualize_attention import plot_matrix

# =====================================================================
# 1. CLUSTER CITIZENSHIP & RESOURCE TUNING
# =====================================================================
# Cap CPU threads BEFORE importing heavy libraries to prevent core saturation
MAX_CPU_THREADS = 4
os.environ["OMP_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_CPU_THREADS)
torch.set_num_threads(MAX_CPU_THREADS)


def setup_environment():
    """Configures clean console logging streamed directly to standard output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("BrainLM_1TR")


# =====================================================================
# 2. PIPELINE CORE FUNCTIONS
# =====================================================================
def load_brainlm_model(checkpoint_dir):
    """Initializes and freezes the BrainLM architecture from a LOCAL checkpoint
    directory (e.g. produced by `pretrain_1tr/train.py`), as opposed to
    `run_model.py`'s HuggingFace-Hub `hf_repo/subfolder` loading.

    Still verifies clean weight load: if transformers reports any weight as
    "newly initialized" or "not used", the checkpoint and BrainLMForPretraining
    disagree on architecture and the encoder would be partly/fully randomly
    initialized. Treated as fatal rather than silently returning a garbage model.
    """
    import transformers
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining
    from brainlm_mae.configuration_brainlm import BrainLMConfig

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    logging.info(f"Loading Model Configuration and Architecture from local checkpoint: {checkpoint_dir} ...")
    config = BrainLMConfig.from_pretrained(checkpoint_dir)

    buf = io.StringIO()

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            buf.write(self.format(record) + "\n")

    transformers.logging.set_verbosity_info()
    hf_logger = transformers.logging.get_logger("transformers.modeling_utils")
    handler = _CaptureHandler()
    hf_logger.addHandler(handler)
    try:
        model = BrainLMForPretraining.from_pretrained(checkpoint_dir, config=config)
    finally:
        hf_logger.removeHandler(handler)

    log = buf.getvalue()
    clean = ("newly initialized" not in log) and ("were not used" not in log)
    logging.info(f"Clean weight load = {clean}")
    if not clean:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_dir!r} does not cleanly match "
            "BrainLMForPretraining -- some weights were randomly initialized "
            "or unused. Loader log:\n" + log
        )

    model.eval()
    model.config.mask_ratio = 0.0
    model.config.output_attentions = True

    for param in model.parameters():
        param.requires_grad = False

    return model, config


def load_spatial_coordinates(coords_file):
    """Loads MNI spatial coordinates and prepares tensor format."""
    logging.info(f"Loading spatial coordinates from: {coords_file}")
    if not os.path.exists(coords_file):
        raise FileNotFoundError(f"Coordinates file missing at: {coords_file}")

    real_coords = np.loadtxt(coords_file)[:, 1:4]
    return torch.tensor(real_coords, dtype=torch.float32).unsqueeze(0)


def preprocess_subject_data(raw_ts, num_windows=3, window_size=32):
    """Applies Robust Scaling and sections data into sliding windows."""
    dat_arr = raw_ts.T  # [num_regions, num_timepoints]

    invalid_cols = np.isnan(dat_arr).any(axis=0)
    if np.any(invalid_cols):
        dat_arr = dat_arr[:, ~invalid_cols]

    num_timepoints = dat_arr.shape[1]
    if num_timepoints < window_size:
        raise ValueError(f"Recording has {num_timepoints} TRs, but requires at least {window_size}.")

    for idx in range(dat_arr.shape[0]):
        voxel_data = dat_arr[idx, :]
        median = np.median(voxel_data)
        q75, q25 = np.percentile(voxel_data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1e-6
        dat_arr[idx, :] = (voxel_data - median) / iqr

    windows = []
    step_size = (num_timepoints - window_size) // (num_windows - 1)

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        if i == num_windows - 1:
            end_idx = num_timepoints
            start_idx = num_timepoints - window_size

        windows.append(dat_arr[:, start_idx:end_idx])

    return torch.tensor(np.stack(windows, axis=0), dtype=torch.float32)


# =====================================================================
# 3. EXECUTION ORCHESTRATOR
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="BrainLM (timepoint_patching_size=1) inference pipeline")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to input timeseries HDF5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save individual attention matrices")
    parser.add_argument("--coords", type=str, required=True, help="Path to A424_Coordinates.dat file")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                         help="Local directory holding the trained checkpoint (config.json + pytorch_model.bin)")
    parser.add_argument("--window_size", type=int, default=32,
                         help="TRs per window. Must match the checkpoint's training WINDOW "
                              "(num_timepoints_per_voxel) -- see pretrain_1tr/README.md.")
    parser.add_argument("--num_windows", type=int, default=3, help="Number of sliding windows averaged per subject")
    parser.add_argument("--ftau_lag", type=int, default=1,
                         help="Time-patch (=TR, since timepoint_patching_size=1) lag tau for the F_tau matrix. "
                              "0 = same-time (instantaneous).")
    parser.add_argument("--figures_dir", type=str, default="figures",
                         help="Where to save the full [seq_len, seq_len] last-layer attention heatmap per subject")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only iterate over the first N rows of the cohort.")
    args = parser.parse_args()

    logger = setup_environment()
    logger.info("Initializing BrainLM (1TR) Execution Sequence...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA Device Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found. Running on CPU.")

    try:
        model, config = load_brainlm_model(args.checkpoint_dir)
        model.to(device)

        if args.window_size % config.timepoint_patching_size != 0:
            raise ValueError(
                f"--window_size {args.window_size} is not a multiple of this checkpoint's "
                f"timepoint_patching_size ({config.timepoint_patching_size}); the region/time "
                "reshape below would be invalid."
            )

        spatial_coords_base = load_spatial_coordinates(args.coords)
    except Exception as e:
        logger.critical(f"Failed to initialize core infrastructure: {str(e)}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    asymmetry_csv = os.path.join(args.output_dir, "asymmetry_scores.csv")
    write_header = not os.path.exists(asymmetry_csv)
    asymmetry_file = open(asymmetry_csv, "a", newline="")
    asymmetry_writer = csv.writer(asymmetry_file)
    if write_header:
        asymmetry_writer.writerow(["subject_id", "ftau_lag", "asymmetry"])

    num_regions = config.num_brain_voxels
    num_patches = args.window_size // config.timepoint_patching_size
    seq_len = num_regions * num_patches
    identity_noise = (torch.arange(seq_len, dtype=torch.float32) / seq_len)

    logger.info(f"num_regions={num_regions}, num_patches={num_patches}, seq_len={seq_len} "
                f"(timepoint_patching_size={config.timepoint_patching_size}, window_size={args.window_size})")

    with h5py.File(args.input_h5, "r") as f_in:
        ts_dataset = f_in["parcel_ts"]
        subject_ids = f_in["subject_ids"][:]
        n_subjects, _, _ = ts_dataset.shape
        if args.limit is not None:
            n_subjects = min(n_subjects, args.limit)

        logger.info(f"Found {n_subjects} subjects. Saving outputs as individual files to: {args.output_dir}")

        for i in range(n_subjects):
            sub_id = subject_ids[i].decode("utf-8")
            out_file = os.path.join(args.output_dir, f"{sub_id}_attention.npy")

            if os.path.exists(out_file):
                logger.info(f"Skipping [{i+1}/{n_subjects}] | Subject ID: {sub_id} (Already processed)")
                continue

            logger.info(f"Processing [{i+1}/{n_subjects}] | Subject ID: {sub_id}")

            try:
                raw_data = ts_dataset[i, :, :]

                fmri_windows = preprocess_subject_data(
                    raw_data, num_windows=args.num_windows, window_size=args.window_size
                )
                num_windows = fmri_windows.shape[0]

                # Run windows ONE AT A TIME rather than batched into a single forward call.
                # At old_13M's seq_len (4,240) batching num_windows together was harmless; at
                # this model's seq_len (num_regions*num_patches, into the tens of thousands),
                # a batch of `num_windows` would multiply the already-large per-window
                # attention memory by `num_windows`x -- infeasible on a single GPU/CPU host.
                # Averaging per-window head-means afterward is mathematically identical to
                # averaging over the batch+head dims in one call.
                attn_no_cls_sum = None
                for w in range(num_windows):
                    single_window = fmri_windows[w : w + 1].to(device)
                    spatial_coords = spatial_coords_base.to(device)
                    noise = identity_noise.unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model.vit(
                            signal_vectors=single_window,
                            xyz_vectors=spatial_coords,
                            noise=noise,
                            output_attentions=True,
                        )

                    ids_restore = outputs.ids_restore.cpu().numpy()
                    if not np.array_equal(ids_restore, np.broadcast_to(np.arange(seq_len), ids_restore.shape)):
                        raise RuntimeError("Token order was shuffled (ids_restore != identity); refusing to reshape.")

                    window_attn_no_cls = outputs.attentions[-1].mean(dim=(0, 1))[1:, 1:]
                    attn_no_cls_sum = (
                        window_attn_no_cls if attn_no_cls_sum is None else attn_no_cls_sum + window_attn_no_cls
                    )
                    del outputs, window_attn_no_cls
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                attn_no_cls = attn_no_cls_sum / num_windows
                attn_reshaped = attn_no_cls.view(num_regions, num_patches, num_regions, num_patches)
                region_mean = attn_reshaped.mean(dim=(1, 3))

                subject_mean_np = region_mean.cpu().numpy().astype("float32")
                np.save(out_file, subject_mean_np)

                attn_no_cls_np = attn_no_cls.cpu().numpy().astype("float32")
                ftau_matrix = compute_ftau(attn_no_cls_np, num_regions, num_patches, tau=args.ftau_lag)
                np.save(os.path.join(args.output_dir, f"{sub_id}_ftau{args.ftau_lag}.npy"), ftau_matrix)
                asymmetry = ftau_asymmetry(ftau_matrix)
                asymmetry_writer.writerow([sub_id, args.ftau_lag, asymmetry])
                asymmetry_file.flush()

                plot_matrix(
                    attn_no_cls_np,
                    f"{sub_id} -- full last-layer attention ({attn_no_cls_np.shape[0]} tokens)\n"
                    f"avg heads/windows, F_tau(tau={args.ftau_lag}) asymmetry={asymmetry:.2e}",
                    os.path.join(args.figures_dir, f"{sub_id}_full_attention.png"),
                    xlabel="target token (region x time patch)",
                    ylabel="source token (region x time patch)",
                )

            except Exception as sub_error:
                logger.error(f"SKIPPED Subject {sub_id} due to processing error: {str(sub_error)}")
                continue

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    asymmetry_file.close()
    logger.info("All operations complete.")


if __name__ == "__main__":
    main()
