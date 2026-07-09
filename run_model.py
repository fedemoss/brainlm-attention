#!/usr/bin/env python3
"""
Production BrainLM inference pipeline: extract the LAST-LAYER region x region
attention matrix for every subject in an HDF5 cohort, using the native
old_13M BrainLM checkpoint (BrainLMForPretraining, region x time tokenization).

Two correctness guarantees are enforced and will hard-fail the run if violated:

  1. Clean weight load. `old_13M` is the only Hub subfolder whose weights match
     BrainLMForPretraining's parameter names. Pointing --subfolder at a ViT-MAE
     subfolder (e.g. vitmae_111M / vitmae_650M) loads that class's weights by
     name into this architecture: every attention/embedding weight fails to
     match and gets randomly re-initialized, while `from_pretrained` reports
     success and returns a model that "runs" but is untrained noise. We parse
     the transformers loader log and abort if any weight was newly initialized
     or unused.
  2. Identity token order. BrainLM's ViT-MAE-style masking permutes tokens via
     `argsort(noise)` even when mask_ratio=0.0 (nothing is actually dropped,
     but the surviving order is a random permutation unless `noise` is given
     explicitly). The region x time reshape below assumes token i corresponds
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
    """Configures clean console logging streamed directly to standard output (handled by SBATCH)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("BrainLM_Cluster")

# =====================================================================
# 2. PIPELINE CORE FUNCTIONS
# =====================================================================
def load_brainlm_model(hf_repo, subfolder="old_13M"):
    """Initializes and freezes the BrainLM architecture.

    Verifies that the checkpoint at `subfolder` actually matches
    BrainLMForPretraining's parameter names -- if transformers reports any
    weight as "newly initialized" or "not used", that means the checkpoint
    and the model class disagree and the encoder would be (partly or fully)
    randomly initialized. We treat that as fatal rather than silently
    returning a garbage model.
    """
    import transformers
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining
    from brainlm_mae.configuration_brainlm import BrainLMConfig

    logging.info(f"Loading Model Configuration and Architecture from HuggingFace ({hf_repo}/{subfolder})...")
    config = BrainLMConfig.from_pretrained(hf_repo, subfolder=subfolder)

    buf = io.StringIO()

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            buf.write(self.format(record) + "\n")

    transformers.logging.set_verbosity_info()
    hf_logger = transformers.logging.get_logger("transformers.modeling_utils")
    handler = _CaptureHandler()
    hf_logger.addHandler(handler)
    try:
        model = BrainLMForPretraining.from_pretrained(hf_repo, subfolder=subfolder, config=config)
    finally:
        hf_logger.removeHandler(handler)

    log = buf.getvalue()
    clean = ("newly initialized" not in log) and ("were not used" not in log)
    logging.info(f"Clean weight load = {clean}")
    if not clean:
        raise RuntimeError(
            f"Checkpoint at subfolder={subfolder!r} does not cleanly match "
            "BrainLMForPretraining -- some weights were randomly initialized "
            "or unused. This almost always means --subfolder points at a "
            "different architecture's checkpoint (e.g. a vitmae_* subfolder). "
            "Use --subfolder old_13M. Loader log:\n" + log
        )

    model.eval()
    model.config.mask_ratio = 0.0
    model.config.output_attentions = True

    # Freeze all parameters explicitly to optimize back-end execution graph
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

def preprocess_subject_data(raw_ts, num_windows=3, window_size=200):
    """Applies Robust Scaling and sections data into sliding windows."""
    # Orient to [424, num_timepoints]
    dat_arr = raw_ts.T

    # Strip out trailing NaNs if present
    invalid_cols = np.isnan(dat_arr).any(axis=0)
    if np.any(invalid_cols):
        dat_arr = dat_arr[:, ~invalid_cols]

    num_timepoints = dat_arr.shape[1]
    if num_timepoints < window_size:
        raise ValueError(f"Recording has {num_timepoints} TRs, but requires at least {window_size}.")

    # Robust Scaler (Median & IQR)
    for idx in range(dat_arr.shape[0]):
        voxel_data = dat_arr[idx, :]
        median = np.median(voxel_data)
        q75, q25 = np.percentile(voxel_data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1e-6
        dat_arr[idx, :] = (voxel_data - median) / iqr

    # Sliding Window Generation
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
    parser = argparse.ArgumentParser(description="Production BrainLM Inference Pipeline for HPC Clusters")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to input timeseries HDF5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save individual attention matrices")
    parser.add_argument("--coords", type=str, required=True, help="Path to A424_Coordinates.dat file")
    parser.add_argument("--hf_repo", type=str, required=True, help="HuggingFace repo id or path to a local snapshot (e.g. vandijklab/brainlm)")
    parser.add_argument("--subfolder", type=str, default="old_13M",
                         help="Repo subfolder holding the checkpoint. Must be old_13M: it is the only "
                              "subfolder whose weights match the BrainLMForPretraining class used here.")
    parser.add_argument("--window_size", type=int, default=200,
                         help="TRs per window. Must be a multiple of the checkpoint's timepoint_patching_size (20 for old_13M).")
    parser.add_argument("--num_windows", type=int, default=3, help="Number of sliding windows averaged per subject")
    parser.add_argument("--ftau_lag", type=int, default=1,
                         help="Time-patch lag tau for the F_tau matrix (region x region attention from "
                              "time t to time t+tau, averaged over t). 0 = same-time (instantaneous).")
    parser.add_argument("--figures_dir", type=str, default="figures",
                         help="Where to save the full [seq_len, seq_len] last-layer attention heatmap per subject")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only iterate over the first N rows of the cohort (combined with "
                              "auto-resume, lets you extend a partial run by a small batch instead "
                              "of walking the full dataset again).")
    args = parser.parse_args()

    # Initialize your custom logger
    logger = setup_environment()
    logger.info("Initializing BrainLM Cluster Execution Sequence...")

    # Hardware Dispatch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA Device Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found. Running on CPU.")

    try:
        model, config = load_brainlm_model(args.hf_repo, args.subfolder)
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

    # Automatically create the output directory
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
    # Identity noise -> argsort(noise) is the identity permutation -> ids_restore
    # stays the identity, so token i keeps meaning (region, timepatch) = divmod(i, num_patches).
    identity_noise = (torch.arange(seq_len, dtype=torch.float32) / seq_len)

    with h5py.File(args.input_h5, 'r') as f_in:
        ts_dataset = f_in['parcel_ts']
        subject_ids = f_in['subject_ids'][:]
        n_subjects, _, _ = ts_dataset.shape
        if args.limit is not None:
            n_subjects = min(n_subjects, args.limit)

        logger.info(f"Found {n_subjects} subjects. Saving outputs as individual files to: {args.output_dir}")

        for i in range(n_subjects):
            sub_id = subject_ids[i].decode('utf-8')
            out_file = os.path.join(args.output_dir, f"{sub_id}_attention.npy")

            # --- AUTO-RESUME CHECK ---
            if os.path.exists(out_file):
                logger.info(f"Skipping [{i+1}/{n_subjects}] | Subject ID: {sub_id} (Already processed)")
                continue

            logger.info(f"Processing [{i+1}/{n_subjects}] | Subject ID: {sub_id}")

            try:
                # 1. Prepare Subject Data
                # ts_dataset[i, :, :] extracts [T, 424]
                raw_data = ts_dataset[i, :, :]

                # Apply sliding window logic, resulting in [num_windows, 424, window_size]
                fmri_windows = preprocess_subject_data(
                    raw_data, num_windows=args.num_windows, window_size=args.window_size
                ).to(device)
                num_windows = fmri_windows.shape[0]

                # Expand coordinates and identity noise to match the number of windows
                spatial_coords = spatial_coords_base.repeat(num_windows, 1, 1).to(device)
                noise = identity_noise.unsqueeze(0).repeat(num_windows, 1).to(device)

                # 2. Forward Pass
                with torch.no_grad():
                    outputs = model.vit(
                        signal_vectors=fmri_windows,
                        xyz_vectors=spatial_coords,
                        noise=noise,
                        output_attentions=True
                    )

                # Guard against silently-scrambled region/time reshapes: every window's
                # restored token order must be the identity permutation.
                ids_restore = outputs.ids_restore.cpu().numpy()
                if not np.array_equal(ids_restore, np.broadcast_to(np.arange(seq_len), ids_restore.shape)):
                    raise RuntimeError("Token order was shuffled (ids_restore != identity); refusing to reshape.")

                # 3. Extract and Process the Final Layer Attention Matrix
                # Shape: [num_windows, num_heads, sequence_length, sequence_length]
                final_layer_attention = outputs.attentions[-1]

                # Average across sliding windows (dim 0) and attention heads (dim 1)
                # Shape becomes: [seq_len+1, seq_len+1]
                avg_attention = final_layer_attention.mean(dim=(0, 1))

                # Drop the [CLS] token at index 0
                # Shape becomes: [seq_len, seq_len]
                attn_no_cls = avg_attention[1:, 1:]

                # Reshape to isolate spatial vs. temporal patch dimensions
                # Layout: [Source_Region, Source_Time, Target_Region, Target_Time]
                attn_reshaped = attn_no_cls.view(num_regions, num_patches, num_regions, num_patches)

                # Single reduction over the temporal patch axes (1 and 3)
                # Shape becomes: [424, 424]
                region_mean = attn_reshaped.mean(dim=(1, 3))

                # Move to CPU and convert to numpy for saving
                subject_mean_np = region_mean.cpu().numpy().astype('float32')

                # 4. Save to Disk
                np.save(out_file, subject_mean_np)

                # 5. Time-lagged F_tau matrix + its asymmetry score (one value per subject)
                attn_no_cls_np = attn_no_cls.cpu().numpy().astype('float32')
                ftau_matrix = compute_ftau(attn_no_cls_np, num_regions, num_patches, tau=args.ftau_lag)
                np.save(os.path.join(args.output_dir, f"{sub_id}_ftau{args.ftau_lag}.npy"), ftau_matrix)
                asymmetry = ftau_asymmetry(ftau_matrix)
                asymmetry_writer.writerow([sub_id, args.ftau_lag, asymmetry])
                asymmetry_file.flush()

                # 6. Figure of the full [seq_len, seq_len] last-layer attention matrix
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
