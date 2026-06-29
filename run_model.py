#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import h5py
import numpy as np
import torch

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
def load_brainlm_model(hf_repo, subfolder):
    """Initializes and freezes the BrainLM architecture."""
    logging.info("Loading Model Configuration and Architecture from HuggingFace...")
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining
    from brainlm_mae.configuration_brainlm import BrainLMConfig
    
    config = BrainLMConfig.from_pretrained(hf_repo, subfolder=subfolder)
    model = BrainLMForPretraining.from_pretrained(hf_repo, subfolder=subfolder, config=config)
    
    model.eval()
    model.config.mask_ratio = 0.0 
    
    # Freeze all parameters explicitly to optimize back-end execution graph
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def load_spatial_coordinates(coords_file):
    """Loads MNI spatial coordinates and prepares tensor format."""
    logging.info(f"Loading spatial coordinates from: {coords_file}")
    if not os.path.exists(coords_file):
        raise FileNotFoundError(f"Coordinates file missing at: {coords_file}")
    
    real_coords = np.loadtxt(coords_file)[:, 1:4] 
    return torch.tensor(real_coords, dtype=torch.float32).unsqueeze(0)

def preprocess_subject_data(raw_ts, num_windows=3, window_size=490):
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
    parser.add_argument("--hf_repo", type=str, required=True, help="Path to local HuggingFace repo")
    parser.add_argument("--subfolder", type=str, default="vitmae_111M", help="Repo subfolder model target")
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
        model = load_brainlm_model(args.hf_repo, args.subfolder).to(device)
        model.eval()
        model.config.mask_ratio = 0.0 # Disable masking for full-brain view
        
        spatial_coords_base = load_spatial_coordinates(args.coords)
    except Exception as e:
        logger.critical(f"Failed to initialize core infrastructure: {str(e)}")
        sys.exit(1)

    # Automatically create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.input_h5, 'r') as f_in:
        ts_dataset = f_in['parcel_ts']
        subject_ids = f_in['subject_ids'][:]
        n_subjects, _, _ = ts_dataset.shape
        
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
                # ts_dataset[i, :, :] extracts [1200, 424]
                raw_data = ts_dataset[i, :, :] 
                
                # Apply sliding window logic, resulting in [num_windows, 424, 490]
                fmri_windows = preprocess_subject_data(raw_data).to(device) 
                num_windows = fmri_windows.shape[0]
                
                # Expand coordinates to match the number of windows
                spatial_coords = spatial_coords_base.repeat(num_windows, 1, 1).to(device)
                
                # 2. Forward Pass
                with torch.no_grad():
                    outputs = model.vit(
                        signal_vectors=fmri_windows,
                        xyz_vectors=spatial_coords,
                        output_attentions=True
                    )
                
                # 3. Extract and Process the Final Layer Attention Matrix
                # Shape: [num_windows, num_heads, sequence_length, sequence_length]
                final_layer_attention = outputs.attentions[-1]

                # Average across sliding windows (dim 0) and attention heads (dim 1)
                # Shape becomes: [4241, 4241]
                avg_attention = final_layer_attention.mean(dim=(0, 1))

                # Drop the [CLS] token at index 0
                # Shape becomes: [4240, 4240]
                attn_no_cls = avg_attention[1:, 1:]

                num_regions = 424
                # Calculate time patches dynamically based on stripped sequence length
                num_patches = attn_no_cls.shape[0] // num_regions  # 10

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
                
            except Exception as sub_error:
                logger.error(f"SKIPPED Subject {sub_id} due to processing error: {str(sub_error)}")
                continue
                
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info("All operations complete.")

if __name__ == "__main__":
    main()