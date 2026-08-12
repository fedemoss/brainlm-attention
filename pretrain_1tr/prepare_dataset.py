"""
Build HuggingFace arrow datasets for the 1-TR-patch BrainLM pretraining run, from the
already-parcellated (A424, 424 regions) HCP-style cohort in `fmri_timeseries.h5`
(`parcel_ts` [N subjects, T timepoints, 424 regions]).

Output layout (matches what train.py / BrainLM/convert_ukbiobank_to_arrow.py expect):

    <out_dir>/train/          arrow Dataset, one row per training subject
    <out_dir>/val/            arrow Dataset, one row per validation subject
    <out_dir>/test/           arrow Dataset, one row per held-out subject (not used by train.py)
    <out_dir>/coords/         arrow Dataset, one row per region, columns X, Y, Z

Each subject row stores the FULL recording as `[T, 424]` (timepoints-major, matching
`convert_ukbiobank_to_arrow.py`'s convention and what train.py's `preprocess_fmri`
expects before it slices a random `num_timepoints_per_voxel`-length window at train
time). We do not pre-window here -- letting train.py pick a random start index per
epoch is the model's existing data-augmentation strategy and needs no changes.

Normalization: per-region (voxel) robust scaling across each subject's full recording,
`(x - median) / IQR`, computed independently per subject -- this is the exact formula
`run_model.py` uses (see its `preprocess_subject_data`) and matches train.py's default
`recording_col_name`, "Voxelwise_RobustScaler_Normalized_Recording". Matching it here
matters because we warm-start from the `old_13M` checkpoint (see
transplant_checkpoint.py), which expects inputs on this scale.

Split: subjects sorted by ID, 80/10/10 train/val/test -- same convention as
`convert_ukbiobank_to_arrow.py`.
"""
import argparse
import os
from math import ceil

import h5py
import numpy as np
from datasets import Dataset


RECORDING_COL_NAME = "Voxelwise_RobustScaler_Normalized_Recording"


def robust_scale_subject(recording):
    """recording: [T, num_regions]. Per-region (column) (x - median) / IQR."""
    out = recording.astype(np.float32).copy()
    for region_idx in range(out.shape[1]):
        col = out[:, region_idx]
        median = np.median(col)
        q75, q25 = np.percentile(col, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1e-6
        out[:, region_idx] = (col - median) / iqr
    return out


def build_split(parcel_ts, subject_ids, indices, split_name, out_dir):
    dataset_dict = {
        "Raw_Recording": [],
        RECORDING_COL_NAME: [],
        "Subject_ID": [],
    }
    for i in indices:
        raw = np.asarray(parcel_ts[i], dtype=np.float32)  # [T, 424]
        dataset_dict["Raw_Recording"].append(raw)
        dataset_dict[RECORDING_COL_NAME].append(robust_scale_subject(raw))
        sid = subject_ids[i]
        dataset_dict["Subject_ID"].append(
            sid.decode() if isinstance(sid, (bytes, bytearray)) else str(sid)
        )

    ds = Dataset.from_dict(dataset_dict)
    save_path = os.path.join(out_dir, split_name)
    ds.save_to_disk(save_path)
    print(f"[{split_name}] {len(indices)} subjects -> {save_path}")


def build_coords(coords_path, out_dir):
    coords = np.loadtxt(coords_path)  # columns: Index, X, Y, Z
    coords = coords[np.argsort(coords[:, 0])]  # ensure row i == region i
    ds = Dataset.from_dict(
        {
            "X": coords[:, 1].astype(np.float32),
            "Y": coords[:, 2].astype(np.float32),
            "Z": coords[:, 3].astype(np.float32),
        }
    )
    save_path = os.path.join(out_dir, "coords")
    ds.save_to_disk(save_path)
    print(f"[coords] {len(coords)} regions -> {save_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_h5", required=True, help="Path to fmri_timeseries.h5")
    ap.add_argument("--coords", required=True, help="Path to A424_Coordinates.dat")
    ap.add_argument("--out_dir", required=True, help="Where to write train/val/test/coords arrow datasets")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.input_h5, "r") as f:
        n_subjects = f["parcel_ts"].shape[0]
        n_regions = f["parcel_ts"].shape[2]
        subject_ids = f["subject_ids"][:] if "subject_ids" in f else np.arange(n_subjects)

        rng = np.random.default_rng(args.seed)
        order = rng.permutation(n_subjects)  # shuffle once, deterministically, before splitting
        train_end = ceil(n_subjects * args.train_frac)
        val_end = ceil(n_subjects * (args.train_frac + args.val_frac))
        train_idx, val_idx, test_idx = order[:train_end], order[train_end:val_end], order[val_end:]
        print(
            f"[split] {n_subjects} subjects, {n_regions} regions -> "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

        parcel_ts = f["parcel_ts"]  # h5py Dataset, index lazily below
        build_split(parcel_ts, subject_ids, sorted(train_idx), "train", args.out_dir)
        build_split(parcel_ts, subject_ids, sorted(val_idx), "val", args.out_dir)
        build_split(parcel_ts, subject_ids, sorted(test_idx), "test", args.out_dir)

    build_coords(args.coords, args.out_dir)
    print("[done]")


if __name__ == "__main__":
    main()
