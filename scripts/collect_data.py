"""
Run once to collect and save TokaMark data to disk.
Streams shots from UKAEA S3, extracts features and time series,
and saves numpy arrays used by all training scripts.

Output files (saved to DATA_DIR):
    train/val/test_X_feat.npy  — (N, 142) feature vectors for XGBoost
    train/val/test_X_ts.npy    — (N, 600, 18) time series for LSTM/Transformer
    train/val/test_y.npy       — (N,) target values
    test_raw_samples.pkl       — raw sample dicts for corruption experiments
    feature_names.json         — feature name strings for X_feat columns
"""

import sys
import os
import json
import pickle
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/workspace/tokamark/src")

from config import RANDOM_SEED, STORE_SETTINGS, TASK_NAME
from data_loader import get_splits, get_dataset
from feature_engineering import extract_features

DATA_DIR = "/workspace/fusion_research/data"
os.makedirs(DATA_DIR, exist_ok=True)

TARGET_T = 600       # fixed timesteps for LSTM/Transformer input
N_TRAIN_SHOTS = 200
N_VAL_SHOTS = 50
N_TEST_SHOTS = 50
MAX_SAMPLES_PER_SHOT = 50


def sample_to_timeseries(sample):
    """
    Convert a TokaMark sample dict to a fixed-length (TARGET_T, F) array.

    Each signal is averaged across spatial dimensions to a scalar time series,
    then resampled to TARGET_T timesteps via linear interpolation. Signals are
    stacked in sorted order matching CATEGORY_CHANNEL_INDICES in corruption.py.

    Returns:
        X: np.ndarray of shape (TARGET_T, F) — float32
        y: float — target value (mean of output plasma current window)
    """
    channels = []

    for sig_name in sorted(sample["input"].keys()):
        values = sample["input"][sig_name]["values"].copy()
        if values.ndim == 1:
            ts = values
        elif values.ndim == 2:
            ts = values.mean(axis=0)
        elif values.ndim == 3:
            ts = values.mean(axis=(0, 1))
        if len(ts) != TARGET_T:
            ts = np.interp(
                np.linspace(0, 1, TARGET_T),
                np.linspace(0, 1, len(ts)),
                ts
            )
        channels.append(ts)

    for sig_name in sorted(sample["actuator"].keys()):
        values = sample["actuator"][sig_name]["values"].copy()
        ts = values.mean(axis=0) if values.ndim == 2 else values
        if len(ts) != TARGET_T:
            ts = np.interp(
                np.linspace(0, 1, TARGET_T),
                np.linspace(0, 1, len(ts)),
                ts
            )
        channels.append(ts)

    X = np.stack(channels, axis=0).T.astype(np.float32)  # (T, F)
    y = float(sample["output"]["summary-ip"]["values"].mean())
    return X, y


def collect_split(shots_list, split_name):
    """
    Stream shots from S3, extract features and time series, save to disk.

    Collects up to MAX_SAMPLES_PER_SHOT windows per shot, up to
    MAX_TOTAL = len(shots_list) * MAX_SAMPLES_PER_SHOT total windows.
    Saves a checkpoint to disk every 1000 samples so progress is preserved
    if the process is interrupted.

    Raw sample dicts are saved only for the test split — they are needed
    by the corruption experiment framework.
    """
    X_feat_list, X_ts_list, y_list, raw_samples = [], [], [], []
    feat_names_saved = []
    shot_sample_count = {}
    MAX_TOTAL = len(shots_list) * MAX_SAMPLES_PER_SHOT

    ds = get_dataset(shots_list)
    pbar = tqdm(total=MAX_TOTAL, desc=f"Collecting {split_name}")

    for sample in ds:
        shot_id = sample["shot_id"]
        if shot_sample_count.get(shot_id, 0) >= MAX_SAMPLES_PER_SHOT:
            continue

        try:
            X_feat, y_feat, names = extract_features(sample)
            if np.isnan(y_feat):
                continue

            X_ts, _ = sample_to_timeseries(sample)

            # Replace NaNs with zeros — natural NaNs are front-loaded
            # acquisition gaps characterized in the paper (Section 3)
            X_feat = np.nan_to_num(X_feat, nan=0.0)
            X_ts   = np.nan_to_num(X_ts,   nan=0.0)

            X_feat_list.append(X_feat)
            X_ts_list.append(X_ts)
            y_list.append(y_feat)

            if split_name == "test":
                raw_samples.append(sample)
            if not feat_names_saved:
                feat_names_saved = names

            shot_sample_count[shot_id] = shot_sample_count.get(shot_id, 0) + 1
            pbar.update(1)

            # Checkpoint every 1000 samples
            if len(X_feat_list) % 1000 == 0:
                _save_arrays(split_name, X_feat_list, X_ts_list, y_list,
                             raw_samples, feat_names_saved)
                pbar.write(f"  Checkpoint: {len(X_feat_list)} samples saved")

            if len(X_feat_list) >= MAX_TOTAL:
                break

        except Exception:
            continue

    pbar.close()
    _save_arrays(split_name, X_feat_list, X_ts_list, y_list,
                 raw_samples, feat_names_saved)

    X_feat = np.array(X_feat_list, dtype=np.float32)
    X_ts   = np.array(X_ts_list,   dtype=np.float32)
    y      = np.array(y_list,       dtype=np.float32)

    print(f"\n  {split_name}: {X_feat.shape[0]} windows")
    print(f"  X_feat: {X_feat.shape}  X_ts: {X_ts.shape}  y: {y.shape}")
    return X_feat, X_ts, y


def _save_arrays(split_name, X_feat_list, X_ts_list, y_list,
                 raw_samples, feat_names_saved):
    """Save current collected arrays to disk."""
    np.save(f"{DATA_DIR}/{split_name}_X_feat.npy",
            np.array(X_feat_list, dtype=np.float32))
    np.save(f"{DATA_DIR}/{split_name}_X_ts.npy",
            np.array(X_ts_list, dtype=np.float32))
    np.save(f"{DATA_DIR}/{split_name}_y.npy",
            np.array(y_list, dtype=np.float32))

    if split_name == "train" and feat_names_saved:
        with open(f"{DATA_DIR}/feature_names.json", "w") as f:
            json.dump(feat_names_saved, f)

    if split_name == "test" and raw_samples:
        with open(f"{DATA_DIR}/test_raw_samples.pkl", "wb") as f:
            pickle.dump(raw_samples, f)


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    train_shots, val_shots, test_shots = get_splits()

    print(f"Collecting {N_TRAIN_SHOTS} train / "
          f"{N_VAL_SHOTS} val / {N_TEST_SHOTS} test shots")
    print(f"Saving to {DATA_DIR}\n")

    collect_split(train_shots[:N_TRAIN_SHOTS], "train")
    collect_split(val_shots[:N_VAL_SHOTS],     "val")
    collect_split(test_shots[:N_TEST_SHOTS],   "test")

    print("\nAll data saved. Files:")
    for fname in sorted(os.listdir(DATA_DIR)):
        size = os.path.getsize(f"{DATA_DIR}/{fname}") / 1024 / 1024
        print(f"  {fname}: {size:.1f} MB")