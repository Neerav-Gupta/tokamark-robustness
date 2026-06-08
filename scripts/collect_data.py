"""
Run this ONCE to collect and save all data to disk.
All models load from these saved files — no re-streaming needed.
"""
import sys
import os
import numpy as np

sys.path.insert(0, "/workspace/tokamark/src")

from config import RANDOM_SEED, RESULTS_DIR
from data_loader import get_splits, get_dataset
from feature_engineering import extract_features

from tokamark.tasks import get_task_config, get_task_metadata
from tokamark.data import initialize_MAST_dataset, initialize_TokaMark_dataset
from config import STORE_SETTINGS, TASK_NAME

DATA_DIR = "/workspace/fusion_research/data"
os.makedirs(DATA_DIR, exist_ok=True)

TARGET_T = 600  # fixed timesteps for LSTM/Transformer

N_TRAIN_SHOTS = 200
N_VAL_SHOTS = 50
N_TEST_SHOTS = 50
MAX_SAMPLES_PER_SHOT = 50


def sample_to_timeseries(sample):
    """
    Convert sample to fixed-length time series tensor (T, F).
    Used by LSTM and Transformer.
    """
    signal_order = sorted(sample["input"].keys())
    actuator_order = sorted(sample["actuator"].keys())
    channels = []

    for sig_name in signal_order:
        values = sample["input"][sig_name]["values"].copy()
        if values.ndim == 1:
            ts = values
        elif values.ndim == 2:
            ts = values.mean(axis=0)
        elif values.ndim == 3:
            ts = values.mean(axis=(0, 1))
        if len(ts) != TARGET_T:
            x_old = np.linspace(0, 1, len(ts))
            x_new = np.linspace(0, 1, TARGET_T)
            ts = np.interp(x_new, x_old, ts)
        channels.append(ts)

    for sig_name in actuator_order:
        values = sample["actuator"][sig_name]["values"].copy()
        if values.ndim == 1:
            ts = values
        elif values.ndim == 2:
            ts = values.mean(axis=0)
        if len(ts) != TARGET_T:
            x_old = np.linspace(0, 1, len(ts))
            x_new = np.linspace(0, 1, TARGET_T)
            ts = np.interp(x_new, x_old, ts)
        channels.append(ts)

    X = np.stack(channels, axis=0).T.astype(np.float32)  # (T, F)
    y = float(sample["output"]["summary-ip"]["values"].mean())
    return X, y


def collect_split(shots_list, split_name):
    from tqdm import tqdm

    X_feat_list = []
    X_ts_list = []
    y_list = []
    raw_samples = []
    feat_names_saved = []

    shot_sample_count = {}
    MAX_TOTAL = len(shots_list) * MAX_SAMPLES_PER_SHOT

    ds = get_dataset(shots_list)
    pbar = tqdm(total=MAX_TOTAL, desc=f"Collecting {split_name}")

    for sample in ds:
        shot_id = sample["shot_id"]

        # Skip if this shot is capped
        if shot_sample_count.get(shot_id, 0) >= MAX_SAMPLES_PER_SHOT:
            continue

        try:
            X_feat, y_feat, names = extract_features(sample)

            # Skip only if target is NaN
            if np.isnan(y_feat):
                continue

            X_ts, _ = sample_to_timeseries(sample)

            # Replace NaNs with zeros — matches TokaMark baseline behavior
            # Natural NaNs are front-loaded acquisition gaps (documented in paper)
            X_feat = np.nan_to_num(X_feat, nan=0.0)
            X_ts = np.nan_to_num(X_ts, nan=0.0)

            X_feat_list.append(X_feat)
            X_ts_list.append(X_ts)
            y_list.append(y_feat)

            if split_name == "test":
                raw_samples.append(sample)

            if not feat_names_saved:
                feat_names_saved = names

            shot_sample_count[shot_id] = shot_sample_count.get(shot_id, 0) + 1
            pbar.update(1)

            # Save checkpoint every 1000 valid samples
            if len(X_feat_list) % 1000 == 0:
                np.save(f"{DATA_DIR}/{split_name}_X_feat.npy",
                        np.array(X_feat_list, dtype=np.float32))
                np.save(f"{DATA_DIR}/{split_name}_X_ts.npy",
                        np.array(X_ts_list, dtype=np.float32))
                np.save(f"{DATA_DIR}/{split_name}_y.npy",
                        np.array(y_list, dtype=np.float32))
                if split_name == "test" and raw_samples:
                    import pickle
                    with open(f"{DATA_DIR}/test_raw_samples.pkl", "wb") as f:
                        pickle.dump(raw_samples, f)
                if split_name == "train" and feat_names_saved:
                    import json
                    with open(f"{DATA_DIR}/feature_names.json", "w") as f:
                        json.dump(feat_names_saved, f)
                pbar.write(f"  Checkpoint saved: {len(X_feat_list)} samples")

            # Hard stop at MAX_TOTAL valid samples
            if len(X_feat_list) >= MAX_TOTAL:
                break

        except Exception as e:
            continue

    pbar.close()

    # Final save
    X_feat = np.array(X_feat_list, dtype=np.float32)
    X_ts = np.array(X_ts_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    np.save(f"{DATA_DIR}/{split_name}_X_feat.npy", X_feat)
    np.save(f"{DATA_DIR}/{split_name}_X_ts.npy", X_ts)
    np.save(f"{DATA_DIR}/{split_name}_y.npy", y)

    if split_name == "train" and feat_names_saved:
        import json
        with open(f"{DATA_DIR}/feature_names.json", "w") as f:
            json.dump(feat_names_saved, f)

    if split_name == "test":
        import pickle
        with open(f"{DATA_DIR}/test_raw_samples.pkl", "wb") as f:
            pickle.dump(raw_samples, f)
        print(f"  Saved {len(raw_samples)} raw test samples")

    print(f"\n  {split_name}: {X_feat.shape[0]} windows")
    print(f"  X_feat: {X_feat.shape}")
    print(f"  X_ts:   {X_ts.shape}")
    print(f"  y:      {y.shape}")

    return X_feat, X_ts, y

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    train_shots, val_shots, test_shots = get_splits()

    print(f"Collecting {N_TRAIN_SHOTS} train / "
          f"{N_VAL_SHOTS} val / {N_TEST_SHOTS} test shots")
    print(f"Saving to {DATA_DIR}\n")

    collect_split(train_shots[:N_TRAIN_SHOTS], "train")
    collect_split(val_shots[:N_VAL_SHOTS], "val")
    collect_split(test_shots[:N_TEST_SHOTS], "test")

    print("\nAll data saved. Files:")
    for f in sorted(os.listdir(DATA_DIR)):
        size = os.path.getsize(f"{DATA_DIR}/{f}") / 1024 / 1024
        print(f"  {f}: {size:.1f} MB")