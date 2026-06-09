"""
Data loading utilities for the TokaMark robustness benchmark.

Provides two loading modes:
- From disk (load_saved_data, load_test_samples): loads pre-collected numpy
  arrays. Used by all training and evaluation scripts.
- From S3 (get_dataset): streams directly from UKAEA S3 via TokaMark loader.
  Used only by collect_data.py to regenerate the saved arrays.
"""

import sys
import json
import pickle
import numpy as np

from config import TOKAMARK_SRC, STORE_SETTINGS, TASK_NAME
sys.path.insert(0, TOKAMARK_SRC)

from tokamark.tasks import get_task_config, get_task_metadata
from tokamark.data_split import get_train_test_val_shots
from tokamark.data import initialize_MAST_dataset, initialize_TokaMark_dataset


def load_saved_data(split_name, data_dir="/workspace/fusion_research/data"):
    """
    Load pre-collected arrays for a given split from disk.

    Returns:
        X_feat: (N, 142) feature vectors for XGBoost
        X_ts:   (N, 600, 18) time series tensors for LSTM/Transformer
        y:      (N,) target values
        feature_names: list of 142 feature name strings
    """
    X_feat = np.load(f"{data_dir}/{split_name}_X_feat.npy")
    X_ts   = np.load(f"{data_dir}/{split_name}_X_ts.npy")
    y      = np.load(f"{data_dir}/{split_name}_y.npy")

    with open(f"{data_dir}/feature_names.json") as f:
        feature_names = json.load(f)

    return X_feat, X_ts, y, feature_names


def load_test_samples(data_dir="/workspace/fusion_research/data"):
    """
    Load raw test sample dicts from disk.
    Used by corruption experiments that need the original signal structure.
    """
    with open(f"{data_dir}/test_raw_samples.pkl", "rb") as f:
        return pickle.load(f)


def get_dataset(shots_list, verbose=False):
    """
    Stream a TokaMark dataset for a given list of shot IDs from UKAEA S3.
    Returns an iterable TokaMark dataset of windowed samples.
    Only used by collect_data.py — training scripts load from disk instead.
    """
    config        = get_task_config(TASK_NAME)
    task_metadata = get_task_metadata(config, verbose=verbose)

    mast_dataset = initialize_MAST_dataset(
        config_task=config,
        shots_list=shots_list,
        local_flag=False,
        use_std_scaling=True,
        remove_outliers=True,
        store_manager_settings=STORE_SETTINGS
    )

    return initialize_TokaMark_dataset(
        dataset=mast_dataset,
        task_metadata=task_metadata,
        config_metadata=config,
        custom_transform=None
    )


def get_splits():
    """Return (train, val, test) shot ID lists using TokaMark's official split."""
    train, test, val = get_train_test_val_shots()
    return train, val, test