import sys
from config import TOKAMARK_SRC, STORE_SETTINGS, TASK_NAME
sys.path.insert(0, TOKAMARK_SRC)

from tokamark.tasks import get_task_config, get_task_metadata
from tokamark.data_split import get_train_test_val_shots
from tokamark.data import initialize_MAST_dataset, initialize_TokaMark_dataset

def load_saved_data(split_name, data_dir="/workspace/fusion_research/data"):
    import numpy as np
    import json
    
    X_feat = np.load(f"{data_dir}/{split_name}_X_feat.npy")
    X_ts = np.load(f"{data_dir}/{split_name}_X_ts.npy")
    y = np.load(f"{data_dir}/{split_name}_y.npy")
    
    with open(f"{data_dir}/feature_names.json") as f:
        feature_names = json.load(f)
    
    return X_feat, X_ts, y, feature_names


def load_test_samples(data_dir="/workspace/fusion_research/data"):
    import pickle
    with open(f"{data_dir}/test_raw_samples.pkl", "rb") as f:
        return pickle.load(f)

def get_dataset(shots_list, verbose=False):
    """
    Load a TokaMark dataset for a given list of shot IDs.
    Returns an iterable TokaMarkDataset.
    """
    config = get_task_config(TASK_NAME)
    task_metadata = get_task_metadata(config, verbose=verbose)

    mast_dataset = initialize_MAST_dataset(
        config_task=config,
        shots_list=shots_list,
        local_flag=False,
        use_std_scaling=True,
        remove_outliers=True,
        store_manager_settings=STORE_SETTINGS
    )

    tokamark_ds = initialize_TokaMark_dataset(
        dataset=mast_dataset,
        task_metadata=task_metadata,
        config_metadata=config,
        custom_transform=None
    )

    return tokamark_ds


def get_splits():
    train, test, val = get_train_test_val_shots()
    return train, val, test


def flatten_sample(sample):
    """
    Flatten a sample dict into a single 1D numpy feature vector for XGBoost.
    Concatenates all input signal values in a consistent order.
    Also returns the output (target) value.
    """
    import numpy as np

    features = []
    signal_order = sorted(sample["input"].keys())

    for sig_name in signal_order:
        values = sample["input"][sig_name]["values"]
        features.append(values.flatten())

    # Also include actuator signals
    actuator_order = sorted(sample["actuator"].keys())
    for sig_name in actuator_order:
        values = sample["actuator"][sig_name]["values"]
        features.append(values.flatten())

    X = np.concatenate(features)

    # Target: mean of output plasma current window
    y = sample["output"]["summary-ip"]["values"].mean()

    return X, y


if __name__ == "__main__":
    train, val, test = get_splits()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Quick test with 2 shots
    ds = get_dataset(train[:2])
    sample = next(iter(ds))
    X, y = flatten_sample(sample)
    print(f"Feature vector length: {X.shape}")
    print(f"Target value: {y:.4f}")