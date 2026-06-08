import numpy as np


def extract_features(sample):
    features = []
    feature_names = []

    signal_order = sorted(sample["input"].keys())

    for sig_name in signal_order:
        values = sample["input"][sig_name]["values"].copy()

        if values.ndim == 1:
            time_series = values
        elif values.ndim == 2:
            time_series = values.mean(axis=0)
        elif values.ndim == 3:
            time_series = values.mean(axis=(0, 1))

        # Replace NaNs before computing stats
        time_series = np.nan_to_num(time_series, nan=0.0)

        n = len(time_series)
        zero_frac = (time_series == 0).mean()

        mean = time_series.mean()
        std = time_series.std()
        minimum = time_series.min()
        maximum = time_series.max()
        median = np.median(time_series)
        first = time_series[0]
        last = time_series[-1]

        if n > 1 and std > 0:
            t = np.arange(n)
            slope = np.polyfit(t, time_series, 1)[0]
        else:
            slope = 0.0

        feats = [mean, std, minimum, maximum, median,
                 first, last, slope, zero_frac]
        features.extend(feats)

        for stat in ["mean", "std", "min", "max", "median",
                     "first", "last", "slope", "zero_frac"]:
            feature_names.append(f"{sig_name}__{stat}")

    actuator_order = sorted(sample["actuator"].keys())
    for sig_name in actuator_order:
        values = sample["actuator"][sig_name]["values"].copy()

        if values.ndim == 1:
            time_series = values
        elif values.ndim == 2:
            time_series = values.mean(axis=0)

        time_series = np.nan_to_num(time_series, nan=0.0)

        mean = time_series.mean()
        std = time_series.std()
        first = time_series[0]
        last = time_series[-1]

        features.extend([mean, std, first, last])
        for stat in ["mean", "std", "first", "last"]:
            feature_names.append(f"{sig_name}__{stat}")

    X = np.array(features, dtype=np.float32)
    y = float(sample["output"]["summary-ip"]["values"].mean())

    return X, y, feature_names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/tokamark/src")
    from data_loader import get_dataset, get_splits

    train, val, test = get_splits()
    ds = get_dataset(train[:2])
    sample = next(iter(ds))

    X, y, names = extract_features(sample)
    print(f"Feature vector length: {X.shape[0]}")
    print(f"Target: {y:.4f}")
    print(f"\nFirst 10 feature names:")
    for n in names[:10]:
        print(f"  {n}")