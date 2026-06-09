"""
Statistical feature extraction for XGBoost input.

Converts variable-length multivariate time series windows into fixed-length
feature vectors by computing summary statistics per signal. Input signals
contribute 9 features each (mean, std, min, max, median, first, last, slope,
zero_frac). Actuator signals contribute 4 features each (mean, std, first, last).

Total feature vector length: 14 input × 9 + 4 actuator × 4 = 142 features.
"""

import numpy as np


def extract_features(sample):
    """
    Extract a fixed-length feature vector from a TokaMark sample dict.

    For each input signal, averages across spatial dimensions to obtain a
    1D time series, then computes 9 summary statistics. For each actuator
    signal, computes 4 summary statistics. NaNs are replaced with 0 before
    computation.

    Returns:
        X: np.ndarray of shape (142,) — feature vector
        y: float — target value (mean of output plasma current window)
        feature_names: list of str — name for each feature in X
    """
    features = []
    feature_names = []

    for sig_name in sorted(sample["input"].keys()):
        values = sample["input"][sig_name]["values"].copy()

        if values.ndim == 1:
            ts = values
        elif values.ndim == 2:
            ts = values.mean(axis=0)
        elif values.ndim == 3:
            ts = values.mean(axis=(0, 1))

        ts = np.nan_to_num(ts, nan=0.0)
        n = len(ts)

        mean   = ts.mean()
        std    = ts.std()
        slope  = float(np.polyfit(np.arange(n), ts, 1)[0]) \
                 if n > 1 and std > 0 else 0.0

        features.extend([
            mean, std, ts.min(), ts.max(),
            float(np.median(ts)), ts[0], ts[-1],
            slope, float((ts == 0).mean())
        ])
        for stat in ["mean", "std", "min", "max", "median",
                     "first", "last", "slope", "zero_frac"]:
            feature_names.append(f"{sig_name}__{stat}")

    for sig_name in sorted(sample["actuator"].keys()):
        values = sample["actuator"][sig_name]["values"].copy()
        ts = values.mean(axis=0) if values.ndim == 2 else values
        ts = np.nan_to_num(ts, nan=0.0)

        features.extend([ts.mean(), ts.std(), ts[0], ts[-1]])
        for stat in ["mean", "std", "first", "last"]:
            feature_names.append(f"{sig_name}__{stat}")

    X = np.array(features, dtype=np.float32)
    y = float(sample["output"]["summary-ip"]["values"].mean())

    return X, y, feature_names