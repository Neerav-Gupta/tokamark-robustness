"""
Corruption and mitigation functions for the TokaMark robustness benchmark.

Two sets of functions are provided:
- corrupt_ts_* / apply_mitigation_ts: operate on pre-built (N, T, F) numpy
  arrays loaded from disk. Used by LSTM and Transformer experiments.
- corrupt_feat_* (defined inline in train_xgboost.py): operate on (N, 142)
  feature arrays. Used by XGBoost experiments.

Signal order constants (INPUT_SIGNAL_ORDER, CATEGORY_CHANNEL_INDICES) define
the mapping between channel indices in X_ts and physical diagnostic signals.
"""

import numpy as np


# ─────────────────────────────────────────
# Signal order constants
# Must match the sorted() order used in collect_data.py
# ─────────────────────────────────────────

INPUT_SIGNAL_ORDER = [
    "interferometer-n_e_line",                                 # 0
    "magnetics-b_field_pol_probe_ccbv_field",                  # 1
    "magnetics-b_field_pol_probe_obr_field",                   # 2
    "magnetics-b_field_pol_probe_obv_field",                   # 3
    "magnetics-b_field_pol_probe_omv_voltage",                 # 4
    "magnetics-b_field_tor_probe_cc_field",                    # 5
    "magnetics-b_field_tor_probe_saddle_voltage",              # 6
    "magnetics-flux_loop_flux",                                # 7
    "pf_active-coil_current",                                  # 8
    "pf_active-solenoid_current",                              # 9
    "soft_x_rays-horizontal_cam_lower",                        # 10
    "soft_x_rays-horizontal_cam_upper",                        # 11
    "spectrometer_visible-filter_spectrometer_dalpha_voltage", # 12
    "summary-ip",                                              # 13
]

ACTUATOR_SIGNAL_ORDER = [
    "gas_injection-total_injected",  # 14
    "pulse_schedule-i_plasma",       # 15
    "pulse_schedule-n_e_line",       # 16
    "summary-power_nbi",             # 17
]

ALL_SIGNAL_ORDER = INPUT_SIGNAL_ORDER + ACTUATOR_SIGNAL_ORDER

# Channel index groups for per-category ablation and correlated failure.
# Indices correspond to positions in ALL_SIGNAL_ORDER.
# *_correlated keys are used for Scenario 5 (correlated group failure).
CATEGORY_CHANNEL_INDICES = {
    "magnetics_flux":              [7],
    "magnetics_pickup":            [1, 2, 3],
    "magnetics_saddle":            [6],
    "mirnov":                      [4, 5],
    "kinetics":                    [0, 12],
    "radiatives":                  [10, 11],
    "active_coils":                [8, 9],
    "plasma_current":              [13],
    "kinetics_correlated":         [0, 12],
    "magnetics_active_correlated": [1, 2, 3, 6],
    "radiatives_correlated":       [10, 11, 12],
    "mirnov_correlated":           [4, 5],
}


# ─────────────────────────────────────────
# Vectorized corruption functions
# All operate on X_ts arrays of shape (N, T, F)
# Return (corrupted_array, boolean_mask) where mask is True
# at positions that were artificially zeroed.
# ─────────────────────────────────────────

def corrupt_ts_random_dropout(X_ts, drop_rate=0.1, rng=None):
    """
    Randomly zeros drop_rate fraction of values across all channels.
    Only zeros positions that were non-zero to avoid double-masking
    natural zeros. Simulates random sensor glitches.
    """
    if rng is None:
        rng = np.random.default_rng()
    c = X_ts.copy()
    mask = (rng.random(c.shape) < drop_rate) & (X_ts != 0)
    c[mask] = 0.0
    return c, mask


def corrupt_ts_channel_ablation(X_ts, channel_indices):
    """
    Zeros out entire channels (feature dimensions) for all samples
    and timesteps. Simulates a completely dead or disconnected diagnostic.
    channel_indices: list of indices into the F dimension of X_ts.
    """
    c = X_ts.copy()
    mask = np.zeros(c.shape, dtype=bool)
    for idx in channel_indices:
        if idx < X_ts.shape[2]:
            mask[:, :, idx] = X_ts[:, :, idx] != 0
            c[:, :, idx] = 0.0
    return c, mask


def corrupt_ts_temporal_gap(X_ts, gap_fraction=0.2, gap_position="front", rng=None):
    """
    Zeros out a contiguous temporal block across all channels and samples.
    gap_position:
        'front'     — zeros first gap_fraction of window (acquisition delay)
        'pre_event' — zeros final gap_fraction of window (worst case)
        'random'    — zeros a randomly positioned block
    Simulates acquisition window failures or late diagnostic startup.
    """
    if rng is None:
        rng = np.random.default_rng()
    c = X_ts.copy()
    T = X_ts.shape[1]
    gap_size = int(T * gap_fraction)
    if gap_position == "front":
        start = 0
    elif gap_position == "pre_event":
        start = T - gap_size
    else:
        start = int(rng.integers(0, max(1, T - gap_size)))
    end = start + gap_size
    mask = np.zeros(c.shape, dtype=bool)
    mask[:, start:end, :] = X_ts[:, start:end, :] != 0
    c[:, start:end, :] = 0.0
    return c, mask


# ─────────────────────────────────────────
# Mitigation function
# ─────────────────────────────────────────

def apply_mitigation_ts(X_ts_corrupted, mask, strategy="mean_fill", channel_means=None):
    """
    Apply imputation to a corrupted (N, T, F) array using the corruption mask.
    Only fills positions marked True in mask (artificially zeroed values).

    Strategies:
        zero_fill    — no-op, returns input unchanged
        mean_fill    — replaces masked values with per-channel training mean
        forward_fill — carries last valid observation forward in time
    """
    if strategy == "zero_fill":
        return X_ts_corrupted
    result = X_ts_corrupted.copy()
    if strategy == "mean_fill" and channel_means is not None:
        for f_idx, mean_val in enumerate(channel_means):
            ch_mask = mask[:, :, f_idx]
            result[:, :, f_idx][ch_mask] = mean_val
    elif strategy == "forward_fill":
        for t in range(1, result.shape[1]):
            m = mask[:, t, :]
            if m.any():
                result[:, t, :][m] = result[:, t - 1, :][m]
    return result