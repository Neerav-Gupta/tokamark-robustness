import numpy as np
import copy


def corrupt_random_dropout(sample, drop_rate=0.1, rng=None):
    """
    Randomly zeros drop_rate fraction of all values across all input signals.
    Simulates random sensor glitches.
    """
    if rng is None:
        rng = np.random.default_rng()
    corrupted = sample
    for split in ["input", "actuator"]:
        for sig_name, sig_data in corrupted[split].items():
            values = sig_data["values"].copy()
            mask = rng.random(values.shape) < drop_rate
            values[mask] = 0.0
            corrupted[split][sig_name]["values"] = values
    return corrupted


def corrupt_channel_ablation(sample, channels_to_kill=None, n_channels=1, rng=None):
    """
    Zeros out entire signals completely.
    Simulates a dead or disconnected diagnostic.
    """
    if rng is None:
        rng = np.random.default_rng()
    corrupted = sample
    all_input_signals = list(corrupted["input"].keys())
    if channels_to_kill is None:
        channels_to_kill = rng.choice(
            all_input_signals,
            size=min(n_channels, len(all_input_signals)),
            replace=False
        ).tolist()
    killed = []
    for sig_name in channels_to_kill:
        if sig_name in corrupted["input"]:
            corrupted["input"][sig_name]["values"] = np.zeros_like(
                corrupted["input"][sig_name]["values"]
            )
            killed.append(sig_name)
    return corrupted, killed


def corrupt_temporal_gap(sample, gap_fraction=0.2, gap_position="front", rng=None):
    """
    Zeros out a contiguous block of time across all signals.
    gap_position: 'front' (acquisition delay), 'random', 'pre_event' (worst case)
    """
    if rng is None:
        rng = np.random.default_rng()
    corrupted = sample
    for split in ["input", "actuator"]:
        for sig_name, sig_data in corrupted[split].items():
            values = sig_data["values"].copy()
            n_time = values.shape[-1]
            gap_size = int(n_time * gap_fraction)
            if gap_position == "front":
                start = 0
            elif gap_position == "pre_event":
                start = n_time - gap_size
            else:
                start = int(rng.integers(0, max(1, n_time - gap_size)))
            end = start + gap_size
            if values.ndim == 1:
                values[start:end] = 0.0
            elif values.ndim == 2:
                values[:, start:end] = 0.0
            elif values.ndim == 3:
                values[:, :, start:end] = 0.0
            corrupted[split][sig_name]["values"] = values
    return corrupted


def corrupt_correlated_failure(sample, diagnostic_group="kinetics", rng=None):
    """
    Zeros out all signals from a physically related diagnostic group.
    Motivated by observed correlated NaN structure in real MAST data.
    """
    diagnostic_groups = {
        "kinetics": [
            "interferometer-n_e_line",
            "spectrometer_visible-filter_spectrometer_dalpha_voltage"
        ],
        "magnetics_active": [
            "magnetics-b_field_pol_probe_ccbv_field",
            "magnetics-b_field_pol_probe_obr_field",
            "magnetics-b_field_pol_probe_obv_field",
            "magnetics-b_field_tor_probe_saddle_voltage",
        ],
        "radiatives": [
            "soft_x_rays-horizontal_cam_lower",
            "soft_x_rays-horizontal_cam_upper",
            "spectrometer_visible-filter_spectrometer_dalpha_voltage"
        ],
        "mirnov": [
            "magnetics-b_field_tor_probe_cc_field",
            "magnetics-b_field_pol_probe_omv_voltage"
        ]
    }
    return corrupt_channel_ablation(
        sample,
        channels_to_kill=diagnostic_groups[diagnostic_group],
        rng=rng
    )


def apply_mitigation(sample, strategy="zero_fill", signal_means=None):
    """
    Apply mitigation strategy to a corrupted sample.
    Strategies: zero_fill, mean_fill, forward_fill
    """
    if strategy == "zero_fill":
        return sample

    mitigated = copy.deepcopy(sample)

    for split in ["input", "actuator"]:
        for sig_name, sig_data in mitigated[split].items():
            values = sig_data["values"].copy()
            zero_mask = (values == 0)
            if not zero_mask.any():
                continue

            if strategy == "mean_fill":
                if signal_means and sig_name in signal_means:
                    values[zero_mask] = signal_means[sig_name]

            elif strategy == "forward_fill":
                if values.ndim == 1:
                    for t in range(1, len(values)):
                        if values[t] == 0 and values[t - 1] != 0:
                            values[t] = values[t - 1]
                elif values.ndim == 2:
                    for t in range(1, values.shape[-1]):
                        mask_t = values[:, t] == 0
                        values[mask_t, t] = values[mask_t, t - 1]
                elif values.ndim == 3:
                    for t in range(1, values.shape[-1]):
                        mask_t = values[:, :, t] == 0
                        values[:, :, t][mask_t] = values[:, :, t - 1][mask_t]

            mitigated[split][sig_name]["values"] = values

    return mitigated

SCORE_BATCH_SIZE = 512

def score_raw(corrupted_samples, masks_list, mitigation="zero_fill", signal_means=None):
    X_list, y_list = [], []
    for s, masks in zip(corrupted_samples, masks_list):
        try:
            if mitigation != "zero_fill":
                s = apply_mitigation_with_mask(
                    s, masks, strategy=mitigation,
                    signal_means=signal_means)
            X, y = sample_to_tensor(s)
            X = np.nan_to_num(X, nan=0.0)
            if not np.isnan(y):
                X_list.append(X)
                y_list.append(y)
        except Exception:
            continue
    if len(X_list) < 2:
        return np.nan

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list)
    all_preds = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_arr), SCORE_BATCH_SIZE):
            batch = torch.tensor(
                X_arr[i:i+SCORE_BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            preds = model(batch).cpu().numpy()
            all_preds.extend(preds)

    return float(nrmse(y_arr, np.array(all_preds)))

def corrupt_with_mask(sample, corruption_fn, **kwargs):
    """
    Apply corruption and return corrupted sample + boolean mask.
    Uses shallow copy + numpy array copies instead of deepcopy for speed.
    """
    import copy

    # Shallow copy the top level structure
    corrupted = {
        'shot_id': sample['shot_id'],
        'window_index': sample['window_index'],
        't_cut': sample['t_cut'],
        'input': {},
        'actuator': {},
        'output': sample['output']
    }

    # Copy only the values arrays we need, not the whole object
    for split in ['input', 'actuator']:
        for sig_name, sig_data in sample[split].items():
            corrupted[split][sig_name] = {
                'time': sig_data['time'],  # don't copy time — never modified
                'values': sig_data['values'].copy()  # copy values only
            }

    # Apply corruption to the pre-copied sample
    killed = None
    result = corruption_fn(corrupted, **kwargs)
    if isinstance(result, tuple):
        corrupted, killed = result
    else:
        corrupted = result

    # Build mask: True where corruption added zeros that weren't there before
    masks = {}
    for split in ['input', 'actuator']:
        masks[split] = {}
        for sig_name in corrupted[split]:
            orig_vals = sample[split][sig_name]['values']
            corr_vals = corrupted[split][sig_name]['values']
            masks[split][sig_name] = (corr_vals == 0) & (orig_vals != 0)

    if killed is not None:
        return corrupted, masks, killed
    return corrupted, masks


def apply_mitigation_with_mask(sample, masks, strategy="mean_fill", signal_means=None):
    """
    Apply mitigation using the corruption mask to only fill artificially
    zeroed values, not natural zeros.
    """
    import copy
    import numpy as np

    if strategy == "zero_fill":
        return sample

    mitigated = copy.deepcopy(sample)

    for split in ["input", "actuator"]:
        for sig_name, sig_data in mitigated[split].items():
            values = sig_data["values"].copy()
            mask = masks[split].get(sig_name)

            if mask is None or not mask.any():
                continue

            if strategy == "mean_fill":
                fill_val = signal_means.get(sig_name, 0.0) if signal_means else 0.0
                values[mask] = fill_val

            elif strategy == "forward_fill":
                if values.ndim == 1:
                    for t in range(1, len(values)):
                        if mask[t] and not mask[t-1]:
                            values[t] = values[t-1]
                elif values.ndim == 2:
                    for t in range(1, values.shape[-1]):
                        col_mask = mask[:, t]
                        if col_mask.any():
                            values[col_mask, t] = values[col_mask, t-1]
                elif values.ndim == 3:
                    for t in range(1, values.shape[-1]):
                        col_mask = mask[:, :, t]
                        if col_mask.any():
                            values[:, :, t][col_mask] = values[:, :, t-1][col_mask]

            mitigated[split][sig_name]["values"] = values

    return mitigated

def build_sample_arrays_from_disk(test_samples, data_dir="/workspace/fusion_research/data"):
    """
    Load pre-saved fixed-length arrays from disk instead of resampling.
    X_ts shape: (N, T, F) — already fixed length from collect_data.py
    """
    import numpy as np

    X_ts = np.load(f"{data_dir}/test_X_ts.npy")   # (N, 600, 18)
    X_feat = np.load(f"{data_dir}/test_X_feat.npy") # (N, 142)
    y = np.load(f"{data_dir}/test_y.npy")            # (N,)

    return {
        "X_ts": X_ts,
        "X_feat": X_feat,
        "y": y,
        "n_samples": len(y)
    }


def corrupt_arrays_random_dropout(arrays, drop_rate=0.1, rng=None):
    """Vectorized random dropout on pre-built arrays."""
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    
    corrupted = {'input': {}, 'actuator': {}}
    masks = {'input': {}, 'actuator': {}}
    
    for split in ['input', 'actuator']:
        for sig_name, arr in arrays[split].items():
            c = arr.copy()
            mask = rng.random(c.shape) < drop_rate
            # Only mask where values aren't already zero
            mask = mask & (arr != 0)
            c[mask] = 0.0
            corrupted[split][sig_name] = c
            masks[split][sig_name] = mask
    
    corrupted['y'] = arrays['y']
    return corrupted, masks


def corrupt_arrays_channel_ablation(arrays, channels_to_kill=None, n_channels=1, rng=None):
    """Vectorized channel ablation on pre-built arrays."""
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    
    all_signals = sorted(arrays['input'].keys())
    if channels_to_kill is None:
        channels_to_kill = rng.choice(
            all_signals, size=min(n_channels, len(all_signals)), replace=False
        ).tolist()
    
    corrupted = {'input': {}, 'actuator': {}}
    masks = {'input': {}, 'actuator': {}}
    
    for split in ['input', 'actuator']:
        for sig_name, arr in arrays[split].items():
            if sig_name in channels_to_kill:
                c = np.zeros_like(arr)
                mask = arr != 0
            else:
                c = arr.copy()
                mask = np.zeros(arr.shape, dtype=bool)
            corrupted[split][sig_name] = c
            masks[split][sig_name] = mask
    
    corrupted['y'] = arrays['y']
    return corrupted, masks, channels_to_kill


def corrupt_arrays_temporal_gap(arrays, gap_fraction=0.2, gap_position='front', rng=None):
    """Vectorized temporal gap on pre-built arrays."""
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    
    corrupted = {'input': {}, 'actuator': {}}
    masks = {'input': {}, 'actuator': {}}
    
    for split in ['input', 'actuator']:
        for sig_name, arr in arrays[split].items():
            c = arr.copy()
            n_time = arr.shape[-1]
            gap_size = int(n_time * gap_fraction)
            
            if gap_position == 'front':
                start = 0
            elif gap_position == 'pre_event':
                start = n_time - gap_size
            else:
                start = int(rng.integers(0, max(1, n_time - gap_size)))
            end = start + gap_size
            
            mask = np.zeros(arr.shape, dtype=bool)
            if arr.ndim == 2:    # (N, T)
                orig_nonzero = arr[:, start:end] != 0
                c[:, start:end] = 0.0
                mask[:, start:end] = orig_nonzero
            elif arr.ndim == 3:  # (N, C, T)
                orig_nonzero = arr[:, :, start:end] != 0
                c[:, :, start:end] = 0.0
                mask[:, :, start:end] = orig_nonzero
            elif arr.ndim == 4:  # (N, C, H, T)
                orig_nonzero = arr[:, :, :, start:end] != 0
                c[:, :, :, start:end] = 0.0
                mask[:, :, :, start:end] = orig_nonzero
            
            corrupted[split][sig_name] = c
            masks[split][sig_name] = mask
    
    corrupted['y'] = arrays['y']
    return corrupted, masks


def corrupt_arrays_correlated(arrays, diagnostic_group='kinetics', rng=None):
    """Vectorized correlated failure on pre-built arrays."""
    diagnostic_groups = {
        'kinetics': [
            'interferometer-n_e_line',
            'spectrometer_visible-filter_spectrometer_dalpha_voltage'
        ],
        'magnetics_active': [
            'magnetics-b_field_pol_probe_ccbv_field',
            'magnetics-b_field_pol_probe_obr_field',
            'magnetics-b_field_pol_probe_obv_field',
            'magnetics-b_field_tor_probe_saddle_voltage',
        ],
        'radiatives': [
            'soft_x_rays-horizontal_cam_lower',
            'soft_x_rays-horizontal_cam_upper',
            'spectrometer_visible-filter_spectrometer_dalpha_voltage'
        ],
        'mirnov': [
            'magnetics-b_field_tor_probe_cc_field',
            'magnetics-b_field_pol_probe_omv_voltage'
        ]
    }
    channels = diagnostic_groups[diagnostic_group]
    return corrupt_arrays_channel_ablation(
        arrays, channels_to_kill=channels, rng=rng)
    
INPUT_SIGNAL_ORDER = [
    "interferometer-n_e_line",                                    # 0
    "magnetics-b_field_pol_probe_ccbv_field",                     # 1
    "magnetics-b_field_pol_probe_obr_field",                      # 2
    "magnetics-b_field_pol_probe_obv_field",                      # 3
    "magnetics-b_field_pol_probe_omv_voltage",                    # 4
    "magnetics-b_field_tor_probe_cc_field",                       # 5
    "magnetics-b_field_tor_probe_saddle_voltage",                 # 6
    "magnetics-flux_loop_flux",                                   # 7
    "pf_active-coil_current",                                     # 8
    "pf_active-solenoid_current",                                 # 9
    "soft_x_rays-horizontal_cam_lower",                           # 10
    "soft_x_rays-horizontal_cam_upper",                           # 11
    "spectrometer_visible-filter_spectrometer_dalpha_voltage",    # 12
    "summary-ip",                                                 # 13
]

ACTUATOR_SIGNAL_ORDER = [
    "gas_injection-total_injected",    # 14
    "pulse_schedule-i_plasma",         # 15
    "pulse_schedule-n_e_line",         # 16
    "summary-power_nbi",               # 17
]

ALL_SIGNAL_ORDER = INPUT_SIGNAL_ORDER + ACTUATOR_SIGNAL_ORDER

CATEGORY_CHANNEL_INDICES = {
    "magnetics_flux":        [7],
    "magnetics_pickup":      [1, 2, 3],
    "magnetics_saddle":      [6],
    "mirnov":                [4, 5],
    "kinetics":              [0, 12],
    "radiatives":            [10, 11],
    "active_coils":          [8, 9],
    "plasma_current":        [13],
    "kinetics_correlated":   [0, 12],
    "magnetics_active_correlated": [1, 2, 3, 6],
    "radiatives_correlated": [10, 11, 12],
    "mirnov_correlated":     [4, 5],
}


def corrupt_ts_random_dropout(X_ts, drop_rate=0.1, rng=None):
    """X_ts: (N, T, F). Returns corrupted copy + mask."""
    import numpy as np
    if rng is None: rng = np.random.default_rng()
    c = X_ts.copy()
    mask = (rng.random(c.shape) < drop_rate) & (X_ts != 0)
    c[mask] = 0.0
    return c, mask


def corrupt_ts_channel_ablation(X_ts, channel_indices, rng=None):
    """Zero out specific channel indices. X_ts: (N, T, F)."""
    import numpy as np
    c = X_ts.copy()
    mask = np.zeros(c.shape, dtype=bool)
    for idx in channel_indices:
        if idx < X_ts.shape[2]:
            orig_nonzero = X_ts[:, :, idx] != 0
            c[:, :, idx] = 0.0
            mask[:, :, idx] = orig_nonzero
    return c, mask


def corrupt_ts_temporal_gap(X_ts, gap_fraction=0.2, gap_position='front', rng=None):
    """Zero out temporal gap. X_ts: (N, T, F)."""
    import numpy as np
    if rng is None: rng = np.random.default_rng()
    c = X_ts.copy()
    T = X_ts.shape[1]
    gap_size = int(T * gap_fraction)
    if gap_position == 'front':
        start = 0
    elif gap_position == 'pre_event':
        start = T - gap_size
    else:
        start = int(rng.integers(0, max(1, T - gap_size)))
    end = start + gap_size
    mask = np.zeros(c.shape, dtype=bool)
    orig_nonzero = X_ts[:, start:end, :] != 0
    c[:, start:end, :] = 0.0
    mask[:, start:end, :] = orig_nonzero
    return c, mask


def apply_mitigation_ts(X_ts_corrupted, mask, strategy='mean_fill', channel_means=None):
    """Apply mitigation to corrupted (N, T, F) array using mask."""
    import numpy as np
    if strategy == 'zero_fill':
        return X_ts_corrupted
    result = X_ts_corrupted.copy()
    if strategy == 'mean_fill' and channel_means is not None:
        for f_idx, mean_val in enumerate(channel_means):
            ch_mask = mask[:, :, f_idx]
            result[:, :, f_idx][ch_mask] = mean_val
    elif strategy == 'forward_fill':
        for t in range(1, result.shape[1]):
            m = mask[:, t, :]
            if m.any():
                result[:, t, :][m] = result[:, t-1, :][m]
    return result


def apply_mitigation_feat(X_feat_corrupted, mask_feat, strategy='mean_fill',
                           feature_means=None):
    """Apply mitigation to corrupted (N, 142) feature array."""
    import numpy as np
    if strategy == 'zero_fill':
        return X_feat_corrupted
    result = X_feat_corrupted.copy()
    if strategy == 'mean_fill' and feature_means is not None:
        result[mask_feat] = feature_means[mask_feat]
    elif strategy == 'forward_fill':
        # For features, forward fill doesn't make sense — use mean fill
        if feature_means is not None:
            result[mask_feat] = feature_means[mask_feat]
    return result