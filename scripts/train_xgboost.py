"""
XGBoost training and robustness evaluation for the TokaMark benchmark.

Loads pre-collected feature vectors from disk, trains an XGBoost regressor
on clean data, then evaluates NRMSE degradation across six corruption scenarios
with three mitigation strategies.

Note: XGBoost operates on statistical feature vectors rather than raw time
series. Temporal gap and dropout corruption are applied as feature-level
proxies (corrupting trajectory statistics) rather than true time-series
masking. Channel ablation and correlated failure are exact.
"""

import sys
import os
import json
import pickle
import numpy as np
import xgboost as xgb

sys.path.insert(0, "/workspace/tokamark/src")

from config import (
    RANDOM_SEED, RESULTS_DIR, CHECKPOINTS_DIR,
    DROP_RATES, GAP_FRACTIONS, N_CHANNELS_TO_KILL, CORRELATED_GROUPS
)
from data_loader import load_saved_data
from corruption import corrupt_ts_channel_ablation, CATEGORY_CHANNEL_INDICES

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

DATA_DIR = "/workspace/fusion_research/data"

# Feature vector layout constants
# Input signals: 14 channels × 9 stats = 126 features
# Actuator signals: 4 channels × 4 stats = 16 features
N_INPUT_CHANNELS = 14
INPUT_STATS      = 9
ACTUATOR_STATS   = 4


def nrmse(y_true, y_pred):
    """Normalized RMSE — matches TokaMark evaluation protocol."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()) /
                 (y_true.std() + 1e-8))


def train_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost regressor on clean feature vectors.
    Uses early stopping on validation RMSE. Saves checkpoint to disk.
    Returns (model, val_nrmse).
    """
    print("\nTraining XGBoost on clean data...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=20,
        eval_metric="rmse",
        verbosity=1
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=50)

    score = nrmse(y_val, model.predict(X_val))
    print(f"Clean val NRMSE: {score:.4f}")

    ckpt_path = os.path.join(CHECKPOINTS_DIR, "xgboost_clean.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved to {ckpt_path}")
    return model, score


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    print("Loading data from disk...")
    X_train, _, y_train, _ = load_saved_data("train")
    X_val,   _, y_val,   _ = load_saved_data("val")
    X_test_feat = np.load(f"{DATA_DIR}/test_X_feat.npy")  # (N, 142)
    y_test      = np.load(f"{DATA_DIR}/test_y.npy")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test_feat.shape}")

    # Per-channel feature means for mean_fill mitigation
    feature_means = np.array([
        float(X_test_feat[:, f][X_test_feat[:, f] != 0].mean())
        if (X_test_feat[:, f] != 0).any() else 0.0
        for f in range(X_test_feat.shape[1])
    ])

    model, clean_nrmse = train_model(X_train, y_train, X_val, y_val)

    rng     = np.random.default_rng(RANDOM_SEED)
    results = {}

    # ── Feature-level corruption helpers ──────────────────────────────

    def corrupt_feat_random_dropout(drop_rate):
        """Randomly zero drop_rate fraction of non-zero feature values."""
        X_c = X_test_feat.copy()
        mask = (rng.random(X_c.shape) < drop_rate) & (X_c != 0)
        X_c[mask] = 0.0
        return X_c, mask

    def corrupt_feat_channels(channel_indices):
        """
        Zero all feature stats corresponding to given channel indices.
        Maps physical channel indices to feature vector blocks.
        """
        X_c = X_test_feat.copy()
        feat_mask = np.zeros(X_test_feat.shape, dtype=bool)
        for ch_idx in channel_indices:
            if ch_idx < N_INPUT_CHANNELS:
                start = ch_idx * INPUT_STATS
                end   = start + INPUT_STATS
            else:
                act_idx = ch_idx - N_INPUT_CHANNELS
                start   = N_INPUT_CHANNELS * INPUT_STATS + act_idx * ACTUATOR_STATS
                end     = start + ACTUATOR_STATS
            X_c[:, start:end] = 0.0
            feat_mask[:, start:end] = X_test_feat[:, start:end] != 0
        return X_c, feat_mask

    def corrupt_feat_temporal_gap(gap_fraction, gap_position):
        """
        Proxy for temporal gap corruption on feature vectors.
        Corrupts trajectory statistics most affected by each gap position:
            front      — first value (index 5)
            pre_event  — last value + slope (indices 6, 7)
            random     — first + last + slope (indices 5, 6, 7)
        Note: this is an approximation. LSTM/Transformer use exact masking.
        """
        affected = {
            "front":     [5],
            "pre_event": [6, 7],
            "random":    [5, 6, 7]
        }[gap_position]

        X_c = X_test_feat.copy()
        feat_mask = np.zeros(X_test_feat.shape, dtype=bool)
        for ch_idx in range(N_INPUT_CHANNELS):
            base = ch_idx * INPUT_STATS
            for stat_idx in affected:
                col = base + stat_idx
                noise = rng.random(X_c.shape[0]) < gap_fraction
                X_c[noise, col] = 0.0
                feat_mask[noise, col] = X_test_feat[noise, col] != 0
        return X_c, feat_mask

    def apply_feat_mitigation(X_c, feat_mask, mitigation):
        """Replace masked values with feature means (mean/forward fill)."""
        if mitigation == "zero_fill" or not feat_mask.any():
            return X_c
        result = X_c.copy()
        rows, cols = np.where(feat_mask)
        result[rows, cols] = feature_means[cols]
        return result

    def score_feat(X_c, feat_mask, mitigation="zero_fill"):
        X_input = np.nan_to_num(
            apply_feat_mitigation(X_c, feat_mask, mitigation), nan=0.0)
        return float(nrmse(y_test, model.predict(X_input)))

    def run_scenario(key_base, X_c, feat_mask):
        for mit in ["zero_fill", "mean_fill", "forward_fill"]:
            results[f"{key_base}__{mit}"] = score_feat(X_c, feat_mask, mit)
        z = results[f"{key_base}__zero_fill"]
        m = results[f"{key_base}__mean_fill"]
        f = results[f"{key_base}__forward_fill"]
        print(f"  {key_base}: zero={z:.4f} mean={m:.4f} fwd={f:.4f}")

    # ── Clean baseline ─────────────────────────────────────────────────
    results["clean"] = score_feat(
        X_test_feat.copy(), np.zeros(X_test_feat.shape, dtype=bool))
    print(f"\nClean NRMSE: {results['clean']:.4f}")

    # ── Scenario 1: Random dropout ─────────────────────────────────────
    print("\nScenario 1: Random dropout")
    for rate in DROP_RATES:
        run_scenario(f"dropout_{int(rate*100)}pct",
                     *corrupt_feat_random_dropout(rate))

    # ── Scenario 2: Channel ablation ───────────────────────────────────
    print("\nScenario 2: Channel ablation")
    for n in N_CHANNELS_TO_KILL:
        indices = rng.choice(N_INPUT_CHANNELS + 4,
                             size=n, replace=False).tolist()
        run_scenario(f"ablation_{n}ch",
                     *corrupt_feat_channels(indices))

    # ── Scenario 3: Per-category channel importance ────────────────────
    print("\nScenario 3: Per-category channel importance")
    for cat_name, indices in CATEGORY_CHANNEL_INDICES.items():
        if "correlated" in cat_name:
            continue
        X_c, feat_mask = corrupt_feat_channels(indices)
        results[f"category_{cat_name}__zero_fill"] = score_feat(
            X_c, feat_mask, "zero_fill")
        print(f"  {cat_name}: {results[f'category_{cat_name}__zero_fill']:.4f}")

    # ── Scenario 4: Temporal gap ───────────────────────────────────────
    print("\nScenario 4: Temporal gap")
    for frac in GAP_FRACTIONS:
        for pos in ["front", "random", "pre_event"]:
            run_scenario(f"gap_{int(frac*100)}pct_{pos}",
                         *corrupt_feat_temporal_gap(frac, pos))

    # ── Scenario 5: Correlated failure ─────────────────────────────────
    print("\nScenario 5: Correlated failure")
    correlated_map = {
        "kinetics":         CATEGORY_CHANNEL_INDICES["kinetics_correlated"],
        "magnetics_active": CATEGORY_CHANNEL_INDICES["magnetics_active_correlated"],
        "radiatives":       CATEGORY_CHANNEL_INDICES["radiatives_correlated"],
        "mirnov":           CATEGORY_CHANNEL_INDICES["mirnov_correlated"],
    }
    for group, indices in correlated_map.items():
        run_scenario(f"correlated_{group}",
                     *corrupt_feat_channels(indices))

    # ── Scenario 6: Disruption-proximate failure ───────────────────────
    print("\nScenario 6: Disruption-proximate failure")
    for rate in [0.10, 0.25, 0.50]:
        run_scenario(f"proximate_{int(rate*100)}pct",
                     *corrupt_feat_temporal_gap(rate, "pre_event"))

    # ── Robustness Score ───────────────────────────────────────────────
    clean = results["clean"]
    scenario_scores = [
        clean / v for k, v in results.items()
        if k.endswith("__zero_fill") and k != "clean"
        and v is not None and not np.isnan(v) and v > 0
    ]
    rs = float(np.mean(scenario_scores)) if scenario_scores else np.nan
    results["robustness_score"] = rs
    print(f"\nXGBoost Robustness Score: {rs:.4f}")

    # ── Save results ───────────────────────────────────────────────────
    results_serializable = {
        k: None if (v is None or (isinstance(v, float) and np.isnan(v)))
        else float(v)
        for k, v in results.items()
    }
    results_path = os.path.join(RESULTS_DIR, "xgboost_results.json")
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("XGBOOST RESULTS SUMMARY")
    print("="*65)
    print(f"{'Scenario':<40} {'zero_fill':>10} {'mean_fill':>10} {'fwd_fill':>10}")
    print("-"*65)
    print(f"{'clean':<40} {clean:>10.4f}")
    for k in results:
        if k.endswith("__zero_fill"):
            base = k.replace("__zero_fill", "")
            z  = results.get(f"{base}__zero_fill") or np.nan
            m  = results.get(f"{base}__mean_fill")  or np.nan
            fw = results.get(f"{base}__forward_fill") or np.nan
            if not np.isnan(z):
                print(f"  {base:<38} {z:>10.4f} {m:>10.4f} "
                      f"{fw:>10.4f}  ({(z-clean)/clean*100:+.1f}%)")
    print(f"\nRobustness Score: {rs:.4f}")