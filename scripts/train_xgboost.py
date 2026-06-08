import sys
import os
import numpy as np
import pickle
import json

sys.path.insert(0, "/workspace/tokamark/src")

from config import (
    RANDOM_SEED, RESULTS_DIR, CHECKPOINTS_DIR,
    DROP_RATES, GAP_FRACTIONS, N_CHANNELS_TO_KILL, CORRELATED_GROUPS
)
from data_loader import load_saved_data, load_test_samples
from feature_engineering import extract_features
from corruption import (
    corrupt_random_dropout,
    corrupt_channel_ablation,
    corrupt_temporal_gap,
    corrupt_correlated_failure,
    corrupt_with_mask,
    apply_mitigation_with_mask
)

import xgboost as xgb

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# NRMSE
# ─────────────────────────────────────────

def nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    return float(rmse / (y_true.std() + 1e-8))


# ─────────────────────────────────────────
# Train
# ─────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val):
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
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    val_pred = model.predict(X_val)
    score = nrmse(y_val, val_pred)
    print(f"Clean val NRMSE: {score:.4f}")

    ckpt_path = os.path.join(CHECKPOINTS_DIR, "xgboost_clean.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved to {ckpt_path}")
    return model, score


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    from corruption import (
        corrupt_ts_random_dropout,
        corrupt_ts_channel_ablation,
        corrupt_ts_temporal_gap,
        CATEGORY_CHANNEL_INDICES,
    )

    print("Loading data from disk...")
    X_train, _, y_train, feat_names = load_saved_data("train")
    X_val, _, y_val, _ = load_saved_data("val")

    DATA_DIR = "/workspace/fusion_research/data"
    X_test_ts   = np.load(f"{DATA_DIR}/test_X_ts.npy")    # (N, T, F)
    X_test_feat = np.load(f"{DATA_DIR}/test_X_feat.npy")  # (N, 142)
    y_test      = np.load(f"{DATA_DIR}/test_y.npy")        # (N,)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Test ts: {X_test_ts.shape}, Test feat: {X_test_feat.shape}")

    # Feature means for mean_fill
    feature_means = np.zeros(X_test_feat.shape[1])
    for f in range(X_test_feat.shape[1]):
        vals = X_test_feat[:, f]
        nonzero = vals[vals != 0]
        feature_means[f] = float(nonzero.mean()) if len(nonzero) > 0 else 0.0

    N_INPUT_CHANNELS = 14
    INPUT_STATS = 9
    ACTUATOR_STATS = 4

    # ── Corruption functions for feature arrays ──

    def corrupt_feat_random_dropout(drop_rate):
        X_c = X_test_feat.copy()
        mask = (rng.random(X_c.shape) < drop_rate) & (X_c != 0)
        X_c[mask] = 0.0
        return X_c, mask

    def corrupt_feat_channels(channel_indices):
        X_c = X_test_feat.copy()
        feat_mask = np.zeros(X_test_feat.shape, dtype=bool)
        for ch_idx in channel_indices:
            if ch_idx < N_INPUT_CHANNELS:
                feat_start = ch_idx * INPUT_STATS
                feat_end = feat_start + INPUT_STATS
            else:
                act_idx = ch_idx - N_INPUT_CHANNELS
                feat_start = (N_INPUT_CHANNELS * INPUT_STATS
                              + act_idx * ACTUATOR_STATS)
                feat_end = feat_start + ACTUATOR_STATS
            X_c[:, feat_start:feat_end] = 0.0
            feat_mask[:, feat_start:feat_end] = (
                X_test_feat[:, feat_start:feat_end] != 0)
        return X_c, feat_mask

    def corrupt_feat_temporal_gap(gap_fraction, gap_position):
        X_c = X_test_feat.copy()
        feat_mask = np.zeros(X_test_feat.shape, dtype=bool)
        # Stats order: mean=0,std=1,min=2,max=3,med=4,first=5,last=6,slope=7,zero_frac=8
        if gap_position == "front":
            affected_stats = [5]       # first value most affected
        elif gap_position == "pre_event":
            affected_stats = [6, 7]    # last value + slope most affected
        else:
            affected_stats = [5, 6, 7] # random gap affects all trajectory stats
        for ch_idx in range(N_INPUT_CHANNELS):
            feat_base = ch_idx * INPUT_STATS
            for stat_idx in affected_stats:
                col = feat_base + stat_idx
                noise = rng.random(X_c.shape[0]) < gap_fraction
                X_c[noise, col] = 0.0
                feat_mask[noise, col] = X_test_feat[noise, col] != 0
        return X_c, feat_mask

    def apply_feat_mitigation(X_c, feat_mask, mitigation):
        if mitigation == "zero_fill":
            return X_c
        result = X_c.copy()
        if feat_mask.any():
            rows, cols = np.where(feat_mask)
            result[rows, cols] = feature_means[cols]
        return result

    def score_feat(X_c, feat_mask, mitigation="zero_fill"):
        X_input = apply_feat_mitigation(X_c, feat_mask, mitigation)
        X_input = np.nan_to_num(X_input, nan=0.0)
        preds = model.predict(X_input)
        return float(nrmse(y_test, preds))

    def run_scenario(key_base, X_c, feat_mask):
        for mit in ["zero_fill", "mean_fill", "forward_fill"]:
            results[f"{key_base}__{mit}"] = score_feat(X_c, feat_mask, mit)
        z = results[f"{key_base}__zero_fill"]
        m = results[f"{key_base}__mean_fill"]
        f = results[f"{key_base}__forward_fill"]
        print(f"  {key_base}: zero={z:.4f} mean={m:.4f} fwd={f:.4f}")

    # ── Train ──
    model, clean_nrmse = train_model(X_train, y_train, X_val, y_val)

    rng = np.random.default_rng(RANDOM_SEED)
    results = {}

    # ── Clean baseline ──
    clean_mask = np.zeros(X_test_feat.shape, dtype=bool)
    results["clean"] = score_feat(X_test_feat.copy(), clean_mask)
    print(f"\nClean NRMSE: {results['clean']:.4f}")

    # ── Scenario 1: Random dropout ──
    print("\nScenario 1: Random dropout")
    for rate in DROP_RATES:
        X_c, feat_mask = corrupt_feat_random_dropout(rate)
        run_scenario(f"dropout_{int(rate*100)}pct", X_c, feat_mask)

    # ── Scenario 2: Channel ablation ──
    print("\nScenario 2: Channel ablation")
    for n in N_CHANNELS_TO_KILL:
        indices = rng.choice(
            N_INPUT_CHANNELS + 4, size=n, replace=False).tolist()
        X_c, feat_mask = corrupt_feat_channels(indices)
        run_scenario(f"ablation_{n}ch", X_c, feat_mask)

    # ── Scenario 3: Per-category channel importance ──
    print("\nScenario 3: Per-category channel importance")
    for cat_name, indices in CATEGORY_CHANNEL_INDICES.items():
        if 'correlated' in cat_name:
            continue
        X_c, feat_mask = corrupt_feat_channels(indices)
        results[f"category_{cat_name}__zero_fill"] = score_feat(
            X_c, feat_mask, "zero_fill")
        print(f"  {cat_name}: "
              f"{results[f'category_{cat_name}__zero_fill']:.4f}")

    # ── Scenario 4: Temporal gap ──
    print("\nScenario 4: Temporal gap")
    for frac in GAP_FRACTIONS:
        for pos in ["front", "random", "pre_event"]:
            X_c, feat_mask = corrupt_feat_temporal_gap(frac, pos)
            run_scenario(f"gap_{int(frac*100)}pct_{pos}", X_c, feat_mask)

    # ── Scenario 5: Correlated failure ──
    print("\nScenario 5: Correlated failure")
    correlated_map = {
        "kinetics":         CATEGORY_CHANNEL_INDICES["kinetics_correlated"],
        "magnetics_active": CATEGORY_CHANNEL_INDICES["magnetics_active_correlated"],
        "radiatives":       CATEGORY_CHANNEL_INDICES["radiatives_correlated"],
        "mirnov":           CATEGORY_CHANNEL_INDICES["mirnov_correlated"],
    }
    for group, indices in correlated_map.items():
        X_c, feat_mask = corrupt_feat_channels(indices)
        run_scenario(f"correlated_{group}", X_c, feat_mask)

    # ── Scenario 6: Disruption-proximate failure ──
    print("\nScenario 6: Disruption-proximate failure")
    for rate in [0.10, 0.25, 0.50]:
        X_c, feat_mask = corrupt_feat_temporal_gap(rate, "pre_event")
        run_scenario(f"proximate_{int(rate*100)}pct", X_c, feat_mask)

    # ── Robustness Score ──
    clean = results["clean"]
    scenario_scores = []
    for k, v in results.items():
        if k == "clean" or not k.endswith("__zero_fill"):
            continue
        if v is not None and not np.isnan(v) and v > 0:
            scenario_scores.append(clean / v)
    rs = float(np.mean(scenario_scores)) if scenario_scores else np.nan
    results["robustness_score"] = rs
    print(f"\nXGBoost Robustness Score: {rs:.4f}")

    # ── Save ──
    results_serializable = {}
    for k, v in results.items():
        if v is None:
            results_serializable[k] = None
        elif isinstance(v, float) and np.isnan(v):
            results_serializable[k] = None
        else:
            results_serializable[k] = float(v)

    results_path = os.path.join(RESULTS_DIR, "xgboost_results.json")
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── Summary ──
    print("\n" + "="*65)
    print("XGBOOST RESULTS SUMMARY")
    print("="*65)
    print(f"{'Scenario':<40} {'zero_fill':>10} "
          f"{'mean_fill':>10} {'fwd_fill':>10}")
    print("-"*65)
    print(f"{'clean':<40} {clean:>10.4f}")
    for k in results:
        if k.endswith("__zero_fill"):
            base = k.replace("__zero_fill", "")
            z = results.get(f"{base}__zero_fill") or np.nan
            m = results.get(f"{base}__mean_fill") or np.nan
            fw = results.get(f"{base}__forward_fill") or np.nan
            if not np.isnan(z):
                deg = (z - clean) / clean * 100
                print(f"  {base:<38} {z:>10.4f} {m:>10.4f} "
                      f"{fw:>10.4f}  ({deg:+.1f}%)")
    print(f"\nRobustness Score: {rs:.4f}")