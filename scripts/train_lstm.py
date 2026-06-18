"""
LSTM training and robustness evaluation for the TokaMark benchmark.

Loads pre-collected time series arrays from disk, trains a 2-layer LSTM
on clean data, then evaluates NRMSE degradation across six corruption
scenarios with three mitigation strategies.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace/tokamark/src")

from config import (
    RANDOM_SEED, RESULTS_DIR, CHECKPOINTS_DIR,
    DROP_RATES, GAP_FRACTIONS, N_CHANNELS_TO_KILL
)
from data_loader import load_saved_data
from corruption import (
    corrupt_ts_random_dropout,
    corrupt_ts_channel_ablation,
    corrupt_ts_temporal_gap,
    apply_mitigation_ts,
    CATEGORY_CHANNEL_INDICES,
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

DATA_DIR = "/workspace/fusion_research/data"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class TokaTensorDataset(Dataset):
    """Wraps (N, T, F) time series and (N,) targets as a PyTorch Dataset."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────

class PlasmaLSTM(nn.Module):
    """
    2-layer LSTM for plasma current prediction.
    Input:  (B, T, F) — batch of time series windows
    Output: (B,)      — predicted plasma current
    Uses the final hidden state for prediction via a 2-layer MLP head.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────

def nrmse(y_true, y_pred):
    """Normalized RMSE — matches TokaMark evaluation protocol."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()) /
                 (y_true.std() + 1e-8))


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────

def train_clean_model(X_train, y_train, X_val, y_val, n_features):
    """
    Train LSTM on clean data with early stopping on validation NRMSE.
    Saves best checkpoint to disk. Returns (model, best_val_nrmse).
    """
    print(f"\nTraining LSTM — input size: {n_features}, device: {DEVICE}")

    model     = PlasmaLSTM(input_size=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TokaTensorDataset(X_train, y_train),
                              batch_size=64, shuffle=True, num_workers=4)
    val_loader   = DataLoader(TokaTensorDataset(X_val, y_val),
                              batch_size=64, shuffle=False, num_workers=4)

    best_val_nrmse  = float("inf")
    best_state      = None
    patience_counter = 0
    MAX_PATIENCE    = 15

    for epoch in range(100):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_preds.extend(model(X_batch.to(DEVICE)).cpu().numpy())
                val_true.extend(y_batch.numpy())

        val_nrmse_val = nrmse(val_true, val_preds)
        scheduler.step(val_nrmse_val)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train_loss={np.mean(train_losses):.4f} | "
                  f"val_nrmse={val_nrmse_val:.4f}")

        if val_nrmse_val < best_val_nrmse:
            best_val_nrmse   = val_nrmse_val
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"\nBest val NRMSE: {best_val_nrmse:.4f}")

    ckpt_path = os.path.join(CHECKPOINTS_DIR, "lstm_clean.pt")
    torch.save({"model_state": best_state, "n_features": n_features}, ckpt_path)
    print(f"Model saved to {ckpt_path}")
    return model, best_val_nrmse


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading data from disk...")
    _, X_train, y_train, _ = load_saved_data("train")
    _, X_val,   y_val,   _ = load_saved_data("val")
    X_test_ts = np.load(f"{DATA_DIR}/test_X_ts.npy")  # (N, T, F)
    y_test    = np.load(f"{DATA_DIR}/test_y.npy")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test_ts.shape}")

    # Per-channel means for mean_fill mitigation
    channel_means = np.array([
        float(X_test_ts[:, :, f][X_test_ts[:, :, f] != 0].mean())
        if (X_test_ts[:, :, f] != 0).any() else 0.0
        for f in range(X_test_ts.shape[2])
    ])

    model, clean_nrmse = train_clean_model(
        X_train, y_train, X_val, y_val, n_features=X_train.shape[2])

    rng     = np.random.default_rng(RANDOM_SEED)
    results = {}
    SCORE_BATCH_SIZE = 512

    def score(X_ts_c, mask, mitigation="zero_fill"):
        """Run batched inference on corrupted+mitigated array, return NRMSE."""
        X = apply_mitigation_ts(X_ts_c, mask,
                                strategy=mitigation,
                                channel_means=channel_means) \
            if mitigation != "zero_fill" else X_ts_c
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), SCORE_BATCH_SIZE):
                preds.extend(
                    model(torch.tensor(X[i:i+SCORE_BATCH_SIZE]).to(DEVICE))
                    .cpu().numpy())
        return float(nrmse(y_test, np.array(preds)))

    def run_scenario(key_base, X_ts_c, mask):
        """Score all three mitigations and print results."""
        for mit in ["zero_fill", "mean_fill", "forward_fill"]:
            results[f"{key_base}__{mit}"] = score(X_ts_c, mask, mit)
        z = results[f"{key_base}__zero_fill"]
        m = results[f"{key_base}__mean_fill"]
        f = results[f"{key_base}__forward_fill"]
        print(f"  {key_base}: zero={z:.4f} mean={m:.4f} fwd={f:.4f}")

    # ── Clean baseline ─────────────────────────────────────────────────
    results["clean"] = score(
        X_test_ts.copy(), np.zeros(X_test_ts.shape, dtype=bool))
    print(f"\nClean NRMSE: {results['clean']:.4f}")

    # ── Scenario 1: Random dropout ─────────────────────────────────────
    print("\nScenario 1: Random dropout")
    for rate in DROP_RATES:
        run_scenario(f"dropout_{int(rate*100)}pct",
                     *corrupt_ts_random_dropout(X_test_ts, drop_rate=rate, rng=rng))

    # ── Scenario 2: Channel ablation ───────────────────────────────────
    print("\nScenario 2: Channel ablation")
    for n in N_CHANNELS_TO_KILL:
        indices = rng.choice(X_test_ts.shape[2], size=n, replace=False).tolist()
        run_scenario(f"ablation_{n}ch",
                     *corrupt_ts_channel_ablation(X_test_ts, channel_indices=indices))

    # ── Scenario 3: Per-category channel importance ────────────────────
    print("\nScenario 3: Per-category channel importance")
    for cat_name, indices in CATEGORY_CHANNEL_INDICES.items():
        if "correlated" in cat_name:
            continue
        c, mask = corrupt_ts_channel_ablation(X_test_ts, channel_indices=indices)
        results[f"category_{cat_name}__zero_fill"] = score(c, mask, "zero_fill")
        print(f"  {cat_name}: {results[f'category_{cat_name}__zero_fill']:.4f}")

    # ── Scenario 4: Temporal gap ───────────────────────────────────────
    print("\nScenario 4: Temporal gap")
    for frac in GAP_FRACTIONS:
        for pos in ["front", "random", "pre_event"]:
            run_scenario(
                f"gap_{int(frac*100)}pct_{pos}",
                *corrupt_ts_temporal_gap(X_test_ts, gap_fraction=frac,
                                         gap_position=pos, rng=rng))

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
                     *corrupt_ts_channel_ablation(X_test_ts, channel_indices=indices))

    # ── Scenario 6: Disruption-proximate failure ───────────────────────
    print("\nScenario 6: Disruption-proximate failure")
    for rate in [0.10, 0.25, 0.50]:
        run_scenario(
            f"proximate_{int(rate*100)}pct",
            *corrupt_ts_temporal_gap(X_test_ts, gap_fraction=rate,
                                     gap_position="pre_event", rng=rng))

    # ── Robustness Score ───────────────────────────────────────────────
    clean = results["clean"]
    scenario_scores = [
        clean / v for k, v in results.items()
        if k.endswith("__zero_fill") and k != "clean"
        and v is not None and not np.isnan(v) and v > 0
    ]
    rs = float(np.mean(scenario_scores)) if scenario_scores else np.nan
    results["robustness_score"] = rs
    print(f"\nLSTM Robustness Score: {rs:.4f}")

    # ── Save results ───────────────────────────────────────────────────
    results_serializable = {
        k: None if (v is None or (isinstance(v, float) and np.isnan(v)))
        else float(v)
        for k, v in results.items()
    }
    results_path = os.path.join(RESULTS_DIR, "lstm_results.json")
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("LSTM RESULTS SUMMARY")
    print("="*65)
    print(f"{'Scenario':<40} {'zero_fill':>10} {'mean_fill':>10} {'fwd_fill':>10}")
    print("-"*65)
    print(f"{'clean':<40} {clean:>10.4f}")
    for k in results:
        if k.endswith("__zero_fill"):
            base = k.replace("__zero_fill", "")
            z  = results.get(f"{base}__zero_fill")  or np.nan
            m  = results.get(f"{base}__mean_fill")   or np.nan
            fw = results.get(f"{base}__forward_fill") or np.nan
            if not np.isnan(z):
                print(f"  {base:<38} {z:>10.4f} {m:>10.4f} "
                      f"{fw:>10.4f}  ({(z-clean)/clean*100:+.1f}%)")
    print(f"\nRobustness Score: {rs:.4f}")