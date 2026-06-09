"""
Generate all publication figures from saved experiment results.

Loads xgboost_results.json, lstm_results.json, and transformer_results.json
from RESULTS_DIR and produces six figures saved as both PDF and PNG:

    fig1_degradation_curves     — NRMSE vs corruption severity (3 scenarios)
    fig2_channel_importance     — heatmap of per-category ablation degradation
    fig3_correlated_failure     — bar chart of correlated group failure
    fig4_proximate_comparison   — front gap vs pre-event vs proximate failure
    fig5_mitigation_effectiveness — mitigation strategy comparison by model
    fig6_robustness_scores      — overall Robustness Score bar chart
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "/workspace/fusion_research/results"
PLOTS_DIR = "/workspace/fusion_research/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

MODELS = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "transformer": "Transformer"
}

COLORS = {
    "xgboost": "#2196F3",
    "lstm": "#FF5722",
    "transformer": "#4CAF50"
}

MITIGATION_STYLES = {
    "zero_fill": "-",
    "mean_fill": "--",
    "forward_fill": ":"
}

MITIGATION_LABELS = {
    "zero_fill": "No mitigation",
    "mean_fill": "Mean fill",
    "forward_fill": "Forward fill"
}


# ─────────────────────────────────────────
# Load results
# ─────────────────────────────────────────

def load_results():
    results = {}
    for model_key in MODELS:
        path = os.path.join(RESULTS_DIR, f"{model_key}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[model_key] = json.load(f)
            print(f"Loaded {model_key}: {len(results[model_key])} entries")
        else:
            print(f"WARNING: {path} not found — skipping {model_key}")
    return results


# ─────────────────────────────────────────
# Figure 1: Main degradation curves
# ─────────────────────────────────────────

def plot_degradation_curves(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Model Degradation Under Sensor Failure Scenarios",
        fontsize=14, fontweight="bold", y=1.02
    )

    # Subplot 1: Random dropout
    ax = axes[0]
    rates = [10, 25, 50]
    for model_key, label in MODELS.items():
        if model_key not in results:
            continue
        r = results[model_key]
        clean = r["clean"]
        for mit, ls in MITIGATION_STYLES.items():
            vals = [r.get(f"dropout_{rate}pct__{mit}") for rate in rates]
            vals = [v if v is not None else np.nan for v in vals]
            ax.plot(rates, vals, color=COLORS[model_key],
                    linestyle=ls, marker="o",
                    label=f"{label} ({MITIGATION_LABELS[mit]})")
        ax.axhline(clean, color=COLORS[model_key],
                   linestyle="dotted", alpha=0.4)

    ax.set_xlabel("Dropout Rate (%)")
    ax.set_ylabel("NRMSE")
    ax.set_title("Random Dropout")
    ax.set_xticks(rates)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Channel ablation
    ax = axes[1]
    n_channels = [1, 3, 6]
    for model_key, label in MODELS.items():
        if model_key not in results:
            continue
        r = results[model_key]
        clean = r["clean"]
        for mit, ls in MITIGATION_STYLES.items():
            vals = [r.get(f"ablation_{n}ch__{mit}") for n in n_channels]
            vals = [v if v is not None else np.nan for v in vals]
            ax.plot(n_channels, vals, color=COLORS[model_key],
                    linestyle=ls, marker="s",
                    label=f"{label} ({MITIGATION_LABELS[mit]})")
        ax.axhline(clean, color=COLORS[model_key],
                   linestyle="dotted", alpha=0.4)

    ax.set_xlabel("Channels Ablated")
    ax.set_ylabel("NRMSE")
    ax.set_title("Channel Ablation")
    ax.set_xticks(n_channels)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Temporal gap (front)
    ax = axes[2]
    fracs = [20, 40, 60]
    for model_key, label in MODELS.items():
        if model_key not in results:
            continue
        r = results[model_key]
        clean = r["clean"]
        for mit, ls in MITIGATION_STYLES.items():
            vals = [r.get(f"gap_{frac}pct_front__{mit}") for frac in fracs]
            vals = [v if v is not None else np.nan for v in vals]
            ax.plot(fracs, vals, color=COLORS[model_key],
                    linestyle=ls, marker="^",
                    label=f"{label} ({MITIGATION_LABELS[mit]})")
        ax.axhline(clean, color=COLORS[model_key],
                   linestyle="dotted", alpha=0.4)

    ax.set_xlabel("Gap Size (% of window)")
    ax.set_ylabel("NRMSE")
    ax.set_title("Temporal Gap (Front)")
    ax.set_xticks(fracs)
    ax.grid(True, alpha=0.3)

    # Shared legend
    handles = []
    for model_key, label in MODELS.items():
        handles.append(mpatches.Patch(
            color=COLORS[model_key], label=label))
    for mit, ls in MITIGATION_STYLES.items():
        handles.append(plt.Line2D(
            [0], [0], color="gray",
            linestyle=ls, label=MITIGATION_LABELS[mit]))
    fig.legend(handles=handles, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.08), fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig1_degradation_curves.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────
# Figure 2: Channel importance heatmap
# ─────────────────────────────────────────

def plot_channel_importance(results):
    categories = [
        "magnetics_flux",
        "magnetics_pickup",
        "magnetics_saddle",
        "mirnov",
        "kinetics",
        "radiatives",
        "active_coils",
        "plasma_current",
    ]

    cat_labels = {
        "magnetics_flux":    "Flux loops",
        "magnetics_pickup":  "Pickup coils",
        "magnetics_saddle":  "Saddle coils",
        "mirnov":            "Mirnov\n(spectrograms)",
        "kinetics":          "Kinetics\n(interf.+D-alpha)",
        "radiatives":        "Soft X-ray",
        "active_coils":      "Active coils",
        "plasma_current":    "Plasma current (ip)",
    }

    model_keys = [k for k in MODELS if k in results]
    matrix = np.full((len(model_keys), len(categories)), np.nan)

    for i, model_key in enumerate(model_keys):
        r = results[model_key]
        clean = r["clean"]
        for j, cat in enumerate(categories):
            key = f"category_{cat}__zero_fill"
            val = r.get(key)
            if val is not None and not np.isnan(val):
                matrix[i, j] = (val - clean) / clean * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto",
                   vmin=0, vmax=max(np.nanmax(matrix), 1))

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        [cat_labels[c] for c in categories],
        rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels(
        [MODELS[k] for k in model_keys], fontsize=11)

    for i in range(len(model_keys)):
        for j in range(len(categories)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmax(matrix) * 0.6 \
                    else "black"
                ax.text(j, i, f"+{val:.1f}%",
                        ha="center", va="center",
                        fontsize=9, color=text_color,
                        fontweight="bold")

    plt.colorbar(im, ax=ax,
                 label="NRMSE degradation vs clean (%)")
    ax.set_title(
        "Channel Importance: NRMSE Degradation When "
        "Diagnostic Category Removed",
        fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig2_channel_importance.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────
# Figure 3: Correlated failure bar chart
# ─────────────────────────────────────────

def plot_correlated_failure(results):
    groups = ["kinetics", "magnetics_active", "radiatives", "mirnov"]
    group_labels = {
        "kinetics":         "Kinetics\n(proven correlated)",
        "magnetics_active": "Active magnetics",
        "radiatives":       "Radiatives",
        "mirnov":           "Mirnov spectrograms"
    }

    model_keys = [k for k in MODELS if k in results]
    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, model_key in enumerate(model_keys):
        r = results[model_key]
        clean = r["clean"]
        vals = []
        for group in groups:
            key = f"correlated_{group}__zero_fill"
            val = r.get(key)
            if val is not None and not np.isnan(val):
                vals.append((val - clean) / clean * 100)
            else:
                vals.append(np.nan)

        bars = ax.bar(x + i * width, vals, width,
                      label=MODELS[model_key],
                      color=COLORS[model_key],
                      alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                if val > 8:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() / 2,
                        f"+{val:.1f}%",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
                elif val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"+{val:.1f}%",
                        ha="center", va="bottom",
                        fontsize=8, color="black", fontweight="bold")
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val - 0.5,
                        f"{val:.1f}%",
                        ha="center", va="top",
                        fontsize=8, color="black", fontweight="bold")

    ax.set_xlabel("Diagnostic Group Failure", fontsize=11)
    ax.set_ylabel("NRMSE Degradation vs Clean (%)", fontsize=11)
    ax.set_title(
        "Correlated Diagnostic Group Failure\n"
        "(physically motivated by observed NaN correlation "
        "in MAST data)",
        fontsize=12, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [group_labels[g] for g in groups],
        rotation=15, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]) - 2)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig3_correlated_failure.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────
# Figure 4: Proximate vs front vs pre-event
# ─────────────────────────────────────────

def plot_proximate_comparison(results):
    model_keys = [k for k in MODELS if k in results]

    fig, axes = plt.subplots(1, len(model_keys),
                              figsize=(14, 4), sharey=False)
    fig.suptitle(
        "Disruption-Proximate Failure vs Front Gap\n"
        "(corruption injected at end of window = "
        "worst case for prediction)",
        fontsize=12, fontweight="bold"
    )

    for ax, model_key in zip(axes, model_keys):
        r = results[model_key]
        clean = r["clean"]

        # Front and pre-event gap use GAP_FRACTIONS [20, 40, 60]
        gap_fracs = [20, 40, 60]
        front_vals = [r.get(f"gap_{f}pct_front__zero_fill")
                      for f in gap_fracs]
        pre_vals   = [r.get(f"gap_{f}pct_pre_event__zero_fill")
                      for f in gap_fracs]

        # Proximate uses [10, 25, 50]
        prox_fracs = [10, 25, 50]
        prox_vals  = [r.get(f"proximate_{f}pct__zero_fill")
                      for f in prox_fracs]

        # Replace None with nan
        front_vals = [v if v is not None else np.nan for v in front_vals]
        pre_vals   = [v if v is not None else np.nan for v in pre_vals]
        prox_vals  = [v if v is not None else np.nan for v in prox_vals]

        ax.plot(gap_fracs, front_vals, "o-",
                color="#2196F3", label="Front gap (20/40/60%)")
        ax.plot(gap_fracs, pre_vals,   "s--",
                color="#FF5722", label="Pre-event gap (20/40/60%)")
        ax.plot(prox_fracs, prox_vals, "^:",
                color="#9C27B0", label="Proximate failure (10/25/50%)")
        ax.axhline(clean, color="gray",
                   linestyle="dotted", label="Clean baseline")

        ax.set_title(MODELS[model_key], fontsize=11, fontweight="bold")
        ax.set_xlabel("Gap / Failure Size (%)")
        ax.set_xticks(sorted(set(gap_fracs + prox_fracs)))
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("NRMSE")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig4_proximate_comparison.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────
# Figure 5: Mitigation effectiveness
# ─────────────────────────────────────────

def plot_mitigation_effectiveness(results):
    scenarios = {
        "dropout_25pct":        "Random dropout 25%",
        "ablation_3ch":         "Channel ablation (3ch)",
        "gap_40pct_front":      "Temporal gap 40% (front)",
        "correlated_kinetics":  "Correlated kinetics failure",
        "proximate_25pct":      "Proximate failure 25%",
    }

    model_keys = [k for k in MODELS if k in results]
    x = np.arange(len(scenarios))
    width = 0.25

    fig, axes = plt.subplots(1, len(model_keys),
                              figsize=(16, 5), sharey=False)
    fig.suptitle(
        "Mitigation Strategy Effectiveness by Model",
        fontsize=13, fontweight="bold"
    )

    mit_colors = {
        "zero_fill":    "#1f77b4",
        "mean_fill":    "#ff7f0e",
        "forward_fill": "#2ca02c"
    }

    for ax, model_key in zip(axes, model_keys):
        r = results[model_key]
        clean = r["clean"]

        for j, mit in enumerate(
                ["zero_fill", "mean_fill", "forward_fill"]):
            vals = []
            for base_key in scenarios:
                full_key = f"{base_key}__{mit}"
                val = r.get(full_key)
                if val is not None and not np.isnan(val):
                    vals.append((val - clean) / clean * 100)
                else:
                    vals.append(np.nan)

            offset = (j - 1) * width
            valid = [(i, v) for i, v in enumerate(vals)
                     if not np.isnan(v)]
            if valid:
                xi, vi = zip(*valid)
                ax.bar(
                    np.array(xi) + offset, vi, width,
                    label=MITIGATION_LABELS[mit],
                    color=mit_colors[mit],
                    alpha=0.85, edgecolor="white")

        ax.set_title(MODELS[model_key],
                     fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            list(scenarios.values()),
            rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("NRMSE degradation vs clean (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig5_mitigation_effectiveness.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────
# Figure 6: Robustness Score summary
# ─────────────────────────────────────────

def plot_robustness_scores(results):
    model_keys = [k for k in MODELS if k in results]
    scores = [results[k].get("robustness_score") for k in model_keys]
    scores = [s if s is not None else np.nan for s in scores]
    labels = [MODELS[k] for k in model_keys]
    colors = [COLORS[k] for k in model_keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, scores, color=colors,
                  alpha=0.85, edgecolor="white", width=0.5)

    for bar, score in zip(bars, scores):
        if score is not None and not np.isnan(score):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{score:.3f}",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.axhline(1.0, color="black", linestyle="--",
               linewidth=1, label="Perfect robustness (RS=1.0)")
    ax.set_ylabel("Robustness Score (RS)", fontsize=11)
    ax.set_title(
        "Overall Robustness Score by Model\n"
        "(RS = mean(clean NRMSE / corrupted NRMSE) "
        "across all scenarios;\nhigher = more robust)",
        fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig6_robustness_scores.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────

def print_summary_table(results):
    print("\n" + "="*80)
    print("FULL RESULTS SUMMARY TABLE")
    print("="*80)
    print(f"{'Scenario':<45} ", end="")
    for k in MODELS:
        if k in results:
            print(f"{MODELS[k]:>12}", end="")
    print()
    print("-"*80)

    first = next(iter(results.values()))
    all_keys = [k for k in first if k.endswith("__zero_fill")]

    for key in all_keys:
        base = key.replace("__zero_fill", "")
        print(f"  {base:<43} ", end="")
        for model_key in MODELS:
            if model_key not in results:
                continue
            val = results[model_key].get(key)
            clean = results[model_key]["clean"]
            if val is not None and not np.isnan(val):
                deg = (val - clean) / clean * 100
                print(f"  {val:>5.3f}({deg:>+5.1f}%)", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()

    print("-"*80)
    print(f"  {'Robustness Score':<43} ", end="")
    for model_key in MODELS:
        if model_key not in results:
            continue
        rs = results[model_key].get("robustness_score")
        if rs is not None and not np.isnan(rs):
            print(f"  {rs:>12.4f}", end="")
        else:
            print(f"  {'N/A':>12}", end="")
    print()
    print("="*80)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...")
    results = load_results()

    if not results:
        print("No results found. Run training scripts first.")
        exit(1)

    print(f"\nModels available: {list(results.keys())}")
    for k, r in results.items():
        rs = r.get("robustness_score")
        rs_str = f"{rs:.4f}" if rs is not None and not np.isnan(rs) \
            else "N/A"
        print(f"  {MODELS[k]}: clean NRMSE={r['clean']:.4f}, RS={rs_str}")

    print_summary_table(results)

    print("\nGenerating figures...")
    plot_degradation_curves(results)
    plot_channel_importance(results)
    plot_correlated_failure(results)
    plot_proximate_comparison(results)
    plot_mitigation_effectiveness(results)
    plot_robustness_scores(results)

    print(f"\nAll figures saved to {PLOTS_DIR}")
    print("Formats: PDF (publication quality) + PNG (for quick viewing)")