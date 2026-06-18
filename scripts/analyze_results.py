"""
Generate all publication figures for the TokaMark robustness benchmark.
Produces 9 figures covering 4 architectures across all corruption scenarios,
plus shot-level alarm metrics using ground-truth disruption timestamps.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "/workspace/fusion_research/results"
PLOTS_DIR   = "/workspace/fusion_research/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

MODELS = {
    "xgboost":     "XGBoost",
    "lstm":        "LSTM",
    "transformer": "Transformer",
    "cnn":         "CNN (TokaMark baseline)",
}

COLORS = {
    "xgboost":     "#2196F3",
    "lstm":        "#FF5722",
    "transformer": "#4CAF50",
    "cnn":         "#9C27B0",
}

MITIGATION_STYLES = {
    "zero_fill":    "-",
    "mean_fill":    "--",
    "forward_fill": ":",
}

MITIGATION_LABELS = {
    "zero_fill":    "No mitigation",
    "mean_fill":    "Mean fill",
    "forward_fill": "Forward fill",
}

MARKERS = {
    "xgboost":     "o",
    "lstm":        "s",
    "transformer": "^",
    "cnn":         "D",
}


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


def save_fig(fig, name):
    for ext in ["pdf", "png"]:
        path = os.path.join(PLOTS_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved: {name}")
    plt.close(fig)


# ── Figure 1: Degradation curves ──────────────────────────────────────────────
def plot_degradation_curves(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Degradation Under Sensor Failure Scenarios",
                 fontsize=14, fontweight="bold", y=1.02)

    configs = [
        ("Random Dropout",       "Dropout Rate (%)",       "o",
         [(f"dropout_{r}pct", r)   for r in [10, 25, 50]]),
        ("Channel Ablation",     "Channels Ablated",        "s",
         [(f"ablation_{n}ch", n)   for n in [1, 3, 6]]),
        ("Temporal Gap (Front)", "Gap Size (% of window)",  "^",
         [(f"gap_{f}pct_front", f) for f in [20, 40, 60]]),
    ]

    for ax, (title, xlabel, marker, scenario_pairs) in zip(axes, configs):
        x_vals = [p[1] for p in scenario_pairs]
        for mk in MODELS:
            if mk not in results:
                continue
            r     = results[mk]
            clean = r["clean"]
            for mit, ls in MITIGATION_STYLES.items():
                vals = [r.get(f"{p[0]}__{mit}") for p in scenario_pairs]
                vals = [v if v is not None else np.nan for v in vals]
                ax.plot(x_vals, vals,
                        color=COLORS[mk], linestyle=ls,
                        marker=MARKERS[mk], linewidth=1.5)
            ax.axhline(clean, color=COLORS[mk],
                       linestyle="dotted", alpha=0.35, linewidth=1)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("NRMSE", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x_vals)
        ax.grid(True, alpha=0.3)

    handles = []
    for mk, label in MODELS.items():
        handles.append(mpatches.Patch(color=COLORS[mk], label=label))
    for mit, ls in MITIGATION_STYLES.items():
        handles.append(plt.Line2D([0], [0], color="gray",
                                  linestyle=ls, linewidth=1.5,
                                  label=MITIGATION_LABELS[mit]))
    fig.legend(handles=handles, loc="lower center",
               ncol=7, bbox_to_anchor=(0.5, -0.1), fontsize=8)

    plt.tight_layout()
    save_fig(fig, "fig1_degradation_curves")


# ── Figure 2: Channel importance heatmap ──────────────────────────────────────
def plot_channel_importance(results):
    categories = [
        "magnetics_flux", "magnetics_pickup", "magnetics_saddle",
        "mirnov", "kinetics", "radiatives", "active_coils", "plasma_current",
    ]
    cat_labels = {
        "magnetics_flux":   "Flux loops",
        "magnetics_pickup": "Pickup coils",
        "magnetics_saddle": "Saddle coils",
        "mirnov":           "Mirnov\n(spectrograms)",
        "kinetics":         "Kinetics\n(interf.+D-alpha)",
        "radiatives":       "Soft X-ray",
        "active_coils":     "Active coils",
        "plasma_current":   "Plasma current (ip)",
    }

    model_keys = [k for k in MODELS if k in results]
    matrix = np.full((len(model_keys), len(categories)), np.nan)

    for i, mk in enumerate(model_keys):
        r     = results[mk]
        clean = r["clean"]
        for j, cat in enumerate(categories):
            val = r.get(f"category_{cat}__zero_fill")
            if val is not None and not np.isnan(val):
                matrix[i, j] = (val - clean) / clean * 100

    vmax = max(np.nanmax(matrix), 1)
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([cat_labels[c] for c in categories],
                       rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels([MODELS[k] for k in model_keys], fontsize=10)

    for i in range(len(model_keys)):
        for j in range(len(categories)):
            val = matrix[i, j]
            if not np.isnan(val):
                tc   = "white" if val > vmax * 0.6 else "black"
                sign = "+" if val >= 0 else ""
                ax.text(j, i, f"{sign}{val:.1f}%",
                        ha="center", va="center",
                        fontsize=8, color=tc, fontweight="bold")

    plt.colorbar(im, ax=ax, label="NRMSE degradation vs clean (%)")
    ax.set_title(
        "Channel Importance: NRMSE Degradation When Diagnostic Category Removed",
        fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig(fig, "fig2_channel_importance")


# ── Figure 3: Correlated failure ───────────────────────────────────────────────
def plot_correlated_failure(results):
    groups = [
        ("correlated_kinetics",         "Kinetics\n(proven correlated)"),
        ("correlated_magnetics_active", "Active magnetics"),
        ("correlated_radiatives",       "Radiatives"),
        ("correlated_mirnov",           "Mirnov spectrograms"),
    ]

    model_keys = [k for k in MODELS if k in results]
    x     = np.arange(len(groups))
    width = 0.8 / len(model_keys)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, mk in enumerate(model_keys):
        r     = results[mk]
        clean = r["clean"]
        vals  = []
        for key, _ in groups:
            v = r.get(f"{key}__zero_fill")
            vals.append((v - clean) / clean * 100
                        if v is not None and not np.isnan(v) else np.nan)

        offset = (i - len(model_keys) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=COLORS[mk], alpha=0.85,
                      label=MODELS[mk])
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                sign = "+" if val >= 0 else ""
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.3 if val >= 0 else -1.2),
                        f"{sign}{val:.1f}%",
                        ha="center", va="bottom", fontsize=7,
                        fontweight="bold", color=COLORS[mk])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in groups], fontsize=10)
    ax.set_xlabel("Diagnostic Group Failure", fontsize=11)
    ax.set_ylabel("NRMSE Degradation vs Clean (%)", fontsize=11)
    ax.set_title(
        "Correlated Diagnostic Group Failure\n"
        "(physically motivated by observed NaN correlation in MAST data)",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_fig(fig, "fig3_correlated_failure")


# ── Figure 4: Proximate comparison ────────────────────────────────────────────
def plot_proximate_comparison(results):
    model_keys = [k for k in MODELS if k in results]
    fig, axes  = plt.subplots(1, len(model_keys),
                               figsize=(5 * len(model_keys), 4),
                               sharey=False)
    if len(model_keys) == 1:
        axes = [axes]

    fig.suptitle(
        "Disruption-Proximate Failure vs Front Gap\n"
        "(corruption injected at end of window = worst case for prediction)",
        fontsize=12, fontweight="bold")

    front_keys   = [("gap_20pct_front",    20),
                    ("gap_40pct_front",    40),
                    ("gap_60pct_front",    60)]
    pre_evt_keys = [("gap_20pct_pre_event", 20),
                    ("gap_40pct_pre_event", 40),
                    ("gap_60pct_pre_event", 60)]
    prox_keys    = [("proximate_10pct",    10),
                    ("proximate_25pct",    25),
                    ("proximate_50pct",    50)]

    for ax, mk in zip(axes, model_keys):
        r     = results[mk]
        clean = r["clean"]

        def get_vals(pairs):
            xs = [p[1] for p in pairs]
            ys = [r.get(f"{p[0]}__zero_fill") for p in pairs]
            ys = [v if v is not None else np.nan for v in ys]
            return xs, ys

        fx, fy = get_vals(front_keys)
        px, py = get_vals(pre_evt_keys)
        rx, ry = get_vals(prox_keys)

        ax.plot(fx, fy, "o-",  color="#2196F3",
                label="Front gap (20/40/60%)", linewidth=1.8, markersize=6)
        ax.plot(px, py, "s--", color="#F44336",
                label="Pre-event gap (20/40/60%)", linewidth=1.8, markersize=6)
        ax.plot(rx, ry, "^:",  color="#9C27B0",
                label="Proximate failure (10/25/50%)", linewidth=1.8, markersize=6)
        ax.axhline(clean, color="black", linestyle="dotted",
                   linewidth=1, label="Clean baseline")

        ax.set_title(MODELS[mk], fontsize=11, fontweight="bold",
                     color=COLORS[mk])
        ax.set_xlabel("Gap / Failure Size (%)", fontsize=9)
        ax.set_ylabel("NRMSE", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([10, 20, 25, 40, 50, 60])

    plt.tight_layout()
    save_fig(fig, "fig4_proximate_comparison")


# ── Figure 5: Mitigation effectiveness (NRMSE) ────────────────────────────────
def plot_mitigation_effectiveness(results):
    scenarios = [
        ("dropout_25pct",       "Random dropout 25%"),
        ("ablation_3ch",        "Channel ablation (3ch)"),
        ("gap_40pct_front",     "Temporal gap 40% (front)"),
        ("correlated_kinetics", "Correlated kinetics failure"),
        ("proximate_25pct",     "Proximate failure 25%"),
    ]

    model_keys = [k for k in MODELS if k in results]
    fig, axes  = plt.subplots(1, len(model_keys),
                               figsize=(5 * len(model_keys), 5))
    if len(model_keys) == 1:
        axes = [axes]

    fig.suptitle("Mitigation Strategy Effectiveness by Model",
                 fontsize=13, fontweight="bold")

    mit_colors = {
        "zero_fill":    "#455A64",
        "mean_fill":    "#FF9800",
        "forward_fill": "#4CAF50",
    }

    x     = np.arange(len(scenarios))
    width = 0.25

    for ax, mk in zip(axes, model_keys):
        r     = results[mk]
        clean = r["clean"]

        for i, (mit, label) in enumerate([
                ("zero_fill",    "No mitigation"),
                ("mean_fill",    "Mean fill"),
                ("forward_fill", "Forward fill")]):
            vals = []
            for key, _ in scenarios:
                v = r.get(f"{key}__{mit}")
                vals.append((v - clean) / clean * 100
                            if v is not None and not np.isnan(v) else 0.0)
            offset = (i - 1) * width
            ax.bar(x + offset, vals, width * 0.9,
                   color=mit_colors[mit], alpha=0.85, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in scenarios],
                           rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("NRMSE degradation vs clean (%)", fontsize=9)
        ax.set_title(MODELS[mk], fontsize=11, fontweight="bold",
                     color=COLORS[mk])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_fig(fig, "fig5_mitigation_effectiveness")


# ── Figure 6: Robustness scores ───────────────────────────────────────────────
def plot_robustness_scores(results):
    model_keys = [k for k in MODELS if k in results]
    scores     = [results[k].get("robustness_score") for k in model_keys]
    scores     = [s if s is not None else np.nan for s in scores]
    labels     = [MODELS[k] for k in model_keys]
    colors     = [COLORS[k] for k in model_keys]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, scores, color=colors,
                  alpha=0.85, edgecolor="white", width=0.5)

    for bar, score in zip(bars, scores):
        if score is not None and not np.isnan(score):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{score:.3f}",
                    ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.axhline(1.0, color="black", linestyle="--",
               linewidth=1, label="Perfect robustness (RS=1.0)")
    ax.set_ylabel("Robustness Score (RS)", fontsize=11)
    ax.set_title(
        "Overall Robustness Score by Model\n"
        "(RS = mean(clean NRMSE / corrupted NRMSE) across all scenarios;\n"
        "higher = more robust)",
        fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_fig(fig, "fig6_robustness_scores")


# ── Figure 7: Shot-level alarm metrics (clean) ────────────────────────────────
def plot_alarm_metrics():
    path = os.path.join(RESULTS_DIR, "shot_level_metrics.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping Fig 7")
        return
    with open(path) as f:
        metrics = json.load(f)

    model_keys = [k for k in MODELS if k in metrics]
    tprs   = [metrics[k]["tpr"] for k in model_keys]
    mwts   = [metrics[k]["mean_warning_time_ms"] or 0 for k in model_keys]
    labels = [MODELS[k] for k in model_keys]
    colors = [COLORS[k] for k in model_keys]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        "Shot-Level Alarm Metrics (50 disruptive test shots, real t_cut timestamps)\n"
        "FAR not computable — all test shots disruptive",
        fontsize=11, fontweight="bold")

    x = np.arange(len(model_keys))

    axes[0].bar(x, tprs, color=colors, alpha=0.85, width=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("TPR — fraction of shots alarmed before disruption\n(higher = better)")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(axes[0].patches, tprs):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     val + 0.02, f"{val:.2f}",
                     ha="center", fontsize=11, fontweight="bold")

    axes[1].bar(x, mwts, color=colors, alpha=0.85, width=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Mean Warning Time (ms)")
    axes[1].set_title("Mean Warning Time before disruption\n(higher = more time for mitigation)")
    axes[1].axhline(50, color="red", linestyle="--", linewidth=1.2,
                    label="ITER minimum (50ms)")
    axes[1].axhline(100, color="darkred", linestyle=":", linewidth=1.2,
                    label="ITER target (100ms)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(axes[1].patches, mwts):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         val + 0.3, f"{val:.1f}ms",
                         ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "fig7_alarm_metrics")


# ── Figure 8: Alarm metrics under corruption ──────────────────────────────────
def plot_alarm_under_corruption():
    path = os.path.join(RESULTS_DIR, "alarm_under_corruption.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping Fig 8")
        return
    with open(path) as f:
        data = json.load(f)

    model_keys  = [k for k in ["lstm", "transformer", "cnn"] if k in data]
    scenarios   = ["clean", "dropout_50pct", "proximate_25pct"]
    scen_labels = {
        "clean":           "Clean",
        "dropout_50pct":   "Random dropout 50%",
        "proximate_25pct": "Proximate failure 25%",
    }
    scen_colors = {
        "clean":           "#455A64",
        "dropout_50pct":   "#FF9800",
        "proximate_25pct": "#F44336",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Shot-Level Alarm Metrics Under Sensor Failure\n"
        "(real t_cut disruption timestamps, 50 disruptive test shots)",
        fontsize=12, fontweight="bold")

    x     = np.arange(len(model_keys))
    width = 0.25

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("tpr", "True Positive Rate", "TPR (higher = better)"),
        ("mean_warning_time_ms", "Mean Warning Time (ms)",
         "Mean Warning Time in ms (higher = better;\nITER requires 50-100ms — all models fall short)"),
    ]):
        ax = axes[ax_idx]
        for i, scen in enumerate(scenarios):
            vals = []
            for mk in model_keys:
                v = data[mk].get(scen, {}).get(metric)
                vals.append(float(v) if v is not None else 0.0)
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width * 0.9,
                          color=scen_colors[scen], alpha=0.85,
                          label=scen_labels[scen])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{val:.2f}" if metric == "tpr" else f"{val:.0f}ms",
                            ha="center", va="bottom", fontsize=7,
                            fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([MODELS[k] for k in model_keys],
                           rotation=10, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linewidth=0.8)
        if metric == "tpr":
            ax.set_ylim(0, 1.1)
        if metric == "mean_warning_time_ms":
            ax.set_ylim(0, 20)
            ax.axhline(50, color="red", linestyle="--", linewidth=1,
                       alpha=0.6, label="ITER min (50ms — above range)")

    plt.tight_layout()
    save_fig(fig, "fig8_alarm_under_corruption")


# ── Figure 9: Mitigation for alarm detection ──────────────────────────────────
def plot_alarm_mitigation():
    path = os.path.join(RESULTS_DIR, "alarm_mitigation_proximate.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping Fig 9")
        return
    with open(path) as f:
        data = json.load(f)

    model_keys  = [k for k in ["lstm", "transformer", "cnn"] if k in data]
    scenarios   = ["proximate_25pct_zero", "proximate_25pct_meanfill",
                   "proximate_25pct_fwdfill"]
    scen_labels = {
        "proximate_25pct_zero":     "Zero fill (no mitigation)",
        "proximate_25pct_meanfill": "Mean fill",
        "proximate_25pct_fwdfill":  "Forward fill",
    }
    scen_colors = {
        "proximate_25pct_zero":     "#455A64",
        "proximate_25pct_meanfill": "#FF9800",
        "proximate_25pct_fwdfill":  "#4CAF50",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Mitigation Strategy Effectiveness for Shot-Level Alarm Detection\n"
        "Under Disruption-Proximate Sensor Failure (25% end-of-window gap)",
        fontsize=12, fontweight="bold")

    x     = np.arange(len(model_keys))
    width = 0.25

    for i, scen in enumerate(scenarios):
        vals = []
        for mk in model_keys:
            v = data[mk].get(scen, {}).get("tpr")
            vals.append(float(v) if v is not None else 0.0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      color=scen_colors[scen], alpha=0.85,
                      label=scen_labels[scen])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODELS[k] for k in model_keys], fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "TPR under proximate failure with different imputation strategies\n"
        "(LSTM zero fill = 0.00; mean/forward fill recover to ~1.00)",
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    save_fig(fig, "fig9_alarm_mitigation")


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary_table(results):
    print("\n" + "="*90)
    print("FULL RESULTS SUMMARY")
    print("="*90)
    header = f"{'Scenario':<45}"
    for k in MODELS:
        if k in results:
            header += f"  {MODELS[k]:>22}"
    print(header)
    print("-"*90)

    first = next(iter(results.values()))
    for key in [k for k in first if k.endswith("__zero_fill")]:
        base = key.replace("__zero_fill", "")
        row  = f"  {base:<43}"
        for mk in MODELS:
            if mk not in results:
                continue
            val   = results[mk].get(key)
            clean = results[mk]["clean"]
            if val is not None and not np.isnan(val):
                deg = (val - clean) / clean * 100
                row += f"  {val:>6.3f}({deg:>+5.1f}%)"
            else:
                row += f"  {'N/A':>14}"
        print(row)

    print("-"*90)
    row = f"  {'Robustness Score':<43}"
    for mk in MODELS:
        if mk not in results:
            continue
        rs = results[mk].get("robustness_score")
        if rs is not None and not np.isnan(rs):
            row += f"  {rs:>22.4f}"
        else:
            row += f"  {'N/A':>22}"
    print(row)
    print("="*90)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading results...")
    results = load_results()

    if not results:
        print("No results found.")
        exit(1)

    print(f"\nModels available: {list(results.keys())}")
    for k, r in results.items():
        rs     = r.get("robustness_score")
        rs_str = f"{rs:.4f}" if rs and not np.isnan(rs) else "N/A"
        print(f"  {MODELS[k]}: clean={r['clean']:.4f}, RS={rs_str}")

    print_summary_table(results)

    print("\nGenerating figures...")
    plot_degradation_curves(results)
    plot_channel_importance(results)
    plot_correlated_failure(results)
    plot_proximate_comparison(results)
    plot_mitigation_effectiveness(results)
    plot_robustness_scores(results)
    plot_alarm_metrics()
    plot_alarm_under_corruption()
    plot_alarm_mitigation()

    print(f"\nAll 9 figures saved to {PLOTS_DIR}")