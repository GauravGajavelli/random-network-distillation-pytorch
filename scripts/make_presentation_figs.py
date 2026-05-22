"""Generate presentation figures for the KeyCorridor SimHash study.

Reads TensorBoard event logs from runs/exp3_keycorridor_*/ and produces four
publication-quality PNG figures in the specified output directory.

Scalar tags used (confirmed by inspecting event files):
  data/extrinsic_return        — episodic extrinsic return (all runs)
  data/simhash_unique_hashes   — cumulative unique hash count (simhash/simhash_only only)

Figures produced:
  fig1_learning_curves.png  — mean ±1 SE learning curves, SimHash+RND vs Vanilla
  fig2_ablation.png         — same but all three methods including SimHash-only
  fig3_steps_to_goal.png    — bar chart of mean steps-to-goal ±1 std
  fig4_summary_table.png    — rendered summary table (return, steps, unique hashes)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs", help="Root directory of all run folders")
    p.add_argument("--out-dir", default="scripts/figures", help="Output directory for PNG files")
    p.add_argument("--success-threshold", type=float, default=0.5,
                   help="Extrinsic return threshold for computing steps-to-goal (default 0.5)")
    p.add_argument("--smoothing-window", type=int, default=10,
                   help="Rolling-mean window for smoothing before threshold crossing (default 10)")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI (default 150)")
    p.add_argument("--inspect-tags", action="store_true",
                   help="Print all available scalar tags from one run per group and exit")
    return p.parse_args()

# ---------------------------------------------------------------------------
# TB loading (reuses project utility if available, else minimal fallback)
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from tb_utils import load_scalar as _tb_load_scalar
    def load_scalar(run_dir, tag):
        steps, vals = _tb_load_scalar(str(run_dir), tag)
        return np.array(steps, dtype=np.float64), np.array(vals, dtype=np.float64)
except ImportError:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    def load_scalar(run_dir, tag):
        ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return np.array([]), np.array([])
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.float64)
        vals  = np.array([e.value for e in events], dtype=np.float64)
        return steps, vals

def available_tags(run_dir):
    try:
        from tb_utils import available_tags as _at
        return _at(str(run_dir))
    except ImportError:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        ea.Reload()
        return ea.Tags().get("scalars", [])

# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
GROUPS = {
    "vanilla":      ["exp3_keycorridor_baseline",
                     "exp3_keycorridor_baseline_seed1",
                     "exp3_keycorridor_baseline_seed2"],
    "simhash":      ["exp3_keycorridor_simhash",
                     "exp3_keycorridor_simhash_seed1",
                     "exp3_keycorridor_simhash_seed2"],
    "simhash_only": ["exp3_keycorridor_simhash_only_seed0",
                     "exp3_keycorridor_simhash_only_seed1",
                     "exp3_keycorridor_simhash_only_seed2"],
}

LABELS = {
    "vanilla":      "Vanilla RND",
    "simhash":      "SimHash+RND",
    "simhash_only": "SimHash-only",
}

COLORS = {
    "vanilla":      "#2C3344",
    "simhash":      "#E07A5F",
    "simhash_only": "#8A8F98",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rolling_mean(arr, window):
    """Simple 1-D causal rolling mean; first (window-1) values use shorter window."""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = arr[lo:i+1].mean()
    return out

def interpolate_to_grid(steps, vals, grid):
    """Linear interpolation of (steps, vals) onto grid; extrapolates with edge values."""
    return np.interp(grid, steps, vals)

def load_group(runs_dir, group_dirs, tag):
    """Return list of (steps, vals) arrays for a group, skipping missing runs."""
    results = []
    for d in group_dirs:
        path = Path(runs_dir) / d
        if not path.exists():
            print(f"  [warn] run not found: {path}")
            continue
        s, v = load_scalar(path, tag)
        if len(s) == 0:
            print(f"  [warn] tag '{tag}' not found in {path}")
            continue
        results.append((s, v))
    return results

def aggregate(series_list, n_grid=500):
    """
    Aggregate a list of (steps, vals) onto a common grid.
    Returns (grid, mean, se) arrays.
    """
    if not series_list:
        return None, None, None
    max_step = min(s[-1] for s, _ in series_list)   # conservative: shortest run
    grid = np.linspace(0, max_step, n_grid)
    interped = np.stack([interpolate_to_grid(s, v, grid) for s, v in series_list])
    mean = interped.mean(axis=0)
    se   = interped.std(axis=0) / np.sqrt(len(series_list))
    return grid, mean, se

def steps_to_goal(steps, vals, threshold, window):
    """
    First global step at which a rolling-smoothed return crosses threshold.
    Returns float step or None if never crossed.
    """
    smoothed = rolling_mean(vals, window)
    idx = np.argmax(smoothed >= threshold)
    if smoothed[idx] < threshold:
        return None
    return float(steps[idx])

def tail_mean(vals, frac=0.05):
    n = max(1, int(len(vals) * frac))
    return float(vals[-n:].mean())

def fmt_k(x):
    return f"{x/1000:.1f}k" if x is not None else "∞"

# ---------------------------------------------------------------------------
# Figure 1 — headline learning curves (vanilla vs SimHash+RND)
# ---------------------------------------------------------------------------
def fig1_learning_curves(runs_dir, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(8, 5))
    tag = "data/extrinsic_return"

    for group in ["vanilla", "simhash"]:
        series = load_group(runs_dir, GROUPS[group], tag)
        grid, mean, se = aggregate(series)
        if grid is None:
            continue
        color = COLORS[group]
        ax.plot(grid / 1e6, mean, color=color, linewidth=2.2, label=LABELS[group])
        ax.fill_between(grid / 1e6, mean - se, mean + se,
                        color=color, alpha=0.18)

    ax.set_xlabel("Environment steps (millions)", fontsize=13)
    ax.set_ylabel("Mean episodic extrinsic return", fontsize=13)
    ax.set_title("KeyCorridor: SimHash+RND vs Vanilla RND\n(n=3 seeds, shading = ±1 SE)",
                 fontsize=16)
    ax.legend(fontsize=12, loc="lower right")
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out = Path(out_dir) / "fig1_learning_curves.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  saved {out}")

# ---------------------------------------------------------------------------
# Figure 2 — ablation: all three methods
# ---------------------------------------------------------------------------
def fig2_ablation(runs_dir, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(8, 5))
    tag = "data/extrinsic_return"

    for group in ["vanilla", "simhash", "simhash_only"]:
        series = load_group(runs_dir, GROUPS[group], tag)
        grid, mean, se = aggregate(series)
        if grid is None:
            continue
        color = COLORS[group]
        ls = "--" if group == "simhash_only" else "-"
        ax.plot(grid / 1e6, mean, color=color, linewidth=2.2,
                linestyle=ls, label=LABELS[group])
        ax.fill_between(grid / 1e6, mean - se, mean + se,
                        color=color, alpha=0.15)

    ax.set_xlabel("Environment steps (millions)", fontsize=13)
    ax.set_ylabel("Mean episodic extrinsic return", fontsize=13)
    ax.set_title("Ablation: RND Provides the Directional Signal\n(n=3 seeds, shading = ±1 SE)",
                 fontsize=16)
    ax.legend(fontsize=12, loc="center right")
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out = Path(out_dir) / "fig2_ablation.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  saved {out}")

# ---------------------------------------------------------------------------
# Figure 3 — bar chart of steps-to-goal
# ---------------------------------------------------------------------------
def fig3_steps_to_goal(runs_dir, out_dir, dpi, threshold, window):
    tag = "data/extrinsic_return"
    results = {}

    print(f"\n  Steps-to-goal per seed (threshold={threshold}, window={window}):")
    for group in ["vanilla", "simhash", "simhash_only"]:
        per_seed = []
        for i, d in enumerate(GROUPS[group]):
            path = Path(runs_dir) / d
            if not path.exists():
                print(f"    {LABELS[group]} seed {i}: run missing")
                continue
            s, v = load_scalar(path, tag)
            if len(s) == 0:
                print(f"    {LABELS[group]} seed {i}: no data")
                continue
            stg = steps_to_goal(s, v, threshold, window)
            if stg is None:
                print(f"    {LABELS[group]} seed {i}: NEVER crossed {threshold}")
            else:
                print(f"    {LABELS[group]} seed {i}: {fmt_k(stg)}")
                per_seed.append(stg)
        results[group] = per_seed

    # Only plot methods with at least one crossing
    groups_to_plot = [g for g in ["vanilla", "simhash", "simhash_only"] if results[g]]
    means = [np.mean(results[g]) for g in groups_to_plot]
    stds  = [np.std(results[g], ddof=0) for g in groups_to_plot]
    x     = np.arange(len(groups_to_plot))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, [m / 1000 for m in means],
                  yerr=[s / 1000 for s in stds],
                  color=[COLORS[g] for g in groups_to_plot],
                  capsize=6, width=0.5, edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=1.8, ecolor="#333333"))

    # Annotate each bar with mean ± std
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s / 1000 + 4,
                f"{fmt_k(m)}\n±{fmt_k(s)}",
                ha="center", va="bottom", fontsize=11, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[g] for g in groups_to_plot], fontsize=13)
    ax.set_ylabel("Steps to goal (thousands)", fontsize=13)
    ax.set_title(f"Mean Steps-to-Goal  (threshold={threshold}, n=3 seeds)\n"
                 f"Error bars = ±1 std", fontsize=16)
    ax.tick_params(labelsize=11)
    ax.set_ylim(0, max(means) / 1000 * 1.35)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out = Path(out_dir) / "fig3_steps_to_goal.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  saved {out}")

# ---------------------------------------------------------------------------
# Figure 4 — summary table
# ---------------------------------------------------------------------------
def fig4_summary_table(runs_dir, out_dir, dpi, threshold, window):
    tag_return = "data/extrinsic_return"
    tag_hashes = "data/simhash_unique_hashes"

    rows = []
    for group in ["vanilla", "simhash", "simhash_only"]:
        stg_vals, final_vals, hash_vals = [], [], []
        for i, d in enumerate(GROUPS[group]):
            path = Path(runs_dir) / d
            if not path.exists():
                continue
            s, v = load_scalar(path, tag_return)
            if len(s):
                stg = steps_to_goal(s, v, threshold, window)
                if stg is not None:
                    stg_vals.append(stg)
                final_vals.append(tail_mean(v, frac=0.05))
            sh, hv = load_scalar(path, tag_hashes)
            if len(hv):
                hash_vals.append(float(hv[-1]))

        def fmt_mean_std(vals, scale=1, fmt=".1f"):
            if not vals:
                return "∞"
            m, s = np.mean(vals) / scale, np.std(vals, ddof=0) / scale
            return f"{m:{fmt}} ±{s:{fmt}}"

        stg_str    = fmt_mean_std(stg_vals, scale=1000, fmt=".1f") + "k" if stg_vals else "∞"
        final_str  = fmt_mean_std(final_vals, scale=1, fmt=".3f")
        hashes_str = fmt_mean_std(hash_vals, scale=1, fmt=".0f") if hash_vals else "—"

        rows.append([LABELS[group], stg_str, final_str, hashes_str])

    col_labels = ["Method", f"Steps-to-Goal\n(thresh={threshold})",
                  "Final Return\n(last 5%)", "Unique Hashes\n(final)"]

    fig, ax = plt.subplots(figsize=(10, 2.4 + 0.45 * len(rows)))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3344")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row shading; highlight SimHash+RND row
    group_keys = ["vanilla", "simhash", "simhash_only"]
    for i, group in enumerate(group_keys):
        shade = "#F5F5F5" if i % 2 == 0 else "white"
        if group == "simhash":
            shade = "#FFF0EB"  # warm highlight for the winning method
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(shade)

    ax.set_title("KeyCorridor Results Summary (n=3 seeds per method)",
                 fontsize=16, pad=14)
    fig.tight_layout()
    out = Path(out_dir) / "fig4_summary_table.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ---------------------------------------------------------------------------
# Tag inspection helper
# ---------------------------------------------------------------------------
def inspect_tags(runs_dir):
    print("=== Available scalar tags (one representative run per group) ===")
    for group, dirs in GROUPS.items():
        for d in dirs:
            path = Path(runs_dir) / d
            if path.exists():
                tags = available_tags(path)
                print(f"\n{group}  ({d}):")
                for t in sorted(tags):
                    print(f"  {t}")
                break
        else:
            print(f"\n{group}: no run dirs found under {runs_dir}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.inspect_tags:
        inspect_tags(args.runs_dir)
        return

    print(f"Runs dir : {args.runs_dir}")
    print(f"Output   : {args.out_dir}")
    print(f"Threshold: {args.success_threshold}  |  Smoothing window: {args.smoothing_window}")
    print()

    fig1_learning_curves(args.runs_dir, args.out_dir, args.dpi)
    fig2_ablation(args.runs_dir, args.out_dir, args.dpi)
    fig3_steps_to_goal(args.runs_dir, args.out_dir, args.dpi,
                       args.success_threshold, args.smoothing_window)
    fig4_summary_table(args.runs_dir, args.out_dir, args.dpi,
                       args.success_threshold, args.smoothing_window)

    print(f"\nDone — 4 figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
