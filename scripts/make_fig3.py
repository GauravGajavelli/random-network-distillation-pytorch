"""Regenerate fig3_steps_to_goal.png to match deck numbers (320.5k / 267.6k).

Uses first-raw-crossing definition (no smoothing) to match eval_simhash.py.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from tb_utils import load_scalar

RUNS_DIR = Path("runs")
OUT = Path("scripts/figures/fig3_steps_to_goal.png")
THRESHOLD = 0.5

GROUPS = {
    "vanilla": [
        "exp3_keycorridor_baseline",
        "exp3_keycorridor_baseline_seed1",
        "exp3_keycorridor_baseline_seed2",
    ],
    "simhash": [
        "exp3_keycorridor_simhash",
        "exp3_keycorridor_simhash_seed1",
        "exp3_keycorridor_simhash_seed2",
    ],
}

COLORS = {"vanilla": "#2C3344", "simhash": "#E07A5F"}
LABELS = {"vanilla": "Vanilla RND", "simhash": "SimHash+RND"}

def first_crossing(steps, vals, threshold):
    for s, v in zip(steps, vals):
        if v >= threshold:
            return float(s)
    return float("inf")

# ── collect per-seed steps ───────────────────────────────────────────────────
print(f"Per-seed steps-to-goal (first crossing of {THRESHOLD}):\n")
means, stds, per_seed_all = {}, {}, {}

for group, dirs in GROUPS.items():
    crossings = []
    for i, d in enumerate(dirs):
        path = RUNS_DIR / d
        steps, vals = load_scalar(str(path), "data/extrinsic_return")
        stg = first_crossing(steps, vals, THRESHOLD)
        label = "∞" if stg == float("inf") else f"{stg/1000:.1f}k"
        print(f"  {LABELS[group]} seed {i}: {label}")
        if stg < float("inf"):
            crossings.append(stg)
    per_seed_all[group] = crossings
    means[group] = np.mean(crossings)
    stds[group]  = np.std(crossings, ddof=0)
    print(f"  → mean {means[group]/1000:.1f}k  std {stds[group]/1000:.1f}k\n")

# ── figure ───────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 1200 / 150, 750 / 150   # 8 × 5 inches at 150 DPI
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_alpha(0)
ax.set_facecolor("none")

order  = ["vanilla", "simhash"]
x      = np.array([0, 1])
bar_w  = 0.46

bars = ax.bar(
    x,
    [means[g] / 1000 for g in order],
    width=bar_w,
    color=[COLORS[g] for g in order],
    yerr=[stds[g] / 1000 for g in order],
    capsize=7,
    error_kw=dict(elinewidth=1.8, ecolor="#111111", capthick=1.8),
    zorder=3,
)

# ── value labels above each bar ──────────────────────────────────────────────
for bar, g, xpos in zip(bars, order, x):
    m_k = means[g] / 1000
    s_k = stds[g]  / 1000
    top = bar.get_height() + stds[g] / 1000 + 5   # just above error cap
    ax.text(
        xpos, top,
        f"{m_k:.1f}k ±{s_k:.1f}k",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        color=COLORS[g],
        zorder=4,
    )

# ── −16.5% bracket annotation ────────────────────────────────────────────────
pct = (means["simhash"] - means["vanilla"]) / means["vanilla"] * 100  # negative
bracket_y  = max(means[g] / 1000 for g in order) + stds["vanilla"] / 1000 + 26
tick_len   = 5   # vertical tick drop from bracket line

# horizontal bracket line
ax.plot([x[0], x[1]], [bracket_y, bracket_y],
        color="#333333", linewidth=1.5, solid_capstyle="round", zorder=4)
# vertical ticks at each end
for xi in x:
    ax.plot([xi, xi], [bracket_y - tick_len, bracket_y],
            color="#333333", linewidth=1.5, zorder=4)

# centred label
ax.text(
    (x[0] + x[1]) / 2, bracket_y + 2,
    f"{pct:.1f}%  faster",
    ha="center", va="bottom",
    fontsize=13, color="#333333",
    fontstyle="italic",
    zorder=4,
)

# ── axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels([LABELS[g] for g in order], fontsize=14)
ax.set_ylabel("Steps to goal (thousands)", fontsize=13)
ax.set_title("Mean Steps-to-Goal — KeyCorridor (n=3 seeds)", fontsize=16, pad=10)

y_max = bracket_y + 22
ax.set_ylim(0, y_max)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}k")
)
ax.tick_params(axis="y", labelsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2, linewidth=0.7, zorder=0)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, transparent=True, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}")
