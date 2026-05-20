"""Experiment 3 verifier: DSC contributes beyond Option B on KeyCorridorS3R3.

Win conditions:
  - DSC reaches the goal in <=80% of the timesteps that the baseline
    (RND+Option B) needs.
  - Anchor-coverage curve exported as PNG for qualitative inspection of the
    discovery-order claim.
  - Optional: noisy_tv_robustness_gap (DSC tv_on vs tv_off) within 10%.

Exit codes: 0 = PASS, 1 = FAIL, 2 = data missing or baseline never reached goal.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tb_utils import load_scalar, tail_mean


def first_goal_step(log_dir: str, target: float = 1.0) -> float:
    """Return the env step at which extrinsic_return first reached `target`.

    Returns float('inf') if never reached. KeyCorridor gives reward in [0, 1]
    proportional to remaining timesteps; reward > 0 means a successful pickup.
    """
    steps, values = load_scalar(log_dir, "data/extrinsic_return")
    for s, v in zip(steps, values):
        if v >= target:
            return s
    return float("inf")


def evaluate(baseline_dir: str, dsc_dir: str, threshold: float = 0.8,
             goal_target: float = 0.5,
             dsc_tv_off_dir: str = None,
             figures_dir: str = "scripts/figures") -> int:
    base_step = first_goal_step(baseline_dir, target=goal_target)
    dsc_step = first_goal_step(dsc_dir, target=goal_target)
    print(f"baseline (RND+Option B) first reaches reward>={goal_target} at step: "
          f"{base_step:.0f}")
    print(f"DSC                     first reaches reward>={goal_target} at step: "
          f"{dsc_step:.0f}")

    if base_step == float("inf"):
        print("FAIL: baseline never reached goal — run longer, check env wiring, "
              "or lower --goal-target.")
        return 2

    if dsc_step == float("inf"):
        print("FAIL: DSC never reached goal — implementation broken or undertrained.")
        return 1

    ratio = dsc_step / base_step
    print(f"time-to-goal ratio (DSC / baseline): {ratio:.3f} "
          f"(target: <={threshold})")

    # Qualitative: anchor coverage over training
    steps, coverage = load_scalar(dsc_dir, "data/anchor_coverage")
    if steps and coverage:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig_dir = Path(figures_dir)
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.plot(steps, coverage)
            plt.xlabel("env steps")
            plt.ylabel("unique anchors visited per episode")
            plt.title("DSC anchor coverage")
            out = fig_dir / "anchor_coverage.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"wrote {out}")
        except ImportError:
            print("matplotlib not installed; skipping anchor coverage plot")
    else:
        print("note: data/anchor_coverage absent; skipping qualitative plot")

    tv_warn = ""
    if dsc_tv_off_dir:
        _, on_returns = load_scalar(dsc_dir, "data/extrinsic_return")
        _, off_returns = load_scalar(dsc_tv_off_dir, "data/extrinsic_return")
        if on_returns and off_returns:
            on_mean, off_mean = tail_mean(on_returns), tail_mean(off_returns)
            regression = (off_mean - on_mean) / max(off_mean, 1e-6)
            print(f"noisy_tv_robustness: tv_on={on_mean:.3f} tv_off={off_mean:.3f} "
                  f"regression={regression:.1%}")
            if regression > 0.10:
                tv_warn = f" (warning: noisy-TV regression {regression:.1%} > 10%)"
        else:
            print("note: noisy-TV comparison skipped (missing extrinsic_return)")

    if ratio <= threshold:
        print(f"PASS{tv_warn}")
        return 0
    print("FAIL")
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-log-dir", required=True,
                   help="RND + Option B run (DSC's actual baseline)")
    p.add_argument("--dsc-log-dir", required=True,
                   help="RND + Option B + DSC run")
    p.add_argument("--dsc-tv-off-log-dir", default=None,
                   help="optional: DSC with tv_on=False for noisy-TV robustness check")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="maximum DSC/baseline time-to-goal ratio to PASS")
    p.add_argument("--goal-target", type=float, default=0.5,
                   help="extrinsic_return value counted as 'reached goal'")
    p.add_argument("--figures-dir", default="scripts/figures")
    args = p.parse_args()
    sys.exit(evaluate(args.baseline_log_dir, args.dsc_log_dir, args.threshold,
                      args.goal_target, args.dsc_tv_off_log_dir, args.figures_dir))


if __name__ == "__main__":
    main()
