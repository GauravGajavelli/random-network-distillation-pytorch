"""Experiment 2 verifier: Option B fixes the Gravitar-style failure on DoorKey-8x8.

Win condition: RND+Option B closes >=80% of the gap between vanilla RND and
PPO-only on extrinsic_return (final 100 episodes mean).

Also exports a gating_factor decay PNG for qualitative inspection of the
mechanism evidence (gate should close as extrinsic critics converge).

Exit codes: 0 = PASS, 1 = FAIL, 2 = data missing or failure not reproduced.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tb_utils import load_scalar, tail_mean


def evaluate(ppo_dir: str, vanilla_rnd_dir: str, option_b_dir: str,
             threshold: float = 0.8, figures_dir: str = "scripts/figures") -> int:
    returns = {}
    for name, log_dir in [("ppo_only", ppo_dir),
                          ("vanilla_rnd", vanilla_rnd_dir),
                          ("option_b", option_b_dir)]:
        _, r = load_scalar(log_dir, "data/extrinsic_return")
        if not r:
            print(f"FAIL: {name} run at {log_dir} has no data/extrinsic_return scalar")
            return 2
        returns[name] = tail_mean(r)

    for name, value in returns.items():
        print(f"{name}: extrinsic_return={value:.3f}")

    gap = returns["ppo_only"] - returns["vanilla_rnd"]
    if gap <= 0:
        print("WARNING: PPO-only does not outperform vanilla RND; "
              "Gravitar failure not reproduced on this env. "
              "Try a higher IntCoef or a denser-reward env.")
        return 2

    closed = (returns["option_b"] - returns["vanilla_rnd"]) / gap
    print(f"gap closure: {closed:.1%} (threshold: >={threshold:.0%})")

    # Qualitative: gating_factor decay plot
    steps, gate = load_scalar(option_b_dir, "data/gating_factor")
    if steps and gate:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig_dir = Path(figures_dir)
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.plot(steps, gate)
            plt.xlabel("env steps")
            plt.ylabel("gating_factor (mean sigmoid applied to intrinsic)")
            plt.title("Option B intrinsic-gate decay")
            out = fig_dir / "gating_factor_decay.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"wrote {out}")
        except ImportError:
            print("matplotlib not installed; skipping gating_factor plot")
        print(f"gating_factor: start={gate[0]:.3f} end={gate[-1]:.3f}")
    else:
        print("note: data/gating_factor absent; skipping qualitative plot")

    if closed >= threshold:
        print("PASS")
        return 0
    print("FAIL")
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ppo-only-log-dir", required=True)
    p.add_argument("--vanilla-rnd-log-dir", required=True)
    p.add_argument("--option-b-log-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--figures-dir", default="scripts/figures")
    args = p.parse_args()
    sys.exit(evaluate(args.ppo_only_log_dir, args.vanilla_rnd_log_dir,
                      args.option_b_log_dir, args.threshold, args.figures_dir))


if __name__ == "__main__":
    main()
