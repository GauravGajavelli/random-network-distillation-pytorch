"""Experiment 1 verifier: Option B fixes the Pitfall-style failure on LavaCrossing.

Win condition: death_rate (averaged over the last 100 episodes) drops by
>=50% from vanilla RND to RND+Option B, with goal_reach_rate at least
~90% of the vanilla baseline.

Exit codes: 0 = PASS, 1 = FAIL, 2 = data missing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tb_utils import load_scalar, tail_mean


def evaluate(vanilla_dir: str, option_b_dir: str, threshold: float = 0.5) -> int:
    metrics = {}
    for name, log_dir in [("vanilla", vanilla_dir), ("option_b", option_b_dir)]:
        _, dr = load_scalar(log_dir, "data/death_rate")
        _, gr = load_scalar(log_dir, "data/goal_reach_rate")
        if not dr:
            print(f"FAIL: {name} run at {log_dir} has no data/death_rate scalar")
            return 2
        metrics[name] = {
            "death_rate": tail_mean(dr),
            "goal_reach_rate": tail_mean(gr) if gr else float("nan"),
        }

    print(f"vanilla RND:   death_rate={metrics['vanilla']['death_rate']:.3f}  "
          f"goal_reach_rate={metrics['vanilla']['goal_reach_rate']:.3f}")
    print(f"RND+Option B:  death_rate={metrics['option_b']['death_rate']:.3f}  "
          f"goal_reach_rate={metrics['option_b']['goal_reach_rate']:.3f}")

    if metrics["vanilla"]["death_rate"] <= 0:
        print("FAIL: vanilla death_rate is non-positive; failure not reproduced.")
        return 1

    reduction = (metrics["vanilla"]["death_rate"] - metrics["option_b"]["death_rate"]) \
        / metrics["vanilla"]["death_rate"]
    print(f"death_rate reduction: {reduction:.1%} (threshold: >={threshold:.0%})")

    goal_kept = metrics["option_b"]["goal_reach_rate"] >= 0.9 * metrics["vanilla"]["goal_reach_rate"]
    death_dropped = reduction >= threshold

    if death_dropped and goal_kept:
        print("PASS")
        return 0
    print(f"FAIL: death_dropped={death_dropped} goal_kept={goal_kept}")
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vanilla-log-dir", required=True)
    p.add_argument("--option-b-log-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="minimum death_rate reduction to PASS (default 0.5)")
    args = p.parse_args()
    sys.exit(evaluate(args.vanilla_log_dir, args.option_b_log_dir, args.threshold))


if __name__ == "__main__":
    main()
