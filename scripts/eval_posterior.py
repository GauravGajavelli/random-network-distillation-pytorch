"""Experiment 4 verifier: posterior sampling for deep exploration.

Tests the hypothesis that Bootstrap DQN-style posterior sampling on the
extrinsic critic ensemble provides "deep exploration" (trajectory-level
commitment to one value model) that per-step curiosity signals (RND,
NovelD, SimHash) lack.

Measures three things, each compared against a relevant baseline:

  1. POSTERIOR ALONE vs vanilla RND
     - Hypothesis: posterior sampling alone provides exploration commitment
       benefit even without curiosity enhancements
     - Metrics: time_to_first_goal, unique_positions_seen at fixed step
                budget, final extrinsic_return

  2. POSTERIOR + NOVELD vs NOVELD ALONE
     - Hypothesis: posterior sampling is orthogonal to NovelD's intrinsic
       reward shaping and provides additional benefit
     - Metrics: same

  3. POSTERIOR + SIMHASH vs SIMHASH ALONE
     - Hypothesis: posterior + SimHash also stack
     - Metrics: same

Each comparison can independently PASS, FAIL, or INCOMPLETE (missing data).
Exit code is 0 only if all available comparisons PASS.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tb_utils import load_scalar, tail_mean


def first_step_at_threshold(log_dir: str, tag: str, threshold: float) -> float:
    steps, vals = load_scalar(log_dir, tag)
    for s, v in zip(steps, vals):
        if v >= threshold:
            return s
    return float("inf")


def value_at_step(log_dir: str, tag: str, target_step: int) -> float:
    """Return the value at the most recent step <= target_step."""
    steps, vals = load_scalar(log_dir, tag)
    last = float("nan")
    for s, v in zip(steps, vals):
        if s > target_step:
            break
        last = v
    return last


def compare_pair(baseline_dir: str, intervention_dir: str,
                 baseline_label: str, intervention_label: str,
                 goal_threshold: float = 0.5,
                 coverage_step: int = 500000,
                 time_to_goal_ratio_target: float = 0.8) -> int:
    """Compare two runs along three deep-exploration axes.

    Returns 0 (pass), 1 (fail), or 2 (incomplete: baseline never reached goal).
    """
    print(f"--- {intervention_label} vs {baseline_label} ---")

    # 1. Time to first goal
    base_t = first_step_at_threshold(baseline_dir, "data/extrinsic_return", goal_threshold)
    int_t = first_step_at_threshold(intervention_dir, "data/extrinsic_return", goal_threshold)
    print(f"  time_to_first_goal (extr >= {goal_threshold}):")
    print(f"    {baseline_label}:     {base_t:.0f}")
    print(f"    {intervention_label}: {int_t:.0f}")

    # 2. Unique positions seen at a fixed step budget
    base_cov = value_at_step(baseline_dir, "data/unique_positions_seen", coverage_step)
    int_cov = value_at_step(intervention_dir, "data/unique_positions_seen", coverage_step)
    print(f"  unique_positions_seen at step {coverage_step}:")
    print(f"    {baseline_label}:     {base_cov:.0f}")
    print(f"    {intervention_label}: {int_cov:.0f}")

    # 3. Final extrinsic return
    _, base_extr = load_scalar(baseline_dir, "data/extrinsic_return")
    _, int_extr = load_scalar(intervention_dir, "data/extrinsic_return")
    base_final = tail_mean(base_extr) if base_extr else float("nan")
    int_final = tail_mean(int_extr) if int_extr else float("nan")
    print(f"  final extrinsic_return (last 100 episodes mean):")
    print(f"    {baseline_label}:     {base_final:.4f}")
    print(f"    {intervention_label}: {int_final:.4f}")

    # Verdict
    if base_t == float("inf"):
        print(f"  VERDICT: INCOMPLETE — baseline never reached goal threshold.")
        return 2
    if int_t == float("inf"):
        print(f"  VERDICT: FAIL — intervention never reached goal threshold.")
        return 1

    time_ratio = int_t / base_t
    coverage_ratio = (int_cov / base_cov) if base_cov > 0 else float("inf")
    print(f"  time_to_goal ratio: {time_ratio:.3f} (target <= {time_to_goal_ratio_target})")
    if base_cov > 0:
        print(f"  coverage ratio:     {coverage_ratio:.3f} (deep-exploration proxy; higher is better)")

    time_pass = time_ratio <= time_to_goal_ratio_target
    coverage_pass = base_cov <= 0 or coverage_ratio >= 1.0
    extrinsic_pass = int_final >= 0.9 * base_final

    if time_pass and (coverage_pass or extrinsic_pass):
        print(f"  VERDICT: PASS")
        return 0
    print(f"  VERDICT: FAIL  (time_pass={time_pass} coverage_pass={coverage_pass} extr_pass={extrinsic_pass})")
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vanilla-log-dir", required=True,
                   help="Vanilla RND baseline (e.g., runs/exp3_keycorridor_baseline)")
    p.add_argument("--posterior-log-dir", required=True,
                   help="RND + posterior sampling (e.g., runs/exp4_keycorridor_posterior)")
    p.add_argument("--noveld-log-dir", default=None,
                   help="NovelD alone (e.g., runs/exp3_keycorridor_noveld)")
    p.add_argument("--posterior-noveld-log-dir", default=None,
                   help="Posterior + NovelD (e.g., runs/exp4_keycorridor_posterior_noveld)")
    p.add_argument("--simhash-log-dir", default=None,
                   help="SimHash alone (e.g., runs/exp3_keycorridor_simhash)")
    p.add_argument("--posterior-simhash-log-dir", default=None,
                   help="Posterior + SimHash (e.g., runs/exp4_keycorridor_posterior_simhash)")
    p.add_argument("--goal-threshold", type=float, default=0.5)
    p.add_argument("--coverage-step", type=int, default=500000)
    p.add_argument("--time-ratio-target", type=float, default=0.8)
    args = p.parse_args()

    results = []

    # Comparison 1: posterior alone vs vanilla RND
    print("\n" + "=" * 60)
    print("Comparison 1: posterior alone vs vanilla RND")
    print("=" * 60)
    r1 = compare_pair(args.vanilla_log_dir, args.posterior_log_dir,
                      "vanilla_rnd", "posterior",
                      args.goal_threshold, args.coverage_step,
                      args.time_ratio_target)
    results.append(("posterior alone vs vanilla", r1))

    # Comparison 2: posterior + NovelD vs NovelD alone
    if args.noveld_log_dir and args.posterior_noveld_log_dir:
        print("\n" + "=" * 60)
        print("Comparison 2: posterior+NovelD vs NovelD alone")
        print("=" * 60)
        r2 = compare_pair(args.noveld_log_dir, args.posterior_noveld_log_dir,
                          "noveld", "posterior+noveld",
                          args.goal_threshold, args.coverage_step,
                          args.time_ratio_target)
        results.append(("posterior+NovelD vs NovelD", r2))

    # Comparison 3: posterior + SimHash vs SimHash alone
    if args.simhash_log_dir and args.posterior_simhash_log_dir:
        print("\n" + "=" * 60)
        print("Comparison 3: posterior+SimHash vs SimHash alone")
        print("=" * 60)
        r3 = compare_pair(args.simhash_log_dir, args.posterior_simhash_log_dir,
                          "simhash", "posterior+simhash",
                          args.goal_threshold, args.coverage_step,
                          args.time_ratio_target)
        results.append(("posterior+SimHash vs SimHash", r3))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, code in results:
        status = {0: "PASS", 1: "FAIL", 2: "INCOMPLETE"}[code]
        print(f"  {name}: {status}")
    sys.exit(0 if all(c == 0 for _, c in results) else 1)


if __name__ == "__main__":
    main()
