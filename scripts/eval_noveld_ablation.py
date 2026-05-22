"""NovelD cluster-type ablation: position keys vs RND-feature cluster keys.

Compares convergence speed (time_to_first_goal), final extrinsic return, and
intrinsic reward trajectory for:
  - NovelD with raw position keys  (UseNovelDClusters=False)
  - NovelD with cluster keys       (UseNovelDClusters=True, K=8)

across two environments that stress-test the clustering mechanism in
opposite directions:
  - LavaCrossing (small ~50-state grid): clustering may HURT by making the
    visit counter coarser than position keys, suppressing exploration too early
  - KeyCorridor (larger corridor): clustering may HELP by grouping similar
    corridor states that would otherwise never repeat within an episode

Usage:
    python scripts/eval_noveld_ablation.py
    python scripts/eval_noveld_ablation.py --runs-dir runs/
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tb_utils import load_scalar, tail_mean


RUNS = {
    'lava': {
        'pos':      'exp1_lava_noveld_pos',
        'clusters': 'exp1_lava_noveld',
    },
    'keycorridor': {
        'pos':      'exp3_keycorridor_noveld_pos',
        'clusters': 'exp3_keycorridor_noveld',
    },
}

ENV_LABEL = {
    'lava':         'LavaCrossing S9N2  (small, ~50 states)',
    'keycorridor':  'KeyCorridor S3R1   (larger corridor)',
}

METHOD_LABEL = {
    'pos':      'NovelD + position keys',
    'clusters': 'NovelD + cluster keys (K=8)',
}


def find_seed_dirs(runs_dir: Path, base_name: str):
    """Return sorted list of (seed, path) for bare and _seedN variants."""
    found = {}
    bare = runs_dir / base_name
    if bare.exists():
        found[0] = bare
    for p in sorted(runs_dir.glob(f'{base_name}_seed*')):
        if not p.is_dir():
            continue
        try:
            seed = int(p.name[len(base_name) + len('_seed'):])
            found[seed] = p
        except ValueError:
            continue
    return sorted(found.items())


def first_step_at(steps, vals, threshold):
    for s, v in zip(steps, vals):
        if v >= threshold:
            return float(s)
    return float('inf')


def pull(run_path, tag):
    steps, vals = load_scalar(str(run_path), tag)
    return steps, vals


def summarise_seeds(seed_paths, tag, reduction='tail_mean'):
    per_seed = {}
    for seed, path in seed_paths:
        steps, vals = pull(path, tag)
        if not vals:
            continue
        if reduction == 'tail_mean':
            per_seed[seed] = tail_mean(vals)
        elif reduction == 'time_to_goal':
            per_seed[seed] = first_step_at(steps, vals, 0.5)
        elif reduction == 'max':
            per_seed[seed] = max(vals)
    if not per_seed:
        return float('nan'), float('nan'), []
    vals = list(per_seed.values())
    finite = [v for v in vals if v != float('inf') and not math.isnan(v)]
    if not finite:
        return float('inf'), 0.0, list(per_seed.keys())
    mean = sum(finite) / len(finite)
    std = math.sqrt(sum((v - mean) ** 2 for v in finite) / len(finite))
    return mean, std, list(per_seed.keys())


def fmt(mean, std, seeds, kind='float'):
    if math.isnan(mean):
        return f'{"--":>12}  (n={len(seeds)})'
    if mean == float('inf'):
        return f'{"inf":>12}  (n={len(seeds)})'
    if kind == 'steps':
        s = f'{mean / 1000:.1f}k'
        e = f'±{std / 1000:.1f}k' if len(seeds) > 1 else ''
    else:
        s = f'{mean:.3f}'
        e = f'±{std:.3f}' if len(seeds) > 1 else ''
    return f'{(s + e):>14}  (n={len(seeds)}, seeds={seeds})'


def print_separator(char='─', width=80):
    print(char * width)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir', default='runs')
    args = p.parse_args()
    runs_dir = Path(args.runs_dir)

    print()
    print('=' * 80)
    print('NovelD Ablation: position keys vs cluster keys')
    print('=' * 80)

    any_data = False

    for env_key, methods in RUNS.items():
        print()
        print_separator('─')
        print(f'ENV: {ENV_LABEL[env_key]}')
        print_separator('─')

        # Check which seeds exist
        seed_dirs = {}
        for method_key, base_name in methods.items():
            seed_dirs[method_key] = find_seed_dirs(runs_dir, base_name)
            status = (f'seeds {[s for s, _ in seed_dirs[method_key]]}'
                      if seed_dirs[method_key] else 'NOT FOUND')
            print(f'  {METHOD_LABEL[method_key]:<35} {status}')

        # Verify matched seeds
        seed_sets = {m: set(s for s, _ in sd) for m, sd in seed_dirs.items()}
        if all(seed_sets.values()):
            matched = set.intersection(*seed_sets.values())
        else:
            matched = set()

        if not matched and not any(seed_dirs.values()):
            print('  (no runs found for this env)')
            continue

        if matched != set.union(*seed_sets.values()) if seed_sets else set():
            missing = set.union(*seed_sets.values()) - matched if seed_sets else set()
            if missing:
                print(f'  NOTE: unmatched seeds {sorted(missing)} excluded from comparison')

        any_data = True
        print()

        metrics = [
            ('Extrinsic return (tail mean)', 'data/extrinsic_return', 'tail_mean', 'float'),
            ('Time to first goal (steps)',   'data/extrinsic_return', 'time_to_goal', 'steps'),
            ('Intrinsic reward (tail mean)', 'data/int_reward_per_rollout', 'tail_mean', 'float'),
            ('Goal reach rate (tail mean)',  'data/goal_reach_rate', 'tail_mean', 'float'),
        ]

        # Use all available seeds for each method (not just matched) so partial
        # data is still surfaced, but flag it clearly.
        results = {}
        for method_key, sd in seed_dirs.items():
            results[method_key] = {}
            for label, tag, reduction, kind in metrics:
                mean, std, seeds = summarise_seeds(sd, tag, reduction)
                results[method_key][label] = (mean, std, seeds, kind)

        print(f'  {"Metric":<35} {"pos keys":>22}  {"cluster keys":>22}')
        print(f'  {"":─<35} {"":─>22}  {"":─>22}')

        for label, tag, reduction, kind in metrics:
            pos_m, pos_s, pos_seeds, _ = results['pos'][label]
            cl_m, cl_s, cl_seeds, _ = results['clusters'][label]

            better = ''
            if not math.isnan(pos_m) and not math.isnan(cl_m):
                if pos_m != float('inf') and cl_m != float('inf'):
                    if reduction == 'time_to_goal':
                        if cl_m < pos_m * 0.95:
                            better = '  <- clusters faster'
                        elif pos_m < cl_m * 0.95:
                            better = '  <- pos faster'
                    else:
                        if cl_m > pos_m * 1.02:
                            better = '  <- clusters better'
                        elif pos_m > cl_m * 1.02:
                            better = '  <- pos better'

            print(f'  {label:<35} {fmt(pos_m, pos_s, pos_seeds, kind):>22}  {fmt(cl_m, cl_s, cl_seeds, kind):>22}{better}')

        # Convergence speed conclusion
        print()
        pos_t = results['pos']['Time to first goal (steps)'][0]
        cl_t  = results['clusters']['Time to first goal (steps)'][0]
        if not math.isnan(pos_t) and not math.isnan(cl_t):
            if pos_t == float('inf') and cl_t == float('inf'):
                print('  VERDICT: both fail to reach goal within budget.')
            elif pos_t == float('inf'):
                print('  VERDICT: clusters reach goal; position keys do not within budget.')
            elif cl_t == float('inf'):
                print('  VERDICT: position keys reach goal; clusters do not within budget.')
            else:
                delta_pct = (pos_t - cl_t) / pos_t * 100
                if abs(delta_pct) < 5:
                    print(f'  VERDICT: convergence speed is similar ({delta_pct:+.1f}%).')
                elif delta_pct > 0:
                    print(f'  VERDICT: clusters are {delta_pct:.1f}% faster to first goal — clustering HELPS on this env.')
                else:
                    print(f'  VERDICT: position keys are {-delta_pct:.1f}% faster to first goal — clustering HURTS on this env.')

    print()
    print('=' * 80)
    print('CLUSTERING SUSCEPTIBILITY TO SPARSE REWARD')
    print('=' * 80)
    print("""
Known risks:

  1. Cold-start collapse (first ClusterRefreshSteps=4096 env steps):
     cluster_filled==0 so every state maps to key ('cluster','__none__').
     All envs share one counter; the multiplier becomes 1/sqrt(episode_step)
     — a time-based decay rather than a state-novelty signal. Affects the
     first ~20 episodes before the first recluster fires.

  2. Coarse-graining on small envs (the primary risk):
     K=8 clusters over LavaCrossing's ~50 reachable states = ~6 states per
     cluster. The agent visits each cluster within a few steps, driving
     1/sqrt(N) to near-zero quickly — faster than position keys would. In
     a sparse-reward env this suppresses the exploration bonus before the
     agent has found external reward.

  3. Cluster refresh discontinuity:
     After each recluster, a state previously mapped to cluster-3 (N=40
     visits) may remap to cluster-6 (N=0 visits), effectively resetting
     its count. This re-ignites exploration of already-visited state types.
     Can be beneficial in large envs; is noise in small ones.

  Where clustering is expected to HELP:
     Envs where the position-key space is large enough that most states
     are only visited once per episode (KeyCorridor, larger grids).
     Clustering groups similar-looking states so experience can pool.

  Where clustering is expected to HURT:
     Small envs (LavaCrossing, DoorKey) where K clusters covers the
     reachable space coarsely and count decay is faster than with raw
     position keys.
""")

    if not any_data:
        print('No runs found. Run the ablation first:')
        print('  bash scripts/run_noveld_ablation.sh a  # terminal 1')
        print('  bash scripts/run_noveld_ablation.sh b  # terminal 2')
        print('  bash scripts/run_noveld_ablation.sh c  # terminal 3')
        sys.exit(2)


if __name__ == '__main__':
    main()
