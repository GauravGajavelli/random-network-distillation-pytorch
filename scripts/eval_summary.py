"""Unified evaluation across all experiments and enhancement variants.

Auto-discovers run directories under runs/, parses their names into
(experiment, env, method), groups by environment, and prints:

  1. Per-env comparison tables of all available methods across the key
     metrics (extr_return, goal_rate, death_rate, time_to_goal, unique
     positions seen, int_reward).
  2. Cross-env "wins" count showing which method dominates on each metric
     across all environments.
  3. Literature-baseline PASS/FAIL comparisons for the canonical hypotheses
     we set out to test.

Tolerant of missing runs — skips comparisons whose runs aren't present.
Use this once after the sweep to see all results in one place.

Usage:
    python scripts/eval_summary.py
    python scripts/eval_summary.py --runs-dir runs/
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tb_utils import load_scalar, tail_mean


# Method canonical names -> display labels and ordering
METHOD_LABELS = {
    'ppo': 'PPO-only',
    'vanilla': 'vanilla RND',
    'vanilla_rnd': 'vanilla RND',
    'baseline': 'vanilla RND (baseline)',
    'option_b': 'Option B (legacy)',
    'dsc': 'DSC (legacy)',
    'dsc_typed': 'DSC-typed (legacy)',
    'dsc_tv_off': 'DSC TV-off (legacy)',
    'noveld': 'NovelD-clustered',
    'simhash': 'SimHash-additive',
    'simhash_tv': 'SimHash + TV (diagnostic)',
    'posterior': 'Posterior sampling',
    'posterior_noveld': 'Posterior + NovelD',
    'posterior_simhash': 'Posterior + SimHash',
}

METHOD_ORDER = [
    'ppo', 'vanilla', 'vanilla_rnd', 'baseline',
    'option_b', 'dsc', 'dsc_typed', 'dsc_tv_off',
    'noveld', 'simhash', 'simhash_tv',
    'posterior', 'posterior_noveld', 'posterior_simhash',
]

ENV_LABELS = {
    'lava': 'MiniGrid-LavaCrossingS9N2-v0  (Exp 1: Pitfall proxy)',
    'doorkey': 'MiniGrid-DoorKey-5x5-v0  (Exp 2: Gravitar proxy)',
    'keycorridor': 'MiniGrid-KeyCorridorS3R1-v0  (Exp 3 + 4)',
}

ENV_ORDER = ['lava', 'doorkey', 'keycorridor']


def parse_run_name(name):
    """Parse 'exp1_lava_vanilla_rnd' -> ('1', 'lava', 'vanilla_rnd'). None if unparseable."""
    parts = name.split('_')
    if len(parts) < 3 or not parts[0].startswith('exp'):
        return None
    exp = parts[0][3:]
    env = parts[1]
    method = '_'.join(parts[2:])
    return (exp, env, method)


def first_step_at_threshold(steps, vals, threshold):
    for s, v in zip(steps, vals):
        if v >= threshold:
            return float(s)
    return float('inf')


def value_at_step(steps, vals, target_step):
    last = float('nan')
    for s, v in zip(steps, vals):
        if s > target_step:
            break
        last = float(v)
    return last


def extract_metrics(log_dir):
    """Pull all metrics we care about from a TB log. Missing -> nan/inf."""
    out = {}
    steps, extr = load_scalar(log_dir, 'data/extrinsic_return')
    out['extr_return'] = tail_mean(extr) if extr else float('nan')
    out['time_to_goal'] = first_step_at_threshold(steps, extr, 0.5) if extr else float('inf')
    out['final_step'] = steps[-1] if steps else 0

    _, goal = load_scalar(log_dir, 'data/goal_reach_rate')
    out['goal_rate'] = tail_mean(goal) if goal else float('nan')

    _, death = load_scalar(log_dir, 'data/death_rate')
    out['death_rate'] = tail_mean(death) if death else float('nan')

    upos_steps, upos = load_scalar(log_dir, 'data/unique_positions_seen')
    out['uniq_pos_500k'] = value_at_step(upos_steps, upos, 500_000) if upos else float('nan')

    _, intr = load_scalar(log_dir, 'data/int_reward_per_rollout')
    out['int_reward'] = tail_mean(intr) if intr else float('nan')
    return out


def discover_runs(runs_dir):
    runs = {}
    for path in sorted(runs_dir.glob('exp*_*')):
        if not path.is_dir():
            continue
        parsed = parse_run_name(path.name)
        if parsed is None:
            continue
        runs[parsed] = str(path)
    return runs


def fmt_val(v, is_best):
    """Format a scalar value with best-in-column marker."""
    if v != v:  # nan
        s = '    --  '
    elif v == float('inf'):
        s = '    inf '
    elif abs(v) >= 1000:
        s = f'{v / 1000:>6.1f}k '
    elif abs(v) >= 1:
        s = f'{v:>7.2f} '
    else:
        s = f'{v:>7.3f} '
    return s[:-1] + ('*' if is_best else ' ')


def best_method_for_metric(metrics_by_method, key, direction):
    """Return method name with best value, or None if all nan/inf."""
    candidates = []
    for m, mdict in metrics_by_method.items():
        v = mdict[key]
        if v != v:  # nan
            continue
        if direction == 'higher' and v == float('-inf'):
            continue
        if direction == 'lower' and v == float('inf'):
            continue
        candidates.append((m, v))
    if not candidates:
        return None
    if direction == 'higher':
        return max(candidates, key=lambda x: x[1])[0]
    return min(candidates, key=lambda x: x[1])[0]


def print_env_table(env_key, runs_for_env):
    """Print comparison table for one env. Returns wins-per-method dict."""
    print()
    print('=' * 80)
    print(f'ENV: {ENV_LABELS.get(env_key, env_key)}')
    print('=' * 80)

    if not runs_for_env:
        print('  (no runs found)')
        return {}

    metrics = {method: extract_metrics(log_dir)
               for (_e, _v, method), log_dir in runs_for_env.items()}

    # Column spec: (metric_key, display_label, direction)
    cols = [('extr_return', 'extr_return', 'higher'),
            ('goal_rate', 'goal_rate', 'higher'),
            ('time_to_goal', 'time2goal', 'lower'),
            ('int_reward', 'int_reward', 'higher'),
            ('uniq_pos_500k', 'uniq_pos', 'higher')]
    if env_key == 'lava':
        cols.insert(3, ('death_rate', 'death_rate', 'lower'))

    best = {key: best_method_for_metric(metrics, key, direction)
            for key, _, direction in cols}

    # Header
    header = f'  {"Method":<26}'
    for _, label, _ in cols:
        header += f' {label:>11}'
    print(header)
    print('  ' + '-' * (26 + 12 * len(cols)))

    # Rows in canonical order
    ordered = [m for m in METHOD_ORDER if m in metrics]
    extras = [m for m in metrics if m not in METHOD_ORDER]

    wins = {m: 0 for m in metrics}
    for m in ordered + extras:
        row = f'  {METHOD_LABELS.get(m, m):<26}'
        for key, _, _ in cols:
            is_best = best.get(key) == m
            row += f' {fmt_val(metrics[m][key], is_best):>11}'
            if is_best:
                wins[m] += 1
        print(row)

    print()
    print('  * = best in column.')
    return wins


def print_cross_env(all_wins):
    print()
    print('=' * 80)
    print('CROSS-ENV WIN COUNTS')
    print('=' * 80)
    print('  (Methods with the most "best in column" wins across all environments)')
    print()

    all_methods = set()
    for ws in all_wins.values():
        all_methods.update(ws.keys())

    envs_present = [e for e in ENV_ORDER if e in all_wins]

    header = f'  {"Method":<26}'
    for e in envs_present:
        header += f' {e:>13}'
    header += f' {"Total":>10}'
    print(header)
    print('  ' + '-' * (26 + 14 * len(envs_present) + 11))

    ordered = [m for m in METHOD_ORDER if m in all_methods]
    extras = sorted(m for m in all_methods if m not in METHOD_ORDER)

    totals = []
    for m in ordered + extras:
        row = f'  {METHOD_LABELS.get(m, m):<26}'
        total = 0
        for e in envs_present:
            w = all_wins.get(e, {}).get(m, None)
            row += f' {("-" if w is None else str(w)):>13}'
            if w is not None:
                total += w
        row += f' {total:>10}'
        totals.append((m, total))
        print(row)

    totals.sort(key=lambda x: -x[1])
    if totals and totals[0][1] > 0:
        print()
        print(f'  Best overall: {METHOD_LABELS.get(totals[0][0], totals[0][0])} '
              f'({totals[0][1]} wins)')


def run_canonical_comparisons(runs):
    """The PASS/FAIL story for each enhancement vs its appropriate baseline."""
    print()
    print('=' * 80)
    print('LITERATURE-BASELINE PASS/FAIL COMPARISONS')
    print('=' * 80)

    def get(exp, env, method):
        return runs.get((exp, env, method))

    def safe_extract(d):
        return extract_metrics(d) if d else None

    # Define comparisons: (label, baseline, intervention, predicate, expected_direction)
    # predicate takes (baseline_metrics, intervention_metrics) and returns True for PASS.
    # expected_direction: 'improvement' (normal) or 'regression' (diagnostic — PASS if worse).
    comparisons = [
        ('Exp 1 NovelD vs vanilla',
         get('1', 'lava', 'vanilla'), get('1', 'lava', 'noveld'),
         lambda b, i: (i['goal_rate'] >= b['goal_rate'])
                       and (i['time_to_goal'] <= b['time_to_goal']),
         'improvement'),

        ('Exp 1 SimHash vs vanilla',
         get('1', 'lava', 'vanilla'), get('1', 'lava', 'simhash'),
         lambda b, i: (i['goal_rate'] >= b['goal_rate'] * 0.9)
                       and (i['extr_return'] >= b['extr_return'] * 0.9),
         'improvement'),

        ('Exp 2 NovelD vs vanilla RND',
         get('2', 'doorkey', 'vanilla_rnd'), get('2', 'doorkey', 'noveld'),
         lambda b, i: i['extr_return'] >= b['extr_return'],
         'improvement'),

        ('Exp 2 SimHash vs vanilla RND',
         get('2', 'doorkey', 'vanilla_rnd'), get('2', 'doorkey', 'simhash'),
         lambda b, i: i['extr_return'] >= b['extr_return'],
         'improvement'),

        ('Exp 3 NovelD vs baseline',
         get('3', 'keycorridor', 'baseline'), get('3', 'keycorridor', 'noveld'),
         lambda b, i: i['time_to_goal'] <= b['time_to_goal']
                       and i['extr_return'] >= b['extr_return'] * 0.9,
         'improvement'),

        ('Exp 3 SimHash vs baseline',
         get('3', 'keycorridor', 'baseline'), get('3', 'keycorridor', 'simhash'),
         lambda b, i: i['time_to_goal'] <= b['time_to_goal']
                       and i['extr_return'] >= b['extr_return'] * 0.9,
         'improvement'),

        ('Exp 4 posterior alone vs vanilla baseline',
         get('3', 'keycorridor', 'baseline'), get('4', 'keycorridor', 'posterior'),
         lambda b, i: i['uniq_pos_500k'] >= b['uniq_pos_500k']
                       and i['time_to_goal'] <= b['time_to_goal'],
         'improvement'),

        ('Exp 4 posterior+NovelD stacking vs NovelD alone',
         get('3', 'keycorridor', 'noveld'), get('4', 'keycorridor', 'posterior_noveld'),
         lambda b, i: i['time_to_goal'] <= b['time_to_goal'] * 1.1
                       and i['uniq_pos_500k'] >= b['uniq_pos_500k'] * 0.95,
         'improvement'),

        ('Exp 4 posterior+SimHash stacking vs SimHash alone',
         get('3', 'keycorridor', 'simhash'), get('4', 'keycorridor', 'posterior_simhash'),
         lambda b, i: i['time_to_goal'] <= b['time_to_goal'] * 1.1
                       and i['uniq_pos_500k'] >= b['uniq_pos_500k'] * 0.95,
         'improvement'),

        ('Exp 3 SimHash TV-vulnerability diagnostic (expect REGRESSION)',
         get('3', 'keycorridor', 'simhash'), get('3', 'keycorridor', 'simhash_tv'),
         lambda b, i: i['extr_return'] < b['extr_return'] * 0.9
                       or i['time_to_goal'] > b['time_to_goal'] * 1.2,
         'regression'),
    ]

    results = []
    for label, b_dir, i_dir, pred, expected in comparisons:
        if b_dir is None or i_dir is None:
            status = 'INCOMPLETE'
            print(f'  {label}')
            print(f'    {"missing run(s)":>50}  ->  {status}')
            results.append((label, status))
            continue
        b_m = extract_metrics(b_dir)
        i_m = extract_metrics(i_dir)
        try:
            ok = pred(b_m, i_m)
        except (TypeError, ValueError, KeyError):
            ok = False
        status = 'PASS' if ok else 'FAIL'
        note = f'(b extr={b_m["extr_return"]:.3f} t2g={b_m["time_to_goal"]:.0f}  '\
               f'i extr={i_m["extr_return"]:.3f} t2g={i_m["time_to_goal"]:.0f})'
        marker = '[diagnostic]' if expected == 'regression' else ''
        print(f'  {label} {marker}')
        print(f'    {note:>50}  ->  {status}')
        results.append((label, status))

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir', default='runs',
                   help='Directory of TB run dirs (default: runs/)')
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f'runs dir not found: {runs_dir}')
        sys.exit(2)

    runs = discover_runs(runs_dir)
    if not runs:
        print(f'No runs found in {runs_dir}')
        sys.exit(2)

    print(f'Discovered {len(runs)} runs:')
    for k in sorted(runs):
        print(f'  exp{k[0]}_{k[1]}_{k[2]}')

    # Group by env
    by_env = defaultdict(dict)
    for key, log_dir in runs.items():
        _exp, env, _method = key
        by_env[env][key] = log_dir

    # Per-env tables
    all_wins = {}
    for env_key in ENV_ORDER:
        if env_key in by_env:
            wins = print_env_table(env_key, by_env[env_key])
            all_wins[env_key] = wins
    # Any envs not in the canonical order
    for env_key in by_env:
        if env_key not in ENV_ORDER:
            wins = print_env_table(env_key, by_env[env_key])
            all_wins[env_key] = wins

    # Cross-env summary
    print_cross_env(all_wins)

    # Pass/fail
    results = run_canonical_comparisons(runs)

    # Final summary line
    print()
    print('=' * 80)
    counts = {'PASS': 0, 'FAIL': 0, 'INCOMPLETE': 0}
    for _, status in results:
        counts[status] = counts.get(status, 0) + 1
    print(f'FINAL: {counts["PASS"]} PASS, {counts["FAIL"]} FAIL, '
          f'{counts["INCOMPLETE"]} INCOMPLETE')
    print('=' * 80)

    sys.exit(1 if counts['FAIL'] > 0 else 0)


if __name__ == '__main__':
    main()
