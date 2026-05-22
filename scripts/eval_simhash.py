"""SimHash-additive RND vs vanilla RND: matched-seed evaluation.

Model under test
----------------
    i_t = RND_normalized(s_{t+1})  +  beta * 1/sqrt(n(hash(s_{t+1})))

    hash  = sign(A^T flatten(s)),  A ~ N(0,1) fixed D x obs_dim matrix
    n(·)  = global cumulative count, never reset across episodes
    beta  = SimHashLambda (config, default 0.5)
    D     = SimHashDim    (config, default 64)

The two terms are combined additively. The hash term cannot suppress or
gate the RND term; its worst case is additive noise (C4-safety property).

Environment: MiniGrid-KeyCorridorS3R1-v0  (the most discriminative env;
SimHash showed -14% convergence vs vanilla at seed 0).

Usage:
    python scripts/eval_simhash.py
    python scripts/eval_simhash.py --runs-dir runs/
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tb_utils import load_scalar, tail_mean

VANILLA_BASE = 'exp3_keycorridor_baseline'
SIMHASH_BASE = 'exp3_keycorridor_simhash'


def find_seed_dirs(runs_dir, base):
    found = {}
    bare = runs_dir / base
    if bare.exists():
        found[0] = bare
    for p in sorted(runs_dir.glob(f'{base}_seed*')):
        if not p.is_dir():
            continue
        try:
            seed = int(p.name[len(base) + len('_seed'):])
            found[seed] = p
        except ValueError:
            pass
    return found


def first_step_at(steps, vals, threshold=0.5):
    for s, v in zip(steps, vals):
        if v >= threshold:
            return float(s)
    return float('inf')


def pull(path, tag):
    return load_scalar(str(path), tag)


def stats(values):
    finite = [v for v in values if v not in (float('inf'), float('nan')) and not math.isnan(v)]
    if not finite:
        return float('nan'), float('nan')
    m = sum(finite) / len(finite)
    s = math.sqrt(sum((v - m) ** 2 for v in finite) / len(finite))
    return m, s


def fmt(m, s, n, kind='float'):
    if math.isnan(m):
        return f'{"--":>10}        (n={n})'
    if m == float('inf'):
        return f'{"∞":>10}        (n={n})'
    if kind == 'steps':
        base = f'{m / 1000:.1f}k'
        err = f' ±{s / 1000:.1f}k' if n > 1 else ''
    else:
        base = f'{m:.3f}'
        err = f' ±{s:.3f}' if n > 1 else ''
    return f'{base + err:>16}  (n={n})'


def summarise(seed_paths):
    t2g, extr, intr, uniq = [], [], [], []
    for _seed, path in sorted(seed_paths.items()):
        steps, vals = pull(path, 'data/extrinsic_return')
        if vals:
            extr.append(tail_mean(vals))
            t2g.append(first_step_at(steps, vals))
        else:
            extr.append(float('nan'))
            t2g.append(float('inf'))
        _, v = pull(path, 'data/int_reward_per_rollout')
        intr.append(tail_mean(v) if v else float('nan'))
        _, v = pull(path, 'data/simhash_unique_hashes')
        uniq.append(v[-1] if v else float('nan'))
    return t2g, extr, intr, uniq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir', default='runs')
    args = p.parse_args()
    runs_dir = Path(args.runs_dir)

    van = find_seed_dirs(runs_dir, VANILLA_BASE)
    sim = find_seed_dirs(runs_dir, SIMHASH_BASE)

    if not van and not sim:
        print('No runs found. Start with:')
        print('  bash scripts/run_simhash_vs_vanilla.sh a  &')
        print('  bash scripts/run_simhash_vs_vanilla.sh b  &')
        print('Or on gebru:')
        print('  bash cuda/run_simhash_vs_vanilla.sh a  &')
        print('  bash cuda/run_simhash_vs_vanilla.sh b  &')
        sys.exit(2)

    matched = sorted(set(van) & set(sim))
    unmatched = sorted((set(van) | set(sim)) - set(matched))

    print()
    print('=' * 70)
    print('SimHash-additive RND vs vanilla RND')
    print('ENV: MiniGrid-KeyCorridorS3R1-v0')
    print('=' * 70)
    print(f'  i_t = RND_normalized  +  beta * 1/sqrt(n(hash(s_t+1)))')
    print(f'  beta=SimHashLambda (default 0.5)   D=SimHashDim (default 64)')
    print()
    print(f'  vanilla seeds found : {sorted(van.keys())}')
    print(f'  simhash seeds found : {sorted(sim.keys())}')
    print(f'  matched seeds       : {matched}')
    if unmatched:
        print(f'  NOTE: seeds {unmatched} present in only one condition — excluded')
    print()

    van_matched = {s: van[s] for s in matched if s in van}
    sim_matched = {s: sim[s] for s in matched if s in sim}

    v_t2g, v_extr, v_intr, v_uniq = summarise(van_matched)
    s_t2g, s_extr, s_intr, s_uniq = summarise(sim_matched)

    n = len(matched)

    rows = [
        ('Extr return  (tail mean)', stats(v_extr), stats(s_extr), 'float', 'higher'),
        ('Time to goal (steps)',     stats(v_t2g),  stats(s_t2g),  'steps', 'lower'),
        ('Int reward   (tail mean)', stats(v_intr), stats(s_intr), 'float', 'higher'),
        ('Unique hashes (final)',    stats(v_uniq), stats(s_uniq), 'float', 'higher'),
    ]

    print(f'  {"Metric":<30} {"vanilla RND":>22}  {"SimHash+RND":>22}')
    print(f'  {"":─<30} {"":─>22}  {"":─>22}')

    for label, (vm, vs), (sm, ss), kind, direction in rows:
        better = ''
        if not (math.isnan(vm) or math.isnan(sm)):
            if direction == 'lower' and sm not in (float('inf'),) and vm not in (float('inf'),):
                pct = (vm - sm) / vm * 100 if vm else 0
                if pct > 3:
                    better = f'  simhash -{pct:.1f}%'
                elif pct < -3:
                    better = f'  vanilla  {-pct:.1f}% faster'
            elif direction == 'higher' and not math.isnan(vm) and not math.isnan(sm):
                pct = (sm - vm) / (abs(vm) + 1e-9) * 100
                if pct > 3:
                    better = f'  simhash +{pct:.1f}%'
        print(f'  {label:<30} {fmt(vm, vs, n, kind):>22}  {fmt(sm, ss, n, kind):>22}{better}')

    print()
    vm, vs = stats(v_t2g)
    sm, ss = stats(s_t2g)
    print('  VERDICT')
    if math.isnan(vm) or math.isnan(sm):
        print('  Insufficient data for verdict.')
    elif vm == float('inf') and sm == float('inf'):
        print('  Both conditions fail to reach goal within the training budget.')
    elif vm == float('inf'):
        print('  SimHash reaches goal; vanilla does not within budget.')
    elif sm == float('inf'):
        print('  Vanilla reaches goal; SimHash does not — additive hash may be')
        print('  distorting the intrinsic signal on this env/seed combination.')
    else:
        delta = (vm - sm) / vm * 100
        if delta > 5:
            print(f'  SimHash converges {delta:.1f}% faster than vanilla (matched seeds, n={n}).')
            print(f'  The additive hash provides a complementary coverage signal')
            print(f'  without suppressing RND — consistent with the C4-safety property.')
        elif delta < -5:
            print(f'  Vanilla converges {-delta:.1f}% faster. The hash bonus is acting')
            print(f'  as noise rather than signal on this env at these seeds.')
        else:
            print(f'  Convergence speed is similar ({delta:+.1f}%). Final returns are')
            print(f'  comparable; the hash term neither helps nor hurts significantly.')

    # Per-seed breakdown for transparency
    if n > 1:
        print()
        print('  Per-seed time-to-goal (steps):')
        print(f'  {"seed":>6}  {"vanilla":>12}  {"simhash":>12}  {"delta":>10}')
        for seed, vt, st in zip(matched, v_t2g[:n], s_t2g[:n]):
            if vt == float('inf') or st == float('inf'):
                d = '--'
            else:
                d = f'{(vt - st) / vt * 100:+.1f}%'
            vt_s = '∞' if vt == float('inf') else f'{vt / 1000:.1f}k'
            st_s = '∞' if st == float('inf') else f'{st / 1000:.1f}k'
            print(f'  {seed:>6}  {vt_s:>12}  {st_s:>12}  {d:>10}')

    print()
    print('=' * 70)


if __name__ == '__main__':
    main()
