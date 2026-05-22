"""3-way comparison: vanilla RND vs SimHash+RND vs SimHash-only on KeyCorridor.

Conditions
----------
  vanilla RND   : i_t = RND_normalized(s_{t+1})
  SimHash+RND   : i_t = RND_normalized(s_{t+1}) + beta * 1/sqrt(n(hash(s_{t+1})))
  SimHash-only  : i_t = beta * 1/sqrt(n(hash(s_{t+1})))   (UseRNDBonus=False)

  hash  = sign(A^T flatten(s)),  A ~ N(0,1) fixed D x obs_dim matrix
  n(·)  = global cumulative count, never reset across episodes
  beta  = SimHashLambda (config, default 0.5)
  D     = SimHashDim    (config, default 64)

The SimHash+RND combination is additive — the hash term cannot suppress the
RND term (C4-safety property). SimHash-only removes RND entirely to isolate
the hash bonus's independent contribution.

Environment: MiniGrid-KeyCorridorS3R1-v0

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

VANILLA_BASE  = 'exp3_keycorridor_baseline'
SIMHASH_BASE  = 'exp3_keycorridor_simhash'
ONLY_BASE     = 'exp3_keycorridor_simhash_only'


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


def pct_str(ref_m, cmp_m, direction='lower'):
    if math.isnan(ref_m) or math.isnan(cmp_m):
        return ''
    if ref_m in (float('inf'), 0.0) or cmp_m == float('inf'):
        return ''
    if direction == 'lower':
        pct = (ref_m - cmp_m) / ref_m * 100
        if pct > 3:
            return f'  -{pct:.1f}% vs van'
        elif pct < -3:
            return f'  +{-pct:.1f}% slower vs van'
    else:
        pct = (cmp_m - ref_m) / (abs(ref_m) + 1e-9) * 100
        if pct > 3:
            return f'  +{pct:.1f}% vs van'
    return ''


def verdict(label, ref_m, cmp_m, direction='lower'):
    if math.isnan(ref_m) or math.isnan(cmp_m):
        return f'  {label}: insufficient data'
    if ref_m == float('inf') and cmp_m == float('inf'):
        return f'  {label}: both fail to reach goal'
    if ref_m == float('inf'):
        return f'  {label}: reaches goal; vanilla does not'
    if cmp_m == float('inf'):
        return f'  {label}: vanilla reaches goal; {label} does not'
    if direction == 'lower':
        delta = (ref_m - cmp_m) / ref_m * 100
        if delta > 5:
            return f'  {label}: converges {delta:.1f}% faster than vanilla'
        elif delta < -5:
            return f'  {label}: vanilla {-delta:.1f}% faster'
        else:
            return f'  {label}: similar convergence ({delta:+.1f}%)'
    return ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir', default='runs')
    args = p.parse_args()
    runs_dir = Path(args.runs_dir)

    van  = find_seed_dirs(runs_dir, VANILLA_BASE)
    sim  = find_seed_dirs(runs_dir, SIMHASH_BASE)
    only = find_seed_dirs(runs_dir, ONLY_BASE)

    if not van and not sim and not only:
        print('No runs found. Start with:')
        print('  bash scripts/run_simhash_vs_vanilla.sh a  &   # baseline seeds 1+2')
        print('  bash scripts/run_simhash_vs_vanilla.sh b  &   # simhash+rnd seeds 1+2')
        print('  bash scripts/run_simhash_vs_vanilla.sh c  &   # simhash-only seeds 0+1+2')
        sys.exit(2)

    # For the two-way comparison (van vs sim) use only seeds present in both.
    matched_2 = sorted(set(van) & set(sim))
    # For simhash-only vs vanilla, use seeds present in both.
    matched_only = sorted(set(van) & set(only))

    print()
    print('=' * 78)
    print('3-way: vanilla RND vs SimHash+RND vs SimHash-only')
    print('ENV: MiniGrid-KeyCorridorS3R1-v0')
    print('=' * 78)
    print(f'  vanilla seeds   : {sorted(van.keys())}')
    print(f'  simhash+rnd     : {sorted(sim.keys())}')
    print(f'  simhash-only    : {sorted(only.keys())}')
    print(f'  matched (van∩sim)  : {matched_2}')
    print(f'  matched (van∩only) : {matched_only}')
    print()

    van_m2   = {s: van[s]  for s in matched_2    if s in van}
    sim_m2   = {s: sim[s]  for s in matched_2    if s in sim}
    van_mo   = {s: van[s]  for s in matched_only if s in van}
    only_mo  = {s: only[s] for s in matched_only if s in only}

    v2_t2g,  v2_extr,  v2_intr,  v2_uniq  = summarise(van_m2)
    s2_t2g,  s2_extr,  s2_intr,  s2_uniq  = summarise(sim_m2)
    vo_t2g,  vo_extr,  vo_intr,  _         = summarise(van_mo)
    o_t2g,   o_extr,   o_intr,   o_uniq    = summarise(only_mo)

    n2 = len(matched_2)
    no = len(matched_only)

    # --- Table: van vs sim (matched seeds) ---
    if matched_2:
        print(f'  ── vanilla RND vs SimHash+RND  (n={n2} matched seeds) ──')
        print(f'  {"Metric":<30} {"vanilla RND":>22}  {"SimHash+RND":>22}')
        print(f'  {"":─<30} {"":─>22}  {"":─>22}')
        rows = [
            ('Extr return  (tail mean)', stats(v2_extr), stats(s2_extr), 'float', 'higher'),
            ('Time to goal (steps)',     stats(v2_t2g),  stats(s2_t2g),  'steps', 'lower'),
            ('Int reward   (tail mean)', stats(v2_intr), stats(s2_intr), 'float', 'higher'),
            ('Unique hashes (final)',    stats(v2_uniq), stats(s2_uniq), 'float', 'higher'),
        ]
        for label, (vm, vs_), (sm, ss), kind, direction in rows:
            ann = pct_str(vm, sm, direction)
            print(f'  {label:<30} {fmt(vm, vs_, n2, kind):>22}  {fmt(sm, ss, n2, kind):>22}{ann}')
        print()

    # --- Table: van vs simhash-only (matched seeds) ---
    if matched_only:
        print(f'  ── vanilla RND vs SimHash-only  (n={no} matched seeds) ──')
        print(f'  {"Metric":<30} {"vanilla RND":>22}  {"SimHash-only":>22}')
        print(f'  {"":─<30} {"":─>22}  {"":─>22}')
        rows_o = [
            ('Extr return  (tail mean)', stats(vo_extr), stats(o_extr), 'float', 'higher'),
            ('Time to goal (steps)',     stats(vo_t2g),  stats(o_t2g),  'steps', 'lower'),
            ('Int reward   (tail mean)', stats(vo_intr), stats(o_intr), 'float', 'higher'),
            ('Unique hashes (final)',    (float('nan'), float('nan')), stats(o_uniq), 'float', 'higher'),
        ]
        for label, (vm, vs_), (om, os_), kind, direction in rows_o:
            ann = pct_str(vm, om, direction)
            print(f'  {label:<30} {fmt(vm, vs_, no, kind):>22}  {fmt(om, os_, no, kind):>22}{ann}')
        print()

    # --- Verdicts ---
    print('  VERDICTS')
    if matched_2:
        vm2, _ = stats(v2_t2g)
        sm2, _ = stats(s2_t2g)
        print(verdict('SimHash+RND', vm2, sm2, 'lower'))
    if matched_only:
        vmo, _ = stats(vo_t2g)
        omo, _ = stats(o_t2g)
        print(verdict('SimHash-only', vmo, omo, 'lower'))

    # --- SimHash+RND vs SimHash-only (if both have matching seeds) ---
    both_seeds = sorted(set(sim) & set(only))
    if both_seeds:
        sim_b  = {s: sim[s]  for s in both_seeds if s in sim}
        only_b = {s: only[s] for s in both_seeds if s in only}
        sb_t2g, _, _, _ = summarise(sim_b)
        ob_t2g, _, _, _ = summarise(only_b)
        sm_b, _ = stats(sb_t2g)
        om_b, _ = stats(ob_t2g)
        print()
        print(f'  ── SimHash+RND vs SimHash-only  (n={len(both_seeds)} matched seeds) ──')
        if not (math.isnan(sm_b) or math.isnan(om_b)) and sm_b not in (float('inf'),) and om_b not in (float('inf'),):
            delta = (om_b - sm_b) / om_b * 100
            if delta > 5:
                print(f'  SimHash+RND converges {delta:.1f}% faster than SimHash-only.')
                print(f'  The RND component adds meaningful signal beyond the hash bonus alone.')
            elif delta < -5:
                print(f'  SimHash-only converges {-delta:.1f}% faster — hash bonus is sufficient;')
                print(f'  the RND term may be acting as noise on this env.')
            else:
                print(f'  Similar convergence ({delta:+.1f}%). RND contribution is marginal on this env.')
        else:
            print(f'  Insufficient data for SimHash+RND vs SimHash-only comparison.')

    # --- Per-seed breakdown ---
    if n2 > 1 and matched_2:
        print()
        print('  Per-seed time-to-goal (steps) — van vs sim:')
        print(f'  {"seed":>6}  {"vanilla":>12}  {"simhash+rnd":>14}  {"delta":>10}')
        for seed, vt, st in zip(matched_2, v2_t2g[:n2], s2_t2g[:n2]):
            d = '--' if vt == float('inf') or st == float('inf') else f'{(vt - st) / vt * 100:+.1f}%'
            vt_s = '∞' if vt == float('inf') else f'{vt / 1000:.1f}k'
            st_s = '∞' if st == float('inf') else f'{st / 1000:.1f}k'
            print(f'  {seed:>6}  {vt_s:>12}  {st_s:>14}  {d:>10}')

    if no > 1 and matched_only:
        print()
        print('  Per-seed time-to-goal (steps) — van vs simhash-only:')
        print(f'  {"seed":>6}  {"vanilla":>12}  {"simhash-only":>14}  {"delta":>10}')
        for seed, vt, ot in zip(matched_only, vo_t2g[:no], o_t2g[:no]):
            d = '--' if vt == float('inf') or ot == float('inf') else f'{(vt - ot) / vt * 100:+.1f}%'
            vt_s = '∞' if vt == float('inf') else f'{vt / 1000:.1f}k'
            ot_s = '∞' if ot == float('inf') else f'{ot / 1000:.1f}k'
            print(f'  {seed:>6}  {vt_s:>12}  {ot_s:>14}  {d:>10}')

    print()
    print('=' * 78)


if __name__ == '__main__':
    main()
