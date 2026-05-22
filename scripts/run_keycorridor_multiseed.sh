#!/usr/bin/env bash
# Multi-seed sweep for KeyCorridor quick-convergers.
#
# Runs the three methods that showed meaningful convergence-speed gains on
# MiniGrid-KeyCorridorS3R1-v0, plus the missing Posterior+SimHash stacking
# run, across seeds 0-2 so matched-seed comparisons are valid.
#
# Seed-0 runs already exist (bare names without _seed0 suffix). This script
# adds seeds 1 and 2 for the existing methods and all three seeds for the
# new Posterior+SimHash stacking condition.
#
# New runs produced (9 total, ~60 min each on M1 Pro):
#   exp3_keycorridor_baseline_seed1/2       (control, matched seeds)
#   exp3_keycorridor_simhash_seed1/2        (best standalone: -14% time2goal)
#   exp4_keycorridor_posterior_seed1/2      (competitive: -11% time2goal)
#   exp4_keycorridor_posterior_simhash_seed0/1/2   (stacking, all seeds missing)
#
# Run in three terminals for ~3-hour wall time:
#   bash scripts/run_keycorridor_multiseed.sh a   # terminal 1
#   bash scripts/run_keycorridor_multiseed.sh b   # terminal 2
#   bash scripts/run_keycorridor_multiseed.sh c   # terminal 3
#
# Or log to files:
#   bash scripts/run_keycorridor_multiseed.sh a > /tmp/kc_a.log 2>&1 &
#   bash scripts/run_keycorridor_multiseed.sh b > /tmp/kc_b.log 2>&1 &
#   bash scripts/run_keycorridor_multiseed.sh c > /tmp/kc_c.log 2>&1 &
#   wait
#   tail /tmp/kc_*.log

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <a|b|c>" >&2
    exit 2
fi

CHUNK="$1"

# Format: "SECTION:run_name:SEED"
# Seed-0 already exists for baseline/simhash/posterior — only adding 1 and 2.
# Posterior+SimHash has no seed-0 run, so all three seeds are here.
case "$CHUNK" in
    a)
        RUNS="EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:1 EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:2 EXP4_KEYCORRIDOR_POSTERIOR_SIMHASH:exp4_keycorridor_posterior_simhash:0"
        ;;
    b)
        RUNS="EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:1 EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:2 EXP4_KEYCORRIDOR_POSTERIOR_SIMHASH:exp4_keycorridor_posterior_simhash:1"
        ;;
    c)
        RUNS="EXP4_KEYCORRIDOR_POSTERIOR:exp4_keycorridor_posterior:1 EXP4_KEYCORRIDOR_POSTERIOR:exp4_keycorridor_posterior:2 EXP4_KEYCORRIDOR_POSTERIOR_SIMHASH:exp4_keycorridor_posterior_simhash:2"
        ;;
    *)
        echo "unknown chunk: $CHUNK (valid: a, b, c)" >&2
        exit 2
        ;;
esac

echo "##########################################################"
echo "## KeyCorridor multi-seed sweep — chunk $CHUNK"
echo "## Runs: $RUNS"
echo "##########################################################"

for triple in $RUNS; do
    section="${triple%%:*}"
    rest="${triple#*:}"
    run_base="${rest%%:*}"
    seed="${rest##*:}"
    run_name="${run_base}_seed${seed}"

    echo
    echo "=========================================================="
    echo "[chunk $CHUNK] $section -> runs/$run_name (seed=$seed)"
    echo "=========================================================="
    SEED="$seed" CONFIG_SECTION="$section" "$PY" -u train.py "$run_name"
done

echo
echo "[chunk $CHUNK] done."
