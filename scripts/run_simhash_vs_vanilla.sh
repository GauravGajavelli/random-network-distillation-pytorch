#!/usr/bin/env bash
# Three-seed comparison: SimHash+RND vs vanilla RND vs SimHash-only on KeyCorridor.
#
# Model under test:
#   SimHash+RND:  i_t = RND_normalized(s_{t+1}) + beta * 1/sqrt(n(hash(s_{t+1})))
#   SimHash-only: i_t = beta * 1/sqrt(n(hash(s_{t+1})))  (UseRNDBonus=False)
#   Vanilla RND:  i_t = RND_normalized(s_{t+1})
#
#   hash = sign(A^T flatten(s)), A fixed random N(0,1), n global cumulative.
#   Config knobs: SimHashLambda=beta, SimHashDim=D.
#
# Seed-0 runs already exist as bare names. This script adds seeds 1 and 2.
# Three chunks run in parallel (one per terminal):
#
#   bash scripts/run_simhash_vs_vanilla.sh a   # baseline seed 1+2
#   bash scripts/run_simhash_vs_vanilla.sh b   # simhash+rnd seed 1+2
#   bash scripts/run_simhash_vs_vanilla.sh c   # simhash-only seed 0+1+2
#
# Evaluate when done:
#   python scripts/eval_simhash.py

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <a|b|c>" >&2
    exit 2
fi

case "$1" in
    a) RUNS="EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:1 EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:2" ;;
    b) RUNS="EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:1 EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:2" ;;
    c) RUNS="EXP3_KEYCORRIDOR_SIMHASH_ONLY:exp3_keycorridor_simhash_only:0 EXP3_KEYCORRIDOR_SIMHASH_ONLY:exp3_keycorridor_simhash_only:1 EXP3_KEYCORRIDOR_SIMHASH_ONLY:exp3_keycorridor_simhash_only:2" ;;
    *) echo "unknown chunk: $1 (valid: a, b, c)" >&2; exit 2 ;;
esac

echo "## SimHash vs vanilla (3-way) — chunk $1"

for triple in $RUNS; do
    section="${triple%%:*}"; rest="${triple#*:}"
    run_base="${rest%%:*}"; seed="${rest##*:}"
    run_name="${run_base}_seed${seed}"
    echo; echo "==> $section -> runs/$run_name (seed=$seed)"
    SEED="$seed" CONFIG_SECTION="$section" "$PY" -u train.py "$run_name"
done

echo; echo "[chunk $1] done."
