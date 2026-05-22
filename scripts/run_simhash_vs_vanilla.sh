#!/usr/bin/env bash
# Three-seed comparison: SimHash-additive RND vs vanilla RND on KeyCorridor.
#
# Model under test:
#   i_t = RND_normalized(s_{t+1}) + beta * 1/sqrt(n(hash(s_{t+1})))
#   where hash = sign(A^T flatten(s)), A fixed random N(0,1), n global cumulative.
#   Combined additively — no gating, no suppression of the RND term.
#   Config knobs: SimHashLambda=beta, SimHashDim=D. (UseSimHash=True)
#
# Baseline: vanilla RND (UseSimHash=False), identical otherwise.
#
# Seed-0 runs already exist as bare names. This script adds seeds 1 and 2.
# Two chunks run in parallel (one per terminal):
#
#   bash scripts/run_simhash_vs_vanilla.sh a   # baseline seed 1+2
#   bash scripts/run_simhash_vs_vanilla.sh b   # simhash  seed 1+2
#
# Evaluate when done:
#   python scripts/eval_simhash.py

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <a|b>" >&2
    exit 2
fi

case "$1" in
    a) RUNS="EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:1 EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:2" ;;
    b) RUNS="EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:1 EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:2" ;;
    *) echo "unknown chunk: $1 (valid: a, b)" >&2; exit 2 ;;
esac

echo "## SimHash vs vanilla — chunk $1"

for triple in $RUNS; do
    section="${triple%%:*}"; rest="${triple#*:}"
    run_base="${rest%%:*}"; seed="${rest##*:}"
    run_name="${run_base}_seed${seed}"
    echo; echo "==> $section -> runs/$run_name (seed=$seed)"
    SEED="$seed" CONFIG_SECTION="$section" "$PY" -u train.py "$run_name"
done

echo; echo "[chunk $1] done."
