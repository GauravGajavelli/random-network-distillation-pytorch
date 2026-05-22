#!/usr/bin/env bash
# CUDA version: SimHash-additive RND vs vanilla RND, 3 seeds on KeyCorridor.
# Each chunk pinned to one dedicated GPU (no contention).
#
#   bash cuda/run_simhash_vs_vanilla.sh a > /tmp/sh_a.log 2>&1 &   # GPU 2
#   bash cuda/run_simhash_vs_vanilla.sh b > /tmp/sh_b.log 2>&1 &   # GPU 3

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-cuda/.venv/bin/python}"
export CONFIG_EXTRA="cuda/cuda_overrides.conf"

if [ $# -lt 1 ]; then
    echo "usage: $0 <a|b>" >&2; exit 2
fi

case "$1" in
    a) GPU=2; RUNS="EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:1 EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:2" ;;
    b) GPU=3; RUNS="EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:1 EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:2" ;;
    *) echo "unknown chunk: $1 (valid: a, b)" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
echo "## CUDA SimHash vs vanilla — chunk $1  GPU=$GPU"

for triple in $RUNS; do
    section="${triple%%:*}"; rest="${triple#*:}"
    run_base="${rest%%:*}"; seed="${rest##*:}"
    run_name="${run_base}_seed${seed}"
    echo; echo "==> $section -> runs/$run_name (seed=$seed, GPU=$GPU)"
    SEED="$seed" CONFIG_SECTION="$section" "$PY" -u train.py "$run_name"
done

echo; echo "[chunk $1 / GPU $GPU] done."
