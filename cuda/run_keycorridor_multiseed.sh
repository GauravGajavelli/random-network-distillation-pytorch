#!/usr/bin/env bash
# CUDA version of scripts/run_keycorridor_multiseed.sh.
# Same 9 new runs; each chunk pinned to one dedicated GPU.
#
# GPU assignment (gebru, GPUs 0-3 available):
#   chunk a -> GPU 2
#   chunk b -> GPU 3
#   chunk c -> GPU 0
#
# Usage:
#   bash cuda/run_keycorridor_multiseed.sh a > /tmp/kc_cuda_a.log 2>&1 &
#   bash cuda/run_keycorridor_multiseed.sh b > /tmp/kc_cuda_b.log 2>&1 &
#   bash cuda/run_keycorridor_multiseed.sh c > /tmp/kc_cuda_c.log 2>&1 &
#   wait

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-cuda/.venv/bin/python}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <a|b|c>" >&2
    exit 2
fi

CHUNK="$1"

case "$CHUNK" in
    a) GPU=2; RUNS="EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:1 EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline:2 EXP4_KEYCORRIDOR_POSTERIOR_SIMHASH:exp4_keycorridor_posterior_simhash:0" ;;
    b) GPU=3; RUNS="EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:1 EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash:2 EXP4_KEYCORRIDOR_POSTERIOR_SIMHASH:exp4_keycorridor_posterior_simhash:1" ;;
    c) GPU=0; RUNS="EXP4_KEYCORRIDOR_POSTERIOR:exp4_keycorridor_posterior:1 EXP4_KEYCORRIDOR_POSTERIOR:exp4_keycorridor_posterior:2 EXP4_KEYCORRIDOR_POSTERIOR_SIMHASH:exp4_keycorridor_posterior_simhash:2" ;;
    *)
        echo "unknown chunk: $CHUNK (valid: a, b, c)" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export CONFIG_EXTRA="cuda/cuda_overrides.conf"

echo "##########################################################"
echo "## CUDA KeyCorridor multi-seed — chunk $CHUNK  GPU=$GPU"
echo "##########################################################"

for triple in $RUNS; do
    section="${triple%%:*}"
    rest="${triple#*:}"
    run_base="${rest%%:*}"
    seed="${rest##*:}"
    run_name="${run_base}_seed${seed}"
    echo
    echo "=========================================================="
    echo "[chunk $CHUNK / GPU $GPU] $section -> runs/$run_name (seed=$seed)"
    echo "=========================================================="
    SEED="$seed" CONFIG_SECTION="$section" "$PY" -u train.py "$run_name"
done

echo
echo "[chunk $CHUNK / GPU $GPU] done."
