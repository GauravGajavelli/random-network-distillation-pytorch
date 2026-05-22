#!/usr/bin/env bash
# CUDA version of scripts/run_noveld_ablation.sh.
# 10 new runs across 3 chunks, each pinned to one dedicated GPU.
#
# GPU assignment:
#   chunk a -> GPU 2
#   chunk b -> GPU 3
#   chunk c -> GPU 0
#
# Usage:
#   bash cuda/run_noveld_ablation.sh a > /tmp/nov_cuda_a.log 2>&1 &
#   bash cuda/run_noveld_ablation.sh b > /tmp/nov_cuda_b.log 2>&1 &
#   bash cuda/run_noveld_ablation.sh c > /tmp/nov_cuda_c.log 2>&1 &
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
    a) GPU=2; RUNS="EXP3_KEYCORRIDOR_NOVELD_POS:exp3_keycorridor_noveld_pos:0 EXP3_KEYCORRIDOR_NOVELD_POS:exp3_keycorridor_noveld_pos:1 EXP1_LAVA_NOVELD_POS:exp1_lava_noveld_pos:0" ;;
    b) GPU=3; RUNS="EXP3_KEYCORRIDOR_NOVELD_POS:exp3_keycorridor_noveld_pos:2 EXP3_KEYCORRIDOR_NOVELD:exp3_keycorridor_noveld:1 EXP1_LAVA_NOVELD_POS:exp1_lava_noveld_pos:1" ;;
    c) GPU=0; RUNS="EXP3_KEYCORRIDOR_NOVELD:exp3_keycorridor_noveld:2 EXP1_LAVA_NOVELD_POS:exp1_lava_noveld_pos:2 EXP1_LAVA_NOVELD:exp1_lava_noveld:1 EXP1_LAVA_NOVELD:exp1_lava_noveld:2" ;;
    *)
        echo "unknown chunk: $CHUNK (valid: a, b, c)" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export CONFIG_EXTRA="cuda/cuda_overrides.conf"

echo "##########################################################"
echo "## CUDA NovelD ablation — chunk $CHUNK  GPU=$GPU"
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
