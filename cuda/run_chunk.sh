#!/usr/bin/env bash
# CUDA version of run_chunk.sh. Each chunk is pinned to one GPU so four
# experiments run truly in parallel (no GPU contention), one per GPU.
#
# GPU assignment on gebru (avoid 4-7, occupied by VLLM):
#   chunk a -> GPU 2  (least loaded, 7% util)
#   chunk b -> GPU 3  (9% util)
#   chunk c -> GPU 0  (66% util, ~23GB free)
#   chunk d -> GPU 1  (83% util, ~23GB free)
#
# Run all four simultaneously:
#   bash cuda/run_chunk.sh a > /tmp/cuda_a.log 2>&1 &
#   bash cuda/run_chunk.sh b > /tmp/cuda_b.log 2>&1 &
#   bash cuda/run_chunk.sh c > /tmp/cuda_c.log 2>&1 &
#   bash cuda/run_chunk.sh d > /tmp/cuda_d.log 2>&1 &
#   wait
#
# Or interactively in four terminals:
#   bash cuda/run_chunk.sh a
#   bash cuda/run_chunk.sh b
#   bash cuda/run_chunk.sh c
#   bash cuda/run_chunk.sh d
#
# SEED override works the same as the root run_chunk.sh:
#   SEED=1 bash cuda/run_chunk.sh a

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-cuda/.venv/bin/python}"
SEED="${SEED:-0}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <a|b|c|d>" >&2
    exit 2
fi

CHUNK="$1"

case "$CHUNK" in
    a) GPU=2; RUNS="EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline EXP3_KEYCORRIDOR_NOVELD:exp3_keycorridor_noveld EXP1_LAVA_VANILLA:exp1_lava_vanilla EXP2_DOORKEY_PPO:exp2_doorkey_ppo" ;;
    b) GPU=3; RUNS="EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash EXP1_LAVA_NOVELD:exp1_lava_noveld EXP1_LAVA_SIMHASH:exp1_lava_simhash EXP2_DOORKEY_VANILLA_RND:exp2_doorkey_vanilla_rnd EXP2_DOORKEY_NOVELD:exp2_doorkey_noveld" ;;
    c) GPU=0; RUNS="EXP3_KEYCORRIDOR_SIMHASH_TV:exp3_keycorridor_simhash_tv EXP2_DOORKEY_SIMHASH:exp2_doorkey_simhash EXP2_DOORKEY_OPTION_B:exp2_doorkey_option_b EXP1_LAVA_OPTION_B:exp1_lava_option_b" ;;
    d) GPU=1; RUNS="EXP4_KEYCORRIDOR_POSTERIOR:exp4_keycorridor_posterior EXP4_KEYCORRIDOR_POSTERIOR_NOVELD:exp4_keycorridor_posterior_noveld EXP4_KEYCORRIDOR_POSTERIOR_SIMHASH:exp4_keycorridor_posterior_simhash" ;;
    *)
        echo "unknown chunk: $CHUNK (valid: a, b, c, d)" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export CONFIG_EXTRA="cuda/cuda_overrides.conf"

echo "##########################################################"
echo "## CUDA chunk $CHUNK  GPU=$GPU  seed=$SEED"
echo "## Runs: $RUNS"
echo "##########################################################"

for pair in $RUNS; do
    section="${pair%%:*}"
    run_name="${pair##*:}_seed${SEED}"
    echo
    echo "=========================================================="
    echo "[chunk $CHUNK / GPU $GPU] $section -> runs/$run_name"
    echo "=========================================================="
    SEED="$SEED" CONFIG_SECTION="$section" "$PY" -u train.py "$run_name"
done

echo
echo "[chunk $CHUNK / GPU $GPU] all runs completed."
