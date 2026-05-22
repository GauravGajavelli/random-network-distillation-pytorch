#!/usr/bin/env bash
# Ablation: NovelD with position keys vs NovelD with cluster keys.
#
# Tests whether K-means clustering over RND target features improves on
# raw position-key counting across two environment scales:
#
#   LavaCrossing (9x9, small):  clusters may HURT — K=8 over ~50 positions
#                                = coarser counter than raw position keys;
#                                multiplier decays faster, suppressing
#                                exploration in a sparse-reward env.
#
#   KeyCorridor (larger):       clusters may HELP — position keys cover
#                                a larger space so repeat visits are rare
#                                within an episode; clustering groups
#                                semantically similar corridor states.
#
# New runs produced (10 total):
#   exp1_lava_noveld_pos_seed{0,1,2}         NovelD, position keys, LavaCrossing
#   exp1_lava_noveld_seed{1,2}               NovelD, cluster keys,  LavaCrossing
#   exp3_keycorridor_noveld_pos_seed{0,1,2}  NovelD, position keys, KeyCorridor
#   exp3_keycorridor_noveld_seed{1,2}        NovelD, cluster keys,  KeyCorridor
#
# Seed-0 cluster variants already exist as bare names (exp1_lava_noveld,
# exp3_keycorridor_noveld) and are used directly by eval_noveld_ablation.py.
#
# Run in three terminals (~2.5-3 hr wall time with contention on M1 Pro):
#   bash scripts/run_noveld_ablation.sh a
#   bash scripts/run_noveld_ablation.sh b
#   bash scripts/run_noveld_ablation.sh c
#
# Or as background jobs:
#   bash scripts/run_noveld_ablation.sh a > /tmp/nov_a.log 2>&1 &
#   bash scripts/run_noveld_ablation.sh b > /tmp/nov_b.log 2>&1 &
#   bash scripts/run_noveld_ablation.sh c > /tmp/nov_c.log 2>&1 &
#   wait

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <a|b|c>" >&2
    exit 2
fi

CHUNK="$1"

# Format: "SECTION:run_base_name:SEED"
# Chunk timing (serial):
#   KeyCorridor runs: ~60 min each
#   LavaCrossing runs: ~35 min each
#
# Chunk a: kc_pos_0(60) + kc_pos_1(60) + lava_pos_0(35)       = 155 min
# Chunk b: kc_pos_2(60) + kc_clusters_1(60) + lava_pos_1(35)  = 155 min
# Chunk c: kc_clusters_2(60) + lava_pos_2(35) + lava_cl_1(35) + lava_cl_2(35) = 165 min

case "$CHUNK" in
    a)
        RUNS="EXP3_KEYCORRIDOR_NOVELD_POS:exp3_keycorridor_noveld_pos:0 EXP3_KEYCORRIDOR_NOVELD_POS:exp3_keycorridor_noveld_pos:1 EXP1_LAVA_NOVELD_POS:exp1_lava_noveld_pos:0"
        ;;
    b)
        RUNS="EXP3_KEYCORRIDOR_NOVELD_POS:exp3_keycorridor_noveld_pos:2 EXP3_KEYCORRIDOR_NOVELD:exp3_keycorridor_noveld:1 EXP1_LAVA_NOVELD_POS:exp1_lava_noveld_pos:1"
        ;;
    c)
        RUNS="EXP3_KEYCORRIDOR_NOVELD:exp3_keycorridor_noveld:2 EXP1_LAVA_NOVELD_POS:exp1_lava_noveld_pos:2 EXP1_LAVA_NOVELD:exp1_lava_noveld:1 EXP1_LAVA_NOVELD:exp1_lava_noveld:2"
        ;;
    *)
        echo "unknown chunk: $CHUNK (valid: a, b, c)" >&2
        exit 2
        ;;
esac

echo "##########################################################"
echo "## NovelD cluster ablation — chunk $CHUNK"
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
