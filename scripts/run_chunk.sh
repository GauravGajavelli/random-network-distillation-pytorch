#!/usr/bin/env bash
# Run one of three balanced compute chunks. Designed to be launched in
# parallel (one per shell) for ~2-2.5x wall-time speedup over serial,
# at the cost of CPU/GPU contention on M1 Pro.
#
# Each chunk has 3 runs and ~130-155 minutes of serial work.
#
# Usage:
#   bash scripts/run_chunk.sh a    # chunk A
#   bash scripts/run_chunk.sh b    # chunk B
#   bash scripts/run_chunk.sh c    # chunk C
#
# Recommended parallel launch (three shells, or background jobs):
#   bash scripts/run_chunk.sh a > /tmp/chunk_a.log 2>&1 &
#   bash scripts/run_chunk.sh b > /tmp/chunk_b.log 2>&1 &
#   bash scripts/run_chunk.sh c > /tmp/chunk_c.log 2>&1 &
#   wait
#
# Tail logs with: tail -f /tmp/chunk_*.log
#
# Override the seed by setting SEED=N before invoking (default: 0).
# E.g.:  SEED=1 bash scripts/run_chunk.sh a
#
# IMPORTANT: run directories are named with a "_seedN" suffix so that
# running with different SEED values produces non-conflicting outputs
# (e.g. runs/exp1_lava_option_b_seed0, runs/exp1_lava_option_b_seed1).
# This ensures fair multi-seed analysis: every Option B run at seed N
# is paired with a baseline run at the same seed N. See
# scripts/analyze_option_b.py for the matched-seed verification.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
SEED="${SEED:-0}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <a|b|c>" >&2
    exit 2
fi

CHUNK="$1"

# Each entry is SECTION:RUN_NAME. Estimated time in minutes per run is in the
# inline comment. Chunks are balanced so total per-chunk wall time is similar
# under serial execution (the parallel scheme then runs three chunks at once).
#
# Sizes:
#   small (Exp 1 & 2): ~35 min each
#   big   (Exp 3):     ~60 min each
#
# Chunk A: 1 big + 2 small = ~130 min (3 runs)
# Chunk B: 1 big + 2 small = ~130 min (3 runs)
# Chunk C: 2 big + 1 small = ~155 min (3 runs)
#
# Chunk C is ~25 min heavier — the asymmetry comes from 4 big runs not
# dividing evenly into 3 chunks. Wall time = max chunk ≈ 155 min, but with
# 3-way contention each chunk slows ~1.5-2x, so realistic wall time
# is 4-5 hours total.

# 15 total runs in the canonical sweep:
#   Exp 1 (LavaCrossing, ~35 min): vanilla, noveld, simhash             [3 small]
#   Exp 2 (DoorKey-5x5,  ~35 min): ppo, vanilla_rnd, noveld, simhash   [4 small]
#                                                                          (Option B and old DSC variants kept but not in default chunks)
#   Exp 3 (KeyCorridorS3R1, ~60 min): baseline, noveld, simhash,
#                                     simhash_tv (diagnostic)             [4 big]
#
# Chunk balance (each ~165 min serial):
#   Chunk A: 2 big + 2 small = 60+60+35+35 = 190
#   Chunk B: 1 big + 4 small = 60+35*4    = 200
#   Chunk C: 1 big + 3 small = 60+35*3    = 165
# Slight imbalance unavoidable.

case "$CHUNK" in
    a)
        RUNS="EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline EXP3_KEYCORRIDOR_NOVELD:exp3_keycorridor_noveld EXP1_LAVA_VANILLA:exp1_lava_vanilla EXP2_DOORKEY_PPO:exp2_doorkey_ppo"
        ;;
    b)
        RUNS="EXP3_KEYCORRIDOR_SIMHASH:exp3_keycorridor_simhash EXP1_LAVA_NOVELD:exp1_lava_noveld EXP1_LAVA_SIMHASH:exp1_lava_simhash EXP2_DOORKEY_VANILLA_RND:exp2_doorkey_vanilla_rnd EXP2_DOORKEY_NOVELD:exp2_doorkey_noveld"
        ;;
    c)
        RUNS="EXP3_KEYCORRIDOR_SIMHASH_TV:exp3_keycorridor_simhash_tv EXP2_DOORKEY_SIMHASH:exp2_doorkey_simhash EXP2_DOORKEY_OPTION_B:exp2_doorkey_option_b EXP1_LAVA_OPTION_B:exp1_lava_option_b"
        ;;
    d)
        # Experiment 4: posterior sampling (and stacking). All on KeyCorridor
        # because trajectory-level exploration commitment matters most there.
        # Each ~60 min serial; chunk D total ~180 min serial.
        RUNS="EXP4_KEYCORRIDOR_POSTERIOR:exp4_keycorridor_posterior EXP4_KEYCORRIDOR_POSTERIOR_NOVELD:exp4_keycorridor_posterior_noveld EXP4_KEYCORRIDOR_POSTERIOR_SIMHASH:exp4_keycorridor_posterior_simhash"
        ;;
    *)
        echo "unknown chunk: $CHUNK (valid: a, b, c, d)" >&2
        exit 2
        ;;
esac

echo "##########################################################"
echo "## Chunk $CHUNK, seed=$SEED"
echo "## Runs: $RUNS"
echo "##########################################################"

for pair in $RUNS; do
    section="${pair%%:*}"
    run_name="${pair##*:}_seed${SEED}"
    echo
    echo "=========================================================="
    echo "[chunk $CHUNK] running $section -> runs/$run_name (seed=$SEED)"
    echo "=========================================================="
    SEED="$SEED" CONFIG_SECTION="$section" "$PY" -u train.py "$run_name"
done

echo
echo "[chunk $CHUNK] all runs completed."
