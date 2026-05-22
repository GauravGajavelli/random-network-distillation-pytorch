#!/usr/bin/env bash
# Convenience wrapper: launch the canonical chunks in parallel as background
# jobs. Defaults to chunks a, b, c (13 runs, ~5-6 hours wall time). Pass a
# space-separated list of chunk letters to include chunk d (Experiment 4
# posterior sampling, 3 extra runs, ~3 hours added):
#
#   bash scripts/run_all_parallel.sh           # a b c
#   bash scripts/run_all_parallel.sh a b c d   # also Experiment 4
#   bash scripts/run_all_parallel.sh d         # only Experiment 4
#
# Logs go to /tmp/chunk_{letter}.log. Waits for all to finish, then prints
# a one-line summary.
#
# Usage:
#   bash scripts/run_all_parallel.sh         # default SEED=0
#   SEED=1 bash scripts/run_all_parallel.sh  # different seed
#
# Monitor progress in another shell:
#   tail -f /tmp/chunk_a.log
#   tail -f /tmp/chunk_b.log
#   tail -f /tmp/chunk_c.log
#
# Estimated wall time on M1 Pro: 4-5 hours (vs ~7 hours serial). The savings
# come at the cost of CPU/GPU contention — each individual run slows ~1.5-2x,
# but three running concurrently more than compensates.
#
# If you'd rather use three separate terminal windows (more visibility, less
# log-tailing), invoke `bash scripts/run_chunk.sh a` etc. directly in each.

set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-0}"

if [ $# -eq 0 ]; then
    CHUNKS="a b c"
else
    CHUNKS="$*"
fi

echo "Launching parallel chunks [$CHUNKS] with SEED=$SEED..."
echo

PIDS=""
LETTERS=""
for ch in $CHUNKS; do
    log="/tmp/chunk_${ch}.log"
    echo "  chunk $ch -> $log"
    SEED="$SEED" bash scripts/run_chunk.sh "$ch" > "$log" 2>&1 &
    PIDS="$PIDS $!"
    LETTERS="$LETTERS $ch"
done
echo "Started at: $(date)"
echo

ALL_OK=1
i=1
for pid in $PIDS; do
    letter=$(echo "$LETTERS" | awk "{print \$$i}")
    wait "$pid"; status=$?
    echo "chunk_$letter finished with status=$status at $(date)"
    if [ "$status" -ne 0 ]; then ALL_OK=0; fi
    i=$((i + 1))
done

echo
if [ "$ALL_OK" -eq 1 ]; then
    echo "ALL CHUNKS SUCCEEDED"
    echo "Next: .venv/bin/python scripts/eval_all.py --runs-dir runs/"
    echo "Also: .venv/bin/python scripts/eval_posterior.py (if chunk d was run)"
    exit 0
fi
echo "SOME CHUNKS FAILED — check the logs"
exit 1
