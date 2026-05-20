#!/usr/bin/env bash
# Run all 9 experiment configurations sequentially, writing TensorBoard logs
# under runs/<exp_name>/ and saving model checkpoints under models/.
#
# After completion, run:
#     python scripts/eval_all.py --runs-dir runs/
# to evaluate against the win conditions.
#
# Each run honors TotalSteps from its config section. On M1 Pro, expect
# ~30-90 min per run (9 runs total).
#
# Usage:
#     bash scripts/run_all_experiments.sh                # all 3 experiments
#     bash scripts/run_all_experiments.sh exp1           # just experiment 1
#     bash scripts/run_all_experiments.sh exp2 exp3      # 2 and 3
#
# Note: this script is compatible with bash 3.2 (macOS default). It uses a
# function instead of associative arrays, which require bash 4+.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"

get_runs_for() {
    case "$1" in
        exp1)
            echo "EXP1_LAVA_VANILLA:exp1_lava_vanilla EXP1_LAVA_OPTION_B:exp1_lava_option_b"
            ;;
        exp2)
            echo "EXP2_DOORKEY_PPO:exp2_doorkey_ppo EXP2_DOORKEY_VANILLA_RND:exp2_doorkey_vanilla_rnd EXP2_DOORKEY_OPTION_B:exp2_doorkey_option_b"
            ;;
        exp3)
            echo "EXP3_KEYCORRIDOR_BASELINE:exp3_keycorridor_baseline EXP3_KEYCORRIDOR_DSC:exp3_keycorridor_dsc EXP3_KEYCORRIDOR_DSC_TYPED:exp3_keycorridor_dsc_typed EXP3_KEYCORRIDOR_DSC_TV_OFF:exp3_keycorridor_dsc_tv_off"
            ;;
        *)
            echo ""
            return 1
            ;;
    esac
}

if [ $# -eq 0 ]; then
    SELECTED="exp1 exp2 exp3"
else
    SELECTED="$*"
fi

for exp in $SELECTED; do
    pairs=$(get_runs_for "$exp") || {
        echo "unknown experiment: $exp (valid: exp1 exp2 exp3)" >&2
        exit 2
    }
    for pair in $pairs; do
        section="${pair%%:*}"
        run_name="${pair##*:}"
        echo "=========================================================="
        echo "Running $section -> runs/$run_name"
        echo "=========================================================="
        CONFIG_SECTION="$section" "$PY" -u train.py "$run_name"
    done
done

echo
echo "All requested experiments completed."
echo "Evaluate with: $PY scripts/eval_all.py --runs-dir runs/"
