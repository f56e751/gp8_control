#!/bin/bash
# Print the task file of the reported comparison: DPPO, IQL and BC on 100 / 200 /
# 500 / 1,000 episodes, training seeds 0-9, each run trained and then evaluated on
# simulator seeds 0-9. Two runs per line (= per GPU task); a line fails if either
# run fails. Run from the parent directory of gp8_control.
#
#     gp8_control/examples/dppo/scripts/make_table_tasks.sh > tasks.txt
#     sbatch --array=1-$(wc -l < tasks.txt) gp8_control/examples/dppo/scripts/slurm_task.sh tasks.txt
set -euo pipefail
[ -d gp8_control/examples/dppo ] || { echo "run from the parent directory of gp8_control" >&2; exit 1; }
PY=${GP8_VENV_PYTHON:-gp8_control/.venv/bin/python}
M=gp8_control.examples.dppo
RUNS=gp8_control/examples/dppo/runs
OUT=$RUNS/table
mkdir -p "$OUT" "$RUNS/slurm"

run() {  # run <tag> <trainer module> <trainer args...>
    local tag=$1 trainer=$2; shift 2
    echo "( $PY -m $M.$trainer $* --out $OUT/$tag.pt > $OUT/train_$tag.log 2>&1 &&" \
         "$PY -m $M.eval_offline_policy --ckpt $OUT/$tag.pt --tag $tag --seeds 0-9 --workers 5" \
         "--out $OUT/test.$tag.jsonl > $OUT/eval_$tag.log 2>&1 )"
}

# Training steps per dataset size: 20 x episodes, except 5,000 for 100 episodes.
for cfg in "100 5000" "200 4000" "500 10000" "1000 20000"; do
    read -r n steps <<< "$cfg"
    data="--data '$RUNS/data_$n/ep_*.npz' --steps $steps"
    for seed in 0 1 2 3 4 5 6 7 8 9; do
        run "dppo_n${n}_s$seed" train_dppo "$data --seed $seed"
        run "iql_n${n}_s$seed"  train_iql  "$data --seed $seed"
        run "bc_n${n}_s$seed"   train_dppo "$data --seed $seed --mode bc"
    done
done | awk 'NR%2==1{a=$0; next} {print a " & p=$!; " $0 "; s=$?; wait $p && exit $s"} END{if(NR%2==1) print a}'
