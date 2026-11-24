#!/usr/bin/env bash
# E1: does removing the far-side guard limits make the policy select push?
#   guard {1, 0} x train seed {0,1,2}; identical PPO settings otherwise
#   (the docs/rl_simulator_status_2026-10-10.md baseline: 16 envs, n_steps 16,
#   batch 128, 100k steps, gamma 1, ent_coef 0) on the hw2026 profile.
# Each job: train -> fixed eval seeds 0..9 (trace) -> push_diag with the model
# on seeds 0..2 (physics + clips).
# Usage: sim_experiment/run_e1.sh OUT_ROOT [PARALLEL_JOBS] [TIMESTEPS]
# Small pre-check: JOBS="0 0" EVAL_SEEDS="0" DIAG_SEEDS="0" run_e1.sh DIR 1 512
set -euo pipefail
OUT_ROOT=${1:?usage: run_e1.sh OUT_ROOT [PARALLEL_JOBS] [TIMESTEPS]}
PAR=${2:-2}   # one 16-env training uses ~17 GB RAM; robot17 has 60 GB
STEPS=${3:-100000}
PY=${PY:-$HOME/envs/gp8_sim/bin/python}
REPO_PARENT=$(cd "$(dirname "$0")/../.." && pwd)
[ -e "$OUT_ROOT" ] && { echo "exists: $OUT_ROOT (pick a new name)"; exit 1; }
mkdir -p "$OUT_ROOT"

# GP8_PUSH_PERCEPTION_LAG=0: the sim has no perception lag (see TODO.md E0b);
# the HW value 0.28 makes sim pushes fire early and miss half the time.
export GP8_PUSH_PERCEPTION_LAG=${GP8_PUSH_PERCEPTION_LAG:-0}
export MUJOCO_GL=egl GP8_SIM_PROFILE=hw2026 GP8_SIM_PHYSICAL_BELT=true \
       GP8_SIM_BELT_CONTACT_PARAMS=old GP8_RL_INCLUDE_SUCTION_P=false \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
DIAG_SEEDS=${DIAG_SEEDS:-"0 1 2"}
# step 상한: skip 한 번은 0.04 s라 skip을 자주 고르는 정책은 1000 step에서 240 s 전에 잘린다(E6). 8000이면 시간 예산이 먼저 끝난다.
MAX_STEPS=${MAX_STEPS:-8000}
JOBS=${JOBS:-"1 0,0 0,1 1,0 1,1 2,0 2"}   # "guard seed" pairs
export PY REPO_PARENT STEPS OUT_ROOT EVAL_SEEDS DIAG_SEEDS MAX_STEPS

job() {
  guard=$1; seed=$2
  run=$OUT_ROOT/guard${guard}_seed${seed}
  mkdir -p "$run"
  cd "$REPO_PARENT"
  export GP8_PUSH_GUARD=$guard
  echo "[$(date +%T)] start train guard=$guard seed=$seed"
  "$PY" -m gp8_control.tools.train_ppo_smoke --out-dir "$run" \
      --timesteps "$STEPS" --n-envs 16 --n-steps 16 --batch-size 128 \
      --seed "$seed" --max-steps "$MAX_STEPS" --max-episode-seconds 240 \
      --bbox-observation corners --truth-tracks --device cpu \
      --eval-episodes 0 --tensorboard --verbose 1 > "$run/train.log" 2>&1 \
    || { echo "FAILED train guard=$guard seed=$seed"; return 0; }
  echo "[$(date +%T)] start eval guard=$guard seed=$seed"
  mkdir -p "$run/eval_10seed"
  for es in $EVAL_SEEDS; do
    GP8_SIM_RECORD= "$PY" -m gp8_control.tools.train_ppo_smoke --eval-only \
        --model "$run/ppo_smoke.zip" --device cpu --seed "$es" --eval-steps "$MAX_STEPS" \
        --max-steps "$MAX_STEPS" --max-episode-seconds 240 --bbox-observation corners \
        --truth-tracks --trace-jsonl "$run/eval_10seed/seed_${es}.jsonl" \
        > "$run/eval_10seed/seed_${es}.log" 2>&1 || echo "FAILED eval guard=$guard seed=$seed es=$es"
  done
  for es in $DIAG_SEEDS; do
    "$PY" -m gp8_control.sim_experiment.push_diag --out "$run/diag" --seed "$es" \
        --policy model --model "$run/ppo_smoke.zip" --max-clips 10 \
        > "$run/diag_seed${es}.log" 2>&1 || echo "FAILED diag guard=$guard seed=$seed es=$es"
  done
  echo "[$(date +%T)] done guard=$guard seed=$seed"
}
export -f job

tr ',' '\n' <<< "$JOBS" \
  | xargs -P "$PAR" -L 1 bash -c 'job "$0" "$1"'
cd "$REPO_PARENT"
"$PY" -m gp8_control.sim_experiment.aggregate_e1 "$OUT_ROOT" | tee "$OUT_ROOT/report.md"
