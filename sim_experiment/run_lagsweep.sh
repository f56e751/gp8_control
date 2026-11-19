#!/usr/bin/env bash
# Sim perception-lag sweep with the HW timing compensations switched ON
# (push PUSH_PERCEPTION_LAG 0.28 s, throw GP8_TRACK_LEAD_T 0.12 s), to find the
# sim lag at which each skill behaves as on HW. No training.
#   lag {0.12, 0.20, 0.28} x policy {push_first, throw_first, heuristic} x seeds 0..9
# Usage: sim_experiment/run_lagsweep.sh OUT_ROOT [PARALLEL] [LAGS]
set -euo pipefail
OUT_ROOT=${1:?usage: run_lagsweep.sh OUT_ROOT [PARALLEL] [LAGS]}
PAR=${2:-3}
LAGS=${3:-"0.12 0.20 0.28"}
PY=${PY:-$HOME/envs/gp8_sim/bin/python}
REPO_PARENT=$(cd "$(dirname "$0")/../.." && pwd)
[ -e "$OUT_ROOT" ] && { echo "exists: $OUT_ROOT"; exit 1; }
mkdir -p "$OUT_ROOT"
export MUJOCO_GL=egl GP8_SIM_PROFILE=hw2026 GP8_SIM_PHYSICAL_BELT=true \
       GP8_SIM_BELT_CONTACT_PARAMS=old OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       GP8_PUSH_GUARD=1 GP8_PUSH_PERCEPTION_LAG=0.28 GP8_TRACK_LEAD_T=0.12
export PY REPO_PARENT OUT_ROOT
for lag in $LAGS; do for policy in push_first throw_first heuristic; do for seed in $(seq 0 9); do
  echo "$lag $policy $seed"; done; done; done \
| xargs -P "$PAR" -L 1 bash -c '
  lag=$0; policy=$1; seed=$2; out=$OUT_ROOT/simlag${lag}; mkdir -p "$out/logs"; cd "$REPO_PARENT"
  GP8_SIM_PERCEPTION_LAG=$lag "$PY" -m gp8_control.sim_experiment.push_diag --out "$out" \
     --seed "$seed" --policy "$policy" --max-clips 4 > "$out/logs/${policy}_s${seed}.log" 2>&1 \
  || echo "FAILED lag=$lag $policy $seed"'
cd "$REPO_PARENT"
"$PY" -m gp8_control.sim_experiment.aggregate_diag "$OUT_ROOT" | tee "$OUT_ROOT/report.md"
