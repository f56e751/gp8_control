#!/usr/bin/env bash
# E0: push diagnostics without training.
#   seeds 0..9  x  guard {1 = current, 0 = far-side guard limits off}
#               x  policy {push_first, throw_first, heuristic}
# Usage: sim_experiment/run_e0.sh OUT_ROOT [PARALLEL]
set -euo pipefail
OUT_ROOT=${1:?usage: run_e0.sh OUT_ROOT [PARALLEL]}
PAR=${2:-12}
PY=${PY:-$HOME/envs/gp8_sim/bin/python}
REPO_PARENT=$(cd "$(dirname "$0")/../.." && pwd)
[ -e "$OUT_ROOT" ] && { echo "exists: $OUT_ROOT (pick a new name)"; exit 1; }
mkdir -p "$OUT_ROOT"

export MUJOCO_GL=egl GP8_SIM_PROFILE=hw2026 GP8_SIM_PHYSICAL_BELT=true \
       GP8_SIM_BELT_CONTACT_PARAMS=old OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

jobs=()
for guard in 1 0; do
  for policy in push_first throw_first heuristic; do
    for seed in $(seq 0 9); do
      jobs+=("$guard $policy $seed")
    done
  done
done

printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -L 1 bash -c '
  guard=$0; policy=$1; seed=$2
  out='"$OUT_ROOT"'/guard${guard}
  mkdir -p "$out/logs"
  cd '"$REPO_PARENT"'
  GP8_PUSH_GUARD=$guard '"$PY"' -m gp8_control.sim_experiment.push_diag \
      --out "$out" --seed "$seed" --policy "$policy" --max-clips 10 \
      > "$out/logs/${policy}_seed${seed}.log" 2>&1 \
    && echo "done guard=$guard $policy seed=$seed" \
    || echo "FAILED guard=$guard $policy seed=$seed (see $out/logs)"
'
cd "$REPO_PARENT"
"$PY" -m gp8_control.sim_experiment.aggregate_diag "$OUT_ROOT" | tee "$OUT_ROOT/report.md"
