# Settings of the grasp-and-throw trajectory generation — the setting of 2026-11-19 (robot6 ~/rge_jc: db_env.sh + setting_now.sh; the real robot and the
# simulation use the same values).  Bins at z = -0.11 m (the openings of the real bins).
# usage:  source env.sh        (from any folder; then run the scripts in planner/ with "$GT_PY")
#   GT_BELT=0.10|0.12|0.14|0.16|0.18|0.20   belt speed of the plan [m/s] (default 0.12)       GT_BIN_X=1.10|1.35|1.60   bin distance [m] (default 1.35)
#   GT_BIN_Z=-0.11                          bin height [m] (default -0.11; only sets built for that height can be used)
#   Each combination needs its own set of learned models in planner/logs (README.md, section 7); env.sh stops with a message when the set is missing.
#   GT_ENV_BASE=1   only the model and optimization values, no set (used by build/build_set.sh while it builds a new set)
GT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; export GT_HOME
gt_num() { awk -v v="$1" 'BEGIN { s = sprintf("%.3f", v); sub(/0$/, "", s); print s }'; }      # 0.1 -> 0.10, 1.6 -> 1.60, 1.475 -> 1.475
# Python with numpy, scipy and casadi (requirements.txt).  On robot6 / robot8 / robot10 the environment the code was developed in is used when it exists.
if [ -z "$GT_PY" ]; then if [ -x "$HOME/gp8_build/.venv/bin/python" ]; then GT_PY="$HOME/gp8_build/.venv/bin/python"; else GT_PY=python3; fi; fi; export GT_PY

# --- model and optimization (db_env.sh)
export GP8_CYC_NCTRL=52 CYC4_EXT=1 CYC4_PEXACT=1 CYC4_OMEGA=1
export GP8_THR_FILTER=box200_80      # controller filter model: dead time 40 ms, moving averages of 200 ms and 80 ms
export GP8_LIM_SCALE=0.9             # command limits = 0.9 x what the controller accepts
export GP8_BELT_SPEED=0.12 GP8_THR_W1=100 LPC_SLP_TR=0.02 LPC_BS=0
export OMP_NUM_THREADS=1             # one thread per solver process
# --- cycle (setting_now.sh)
export LPP_VIN=1 LPP_XMIN=0.25       # TCP kept at x >= 0.25 m
export RT_LIM=0.8,6,160 DB_LIM=0.8,6,160   # limits of the planned motion: 0.8 x rated speed, 6 x rated speed / s, 160 x rated speed / s^2
export RT_DW=89 RT_DWC=60            # hold on the object: 60 samples = 0.12 s
export RT_AHEAD=2.12                 # the planning of a cycle starts 2.0 s before the contact that starts it (2.0 + the hold)
GT_BIN_Z=$(gt_num "${GT_BIN_Z:--0.11}"); export GT_BIN_Z RGE_BIN_Z=$GT_BIN_Z      # height of the bin targets [m] (cyc2.py BINS); bins at x = GT_BIN_X, y = +0.33 (metal) / 0 (paper) / -0.33 (transparent)
# --- command fit (setting_now.sh, 11-18): the command whose filtered output follows the planned realized motion
export LPC_BS_DEG=3 LPP_BS_DEG=3     # command = cubic B-spline, knots 10 ms (both variables are needed: lp_planner.py copies LPP_BS_DEG into LPC_BS_DEG)
export RT_CMD_WIN=1                  # the 13 nodes of the release window (+-0.05 s) weighted equally (together as much as one key instant)
export RT_CMD_SOFT=1                 # release window and next contact as weights (1e6 x RT_SOFT_W default), solved at once
export RT_KEY_VW=1e-4                # weight of the velocity at a key point relative to its position
export RT_TRACK_W=10                 # weight of following the whole planned realized motion
export RT_CMD_SMOOTH=1e-4            # weight of (command acceleration / rated speed)^2.  Never 0 on the real robot: the command then swings and trips the controller
export RT_INV_SOLVER=ipm             # interior-point solve of the joint QPs (qpOASES only when a QP has equality rows)
export RT_CMD_MARGIN=0.35            # a cycle solve ends at least 0.35 s before the first sample of its command
export RT_ORDER=${RT_ORDER:-nn}      # next object: nn = learned ordering policy, fifo = most downstream candidate
unset AR_LIM RN_DB ORD_Z RGE_BELT_V RGE_BIN_X
[ "${GT_ENV_BASE:-0}" = 1 ] && return 0 2>/dev/null

# --- set of learned models for GT_BELT x GT_BIN_X (paths relative to planner/).  Names: tag = [b<VVV>_][x<XXX>_]zm011, e.g. zm011 (0.12, 1.35),
#     b016_zm011 (0.16, 1.35), x110_zm011 (0.12, 1.10); zm011 = bin height -0.11; a set whose models take the object tops as inputs has the suffix _z (only zm011_z so far).
GT_BELT=$(gt_num "${GT_BELT:-0.12}"); GT_BIN_X=$(gt_num "${GT_BIN_X:-1.35}"); export GT_BELT GT_BIN_X
GT_TAG=""; [ "$GT_BELT" = 0.12 ] || { GT_TAG="b$(echo $GT_BELT | tr -d .)_"; export RGE_BELT_V=$GT_BELT; }
[ "$GT_BIN_X" = 1.35 ] || { GT_TAG="${GT_TAG}x$(echo $GT_BIN_X | tr -d .)_"; export RGE_BIN_X=$GT_BIN_X; }
case "$GT_BIN_Z" in -*) GT_TAG="${GT_TAG}zm$(echo ${GT_BIN_Z#-} | tr -d .)" ;; *) GT_TAG="${GT_TAG}zp$(echo $GT_BIN_Z | tr -d .)" ;; esac; [ -f "$GT_HOME/planner/logs/tmodel_a6_${GT_TAG}_z.npz" ] && GT_TAG="${GT_TAG}_z"; export GT_TAG
export RN_NN=logs/nnws_$GT_TAG/w2_n2c.npz             # start of every cycle optimization (durations network + trajectory network)
export RT_TMODEL=logs/tmodel_a6_$GT_TAG.npz           # cycle-time model
export RT_ORDER_MODEL=logs/ord_a6_$GT_TAG/m1.npz      # ordering policy
for _f in $RN_NN $RT_TMODEL $RT_ORDER_MODEL; do
  if [ ! -f "$GT_HOME/planner/$_f" ]; then
    echo "[env.sh] no set of models for belt $GT_BELT m/s, bin x $GT_BIN_X m, bin z $GT_BIN_Z m ($GT_HOME/planner/$_f is missing). Sets here:" >&2
    ls "$GT_HOME/planner/logs" | sed -n 's/^tmodel_a6_\(.*\)\.npz$/  \1/p' >&2; echo "  build one with build/build_set.sh (README.md, section 7)" >&2
    export RN_NN=logs/NO_SUCH_SET; return 1 2>/dev/null || exit 1
  fi
done; unset _f
