#!/bin/bash
# One command for the cycle-throw demo (robot PC):
#   1. vision on robot6 (docker iitp_local): started if it is not running, then
#      waits until the detection stream answers with schema_version 4
#   2. robot6 CPU check: the cycle planner runs there too (busy cores, 2 s sample)
#   3. camera_debug (= gp8_camera) in the background, log to a file
#   4. gp8_app in cycle mode (driver + app; the app starts the planner on robot6)
# Ctrl+C stops in order: the app winds down (arm to its wait point, controller
# back to JGPC), then the driver, then camera_debug; a second Ctrl+C stops all
# at once. Vision keeps running (it takes a while to start);
# GP8_DEMO_STOP_VISION=1 stops it as well.
# Any value below can be set on the command line, e.g.
#   GP8_FORCE_SKILL=throw gp8_cycle        (throw everything: first runs)
#   GT_BELT=0.14 gp8_cycle                 (another model set)
#   GP8_DEMO_CHECK_ONLY=1 gp8_cycle        (vision + robot6 check + robot-free planner check, no robot)
#   GP8_DEMO_MODE=pieces gp8_cycle         (piece mode: push and throw on one command path)
#   GP8_RUN_RECORD=0 gp8_cycle             (no real-to-sim recording; default on, <log>/record)
# Details: cycle_throw/RUN_HW.md
D="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$D/.." && pwd)"
source "$D/env.sh" >/dev/null      # ROS setup files are not `set -u` clean
set -u
say() { echo "[gp8_cycle] $*"; }

# .env (gitignored) without overriding what the shell already set
if [ -f "$PKG/.env" ]; then
  while IFS='=' read -r k v; do
    case "$k" in ''|\#*) continue ;; esac
    [ -z "${!k+x}" ] && export "$k=$v"
  done < "$PKG/.env"
fi
H="${GP8_CYCLE_PLANNER_HOST:-robot6-planner}"
URL="${GP8_PERCEPTION_URL:-http://147.46.175.15:8080/detections/stream}"
VISION_DIR="${GP8_VISION_DIR:-/PublicSSD/iitp/demo/iitp_perception}"
LOG="$HOME/gp8_logs/cycle_demo/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 "$H")

"${SSH[@]}" true || { say "cannot reach the planner host $H by key (ssh $H)"; exit 1; }

# 1. vision ------------------------------------------------------------------
stream_ok() { curl -s -m 3 "$URL" 2>/dev/null | head -c 4000 | grep -q '"schema_version": *4'; }
if "${SSH[@]}" "docker ps --filter name=^iitp_local\$ --format {{.Names}}" | grep -q iitp_local; then
  say "vision: iitp_local already running on $H"
else
  say "vision: starting iitp_local on $H ($VISION_DIR/docker_local.sh, log /tmp/iitp_local_gp8_cycle.log there)"
  # `cd X; job &` (not `cd X && job &`): only the job itself goes to the
  # background with its fds redirected, so ssh returns at once.
  "${SSH[@]}" -n "cd $VISION_DIR || exit 1; setsid nohup ./docker_local.sh > /tmp/iitp_local_gp8_cycle.log 2>&1 < /dev/null &"
fi
for _ in $(seq 1 "${GP8_DEMO_VISION_WAIT:-90}"); do stream_ok && break; sleep 1; done
stream_ok || { say "vision stream $URL does not answer with schema_version 4 (robot6: /tmp/iitp_local_gp8_cycle.log)"; exit 1; }
say "vision stream OK ($URL)"

# 2. robot6 CPU (vision + whatever else runs there) ------------------------------
BUSY=$("${SSH[@]}" 'python3 -c "
import time
def s():
    v = [int(x) for x in open(\"/proc/stat\").readline().split()[1:]]
    return sum(v), v[3] + v[4]
t0, i0 = s(); time.sleep(2); t1, i1 = s()
import os; print(round((1 - (i1 - i0) / (t1 - t0)) * os.cpu_count(), 1), os.cpu_count())
"')
say "robot6 busy cores: ${BUSY% *} of ${BUSY#* } (load $("${SSH[@]}" "cut -d' ' -f1-3 /proc/loadavg"))"
if awk -v b="${BUSY% *}" -v m="${GP8_DEMO_MAX_BUSY:-4}" 'BEGIN { exit !(b > m) }'; then
  say "robot6 is busy (> ${GP8_DEMO_MAX_BUSY:-4} cores): the planner would miss deadlines. Stop the other job (DB build / Isaac) or set GP8_DEMO_MAX_BUSY"
  exit 1
fi

# Pre-flight only: the robot-free planner check with vision running, no robot.
if [ "${GP8_DEMO_CHECK_ONLY:-0}" = 1 ]; then
  say "check only: robot-free planner check against $H (${GP8_DEMO_CHECK_S:-30} s, fake controller buffer, ${GP8_DEMO_MODE:-cycle} mode)"
  OUT=$(PYTHONPATH="$PKG/..:${PYTHONPATH:-}" "$PKG/.venv/bin/python" -m gp8_control.tools.cycle_dry_check \
        --mode "${GP8_DEMO_MODE:-cycle}" --host "$H" --seconds "${GP8_DEMO_CHECK_S:-30}" 2>&1 \
        | tee "$LOG/dry_check.log" | grep SUMMARY)
  echo "$OUT"
  case "$OUT" in *"SUMMARY PASS"*) exit 0 ;; *) exit 1 ;; esac
fi

# 3. camera_debug ---------------------------------------------------------------
# Own process group too: a backgrounded `ros2 run` ignores SIGINT, so it is
# stopped by SIGTERM to its whole group (ros2 run + the node it starts).
setsid "$D/camera.sh" > "$LOG/camera_debug.log" 2>&1 &
CAM=$!
cleanup() {
  kill -TERM -- -"$CAM" 2>/dev/null
  for _ in 1 2 3 4 5; do pgrep -g "$CAM" >/dev/null || break; sleep 1; done
  pgrep -g "$CAM" >/dev/null && kill -KILL -- -"$CAM" 2>/dev/null
  if [ "${GP8_DEMO_STOP_VISION:-0}" = 1 ]; then
    "${SSH[@]}" "docker stop iitp_local" >/dev/null 2>&1 && say "vision stopped"
  fi
  if [ -n "${BAG:-}" ]; then          # encoder bag: stop it, then into the recording
    kill -INT -- -"$BAG" 2>/dev/null
    for _ in $(seq 1 10); do pgrep -g "$BAG" >/dev/null || break; sleep 1; done
    python3 -m gp8_control.tools.bag_fb "$GP8_RUN_RECORD_DIR" 2>&1 | tail -2
  fi
  say "logs: $LOG"
  [ -n "${GP8_RUN_RECORD_DIR:-}" ] && say "recording: $GP8_RUN_RECORD_DIR (python -m gp8_control.tools.real2sim summary <dir>)"
}
trap cleanup EXIT
sleep 3
pgrep -g "$CAM" >/dev/null || { say "camera_debug exited — see $LOG/camera_debug.log"; exit 1; }
say "camera_debug running (log $LOG/camera_debug.log)"

# 4. driver + app in cycle mode --------------------------------------------------
# cycle = push slots between planner segments (HW-run); pieces = piece mode
case "${GP8_DEMO_MODE:-cycle}" in
  cycle|pieces) export GP8_MODE="${GP8_DEMO_MODE:-cycle}" ;;
  *) say "GP8_DEMO_MODE=${GP8_DEMO_MODE}: expected cycle or pieces"; exit 1 ;;
esac
export GP8_STREAM_HZ="${GP8_STREAM_HZ:-500}"
# Piece-mode control settings: config/piece_mode.env, the same file the twin /
# RL env read (cycle_throw/piece_settings.py); shell or .env values win.
if [ "$GP8_MODE" = pieces ]; then
  while IFS='=' read -r k v; do
    case "$k" in ''|\#*) continue ;; esac
    [ -z "${!k+x}" ] && export "$k=$v"
  done < "$PKG/config/piece_mode.env"
fi
export GP8_SAMPLE_BUFFER="${GP8_SAMPLE_BUFFER:-true}"
# Real-to-sim recording (utils/run_record.py, read with tools/real2sim.py):
# joints, commands, link messages, detections, tracks. GP8_RUN_RECORD=0 = off.
[ "${GP8_RUN_RECORD:-1}" = 1 ] && export GP8_RUN_RECORD_DIR="${GP8_RUN_RECORD_DIR:-$LOG/record}"
export GP8_CYCLE_PLANNER_HOST="$H"
export GP8_CYCLE_TOP_Z="${GP8_CYCLE_TOP_Z:-0.042}"
say "app: $GP8_MODE mode, skill routing ${GP8_FORCE_SKILL:-by class (metal push, transparent throw)}," \
    "belt ${GT_BELT:-0.12} m/s, bins x ${GT_BIN_X:-1.35} m, release lead ${GP8_CYCLE_RELEASE_LEAD:-0.2 (VALVE_S)} s, top z $GP8_CYCLE_TOP_Z m, push reach ${GP8_PUSH_MAX_REACH:-0.65} accel ${GP8_PUSH_ACCEL:-12}"

# Encoder feedback (pos_fdb on /dynamic_joint_states) for the recording: the C++
# bag recorder in its own process group (inside the Python app a 500 Hz
# subscription cost ~20 % of a core). tools/bag_fb.py merges it at the end.
BAG=""
if [ -n "${GP8_RUN_RECORD_DIR:-}" ]; then
  mkdir -p "$GP8_RUN_RECORD_DIR"
  setsid ros2 bag record -o "$GP8_RUN_RECORD_DIR/fb_bag" /dynamic_joint_states > "$LOG/fb_bag.log" 2>&1 < /dev/null &
  BAG=$!
fi

# The driver + app run in their OWN process group, so Ctrl+C (or kill -INT of
# this script) reaches only this script, which then stops things in order:
#   1. SIGUSR1 to the app: no new object, the planner throws what it holds and
#      ends at its wait point, the buffer stops and JGPC gets the interfaces back
#   2. only then SIGINT to the driver/launch group.
# A second Ctrl+C stops everything at once.
setsid "$D/app.sh" "$@" &
APP_PG=$!
STOPPING=0
app_pid() { pgrep -g "$APP_PG" -f "${GP8_DEMO_APP_PATTERN:-m gp8_control.app}" | head -1; }
group_alive() { pgrep -g "$APP_PG" >/dev/null; }
stop_in_order() {
  if [ "$STOPPING" = 1 ]; then
    say "second Ctrl+C: stopping everything now"
    kill -INT -- -"$APP_PG" 2>/dev/null
    return
  fi
  STOPPING=1
  local p; p=$(app_pid)
  if [ -n "$p" ]; then
    say "stopping: the app winds down first (arm to its wait point, controller back to JGPC) ..."
    kill -USR1 "$p" 2>/dev/null
    for _ in $(seq 1 "${GP8_DEMO_STOP_WAIT:-25}"); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -0 "$p" 2>/dev/null && say "the app did not finish in ${GP8_DEMO_STOP_WAIT:-25} s; stopping the driver anyway"
  fi
  say "stopping the driver"
  kill -INT -- -"$APP_PG" 2>/dev/null
  for _ in $(seq 1 15); do group_alive || break; sleep 1; done
  group_alive && { say "driver group still up after 15 s: SIGTERM"; kill -TERM -- -"$APP_PG" 2>/dev/null; }
}
trap stop_in_order INT TERM
# `wait` returns early when a trapped signal arrives: keep waiting until the
# whole driver/app group is gone.
while group_alive; do wait "$APP_PG" 2>/dev/null; sleep 0.2; done
