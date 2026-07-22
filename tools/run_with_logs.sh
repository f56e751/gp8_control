#!/usr/bin/env bash
# One-command GP8 run with EVERYTHING logged into a per-run directory.
#
#   ~/ros2_ws/src/gp8_control/tools/run_with_logs.sh [launch args...]
#   e.g.  tools/run_with_logs.sh release_lead:=0.06
#
# Creates ~/gp8_runs/<timestamp>/ containing:
#   meta.txt                  branch / HEAD / dirty files / args  (which code ran?!)
#   console.log               full launch console (veto lines, fire leads, clamps)
#   motion/                   GP8_MOTION_LOG_DIR per-dispatch stream CSVs
#   gp8_pick_log.push.csv     this run's push cycle rows (moved here on exit)
#   gp8_pick_log.throw.csv    shared throw CSV copy, if throw wrote during the run
#   analysis.txt              tools/analyze_pick_log.py output
#   summary.txt               veto/clamp/chain grep counts
#
# Ctrl+C stops the launch; collection + analysis run automatically afterwards.

set -u
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- ROS env (no-op if the shell already sourced it) ------------------------
if ! command -v ros2 >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    [ -f "$HOME/ros2_ws/install/setup.bash" ] && source "$HOME/ros2_ws/install/setup.bash"
fi

RUN_DIR="$HOME/gp8_runs/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RUN_DIR/motion"

# ---- Pre-flight -------------------------------------------------------------
BRANCH="$(git -C "$PKG_DIR" branch --show-current)"
{
    echo "date:    $(date '+%F %T')"
    echo "branch:  $BRANCH"
    echo "HEAD:    $(git -C "$PKG_DIR" log -1 --oneline)"
    echo "args:    $*"
    echo "dirty:"
    git -C "$PKG_DIR" status --short
} > "$RUN_DIR/meta.txt"

if [ "$BRANCH" != "fix/pushing" ]; then
    # 2026-07-22 lesson: a parallel session switched the shared tree and the
    # robot launched foreign-kinematics code straight into the belt.
    echo "!! branch is '$BRANCH', not fix/pushing — is that intended?"
    read -r -p "   continue anyway? [y/N] " ans
    [ "${ans:-n}" = "y" ] || { echo "aborted."; exit 1; }
fi

if ! timeout 3 ros2 topic echo /conveyor/speed --once >/dev/null 2>&1; then
    echo "!! /conveyor/speed silent (encoder down?) — fire_early_ms will be blank"
    read -r -p "   continue anyway? [y/N] " ans
    [ "${ans:-n}" = "y" ] || { echo "aborted."; exit 1; }
fi

# Leftover cycle CSV from a run not started by this script → stash it aside so
# this run's file starts fresh (schema-safe) and stays per-run.
if [ -f "$HOME/gp8_pick_log.push.csv" ]; then
    mv "$HOME/gp8_pick_log.push.csv" \
       "$HOME/gp8_pick_log.push.stale-$(date +%H%M%S).csv"
    echo "(stashed a leftover ~/gp8_pick_log.push.csv from a previous run)"
fi
THROW_CSV="$HOME/gp8_pick_log.csv"
THROW_MTIME_BEFORE="$([ -f "$THROW_CSV" ] && stat -c %Y "$THROW_CSV" || echo 0)"

echo "[run] logging into $RUN_DIR"
echo "[run] Ctrl+C to stop; collection/analysis run automatically afterwards."

# ---- Launch (console tee + motion CSVs) ------------------------------------
GP8_MOTION_LOG_DIR="$RUN_DIR/motion" \
    ros2 launch gp8_control gp8_bringup.launch.py "$@" 2>&1 \
    | tee "$RUN_DIR/console.log"

# ---- Post-run collection (also runs after Ctrl+C) ---------------------------
echo ""
echo "[run] collecting…"
if [ -f "$HOME/gp8_pick_log.push.csv" ]; then
    mv "$HOME/gp8_pick_log.push.csv" "$RUN_DIR/gp8_pick_log.push.csv"
fi
THROW_MTIME_AFTER="$([ -f "$THROW_CSV" ] && stat -c %Y "$THROW_CSV" || echo 0)"
if [ "$THROW_MTIME_AFTER" -gt "$THROW_MTIME_BEFORE" ]; then
    cp "$THROW_CSV" "$RUN_DIR/gp8_pick_log.throw.csv"
fi

if [ -f "$RUN_DIR/gp8_pick_log.push.csv" ]; then
    python3 "$PKG_DIR/tools/analyze_pick_log.py" \
        "$RUN_DIR/gp8_pick_log.push.csv" > "$RUN_DIR/analysis.txt" 2>&1 || true
fi

C="$RUN_DIR/console.log"
{
    echo "== code sanity =="
    grep -m1 "perception lag" "$C" || echo "(no 'perception lag' line — old code?!)"
    echo; echo "== veto =="
    echo "total:      $(grep -c "push-veto" "$C" || true)"
    echo "near base:  $(grep -c "squeezed near base" "$C" || true)"
    echo "reach edge: $(grep -c "outside reach disc" "$C" || true)"
    echo; echo "== chain =="
    grep "Chain toward next" "$C" | tail -8
    echo; echo "== stroke health =="
    echo "clamped:     $(grep -c "Push stroke clamped" "$C" || true)"
    echo "arc slowed:  $(grep -c "Arc transit slowed" "$C" || true)"
    echo "late warns:  $(grep -c "\[late\]" "$C" || true)"
    echo "aborts:      $(grep -c "Push abort" "$C" || true)"
} > "$RUN_DIR/summary.txt"

echo "[run] done — everything in $RUN_DIR"
echo "---- summary.txt ----"
cat "$RUN_DIR/summary.txt"
[ -f "$RUN_DIR/analysis.txt" ] && { echo "---- analysis.txt (head) ----"; head -25 "$RUN_DIR/analysis.txt"; }
