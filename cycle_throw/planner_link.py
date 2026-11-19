"""Child-process side of the cycle-throw mode: runs the vendored planner unchanged.

Started by ``cycle_throw.runner.CycleThrowRunner`` as a separate Python process,
with the working directory set to the vendored planner directory and the
planner's own ``env.sh`` sourced. It does what that folder's ``rt_dry.py`` does
and calls ``rt_cont.run`` unchanged. Only the world is replaced by a pipe to
the app:

  now()          = the app's monotonic clock - t0. The child may run on another
                   machine (robot6, over ssh), so it estimates the offset to
                   the app's clock by ping/pong (minimum round trip wins) and
                   never assumes the two clocks are the same.
  cands()        = latest object list pushed by the app (README sec. 6 format).
  commit(k, U)   = forwarded to the app (robot sign, sample k at k * 2 ms).
  suction(on, t) = forwarded to the app (planned time, no valve lead).

This file must NOT import the app's ``gp8_control`` package. The vendored
planner brings its own ``gp8_control`` fork, which must win the import.

Protocol: JSON lines. App -> child on stdin:
  {"t": "start", "t0": <monotonic>, "home": [6], "run_s": <float>}
  {"t": "cands", "objs": [{"id", "cls", "x", "y0", "b"[, "z"]}, ...]}
  {"t": "stop"}
  {"t": "pong", "c": <child monotonic of the ping>, "a": <app monotonic>}
Child -> app on the ORIGINAL stdout fd (all other prints go to stderr):
  {"t": "ready"} / {"t": "ping", "c"} / {"t": "cmd", "k", "U"} / {"t": "suc", "on", "at"} /
  {"t": "log", "msg"} / {"t": "done", "summary"} / {"t": "error", "msg"}
"""
import json
import os
import sys
import threading
import time
import traceback

PLANNER_DIR = os.path.abspath(os.environ.get("GT_PLANNER_DIR", os.getcwd()))

# Keep the protocol channel private: every print of the planner (and of
# IPOPT) goes to stderr from here on.
_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
_proto_lock = threading.Lock()


def send(msg):
    line = json.dumps(msg, default=float)
    with _proto_lock:
        _proto.write(line + "\n")
        _proto.flush()


def main():
    sys.path.insert(0, PLANNER_DIR)
    os.chdir(PLANNER_DIR)
    os.environ["LPP_VIN"] = "1"
    import numpy as np
    import db_vin as V
    import rt_core as C
    import rt_inv as INV
    import rt_cont as S

    state = {"objs": [], "stopped": False}
    lock = threading.Lock()
    started = threading.Event()
    start = {}
    # Clock sync: app_monotonic ~= time.monotonic() + offset. Keep the recent
    # samples and use the one with the smallest round trip (its error is at
    # most half that round trip).
    sync = {"samples": [], "offset": 0.0, "rtt": float("inf")}
    synced = threading.Event()

    def on_pong(msg):
        t_back = time.monotonic()
        c, a = float(msg["c"]), float(msg["a"])
        rtt = t_back - c
        with lock:
            sync["samples"] = (sync["samples"] + [(rtt, a - (c + 0.5 * rtt))])[-30:]
            best = min(sync["samples"])
            sync["rtt"], sync["offset"] = best
            if len(sync["samples"]) >= 8:
                synced.set()

    def pinger():
        n = 0
        while not state["stopped"]:
            send({"t": "ping", "c": time.monotonic()})
            n += 1
            time.sleep(0.05 if n < 20 else 1.0)

    def reader():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            kind = msg.get("t")
            if kind == "start":
                start.update(msg)
                started.set()
            elif kind == "cands":
                with lock:
                    if not state["stopped"]:
                        state["objs"] = list(msg.get("objs", []))
            elif kind == "pong":
                on_pong(msg)
            elif kind == "stop":
                with lock:
                    state["stopped"] = True
                    state["objs"] = []
        # stdin closed: the app is gone. Stop picking anything new.
        with lock:
            state["stopped"] = True
            state["objs"] = []
        started.set()

    INV.setup(V.M)
    INV.pool()
    srv = C.Srv(sys.executable)
    # Start reading stdin only AFTER rt_inv's fork() pool exists: a fork while
    # this thread blocks in readline (holding stdin's buffer lock) deadlocks
    # every pool worker when multiprocessing closes sys.stdin in the child.
    # Messages sent before this point wait in the pipe.
    threading.Thread(target=reader, name="link_reader", daemon=True).start()
    threading.Thread(target=pinger, name="link_ping", daemon=True).start()
    send({"t": "ready"})
    started.wait()
    if "t0" not in start or not synced.wait(10.0):
        send({"t": "error", "msg": "no start / clock sync with the app"})
        srv.close()
        return
    with lock:
        send({"t": "log", "msg": f"clock sync: offset {sync['offset']:+.6f} s, "
                                 f"best round trip {1e3 * sync['rtt']:.2f} ms"})

    t0 = float(start["t0"])
    home = np.asarray(start["home"], float)
    run_s = float(start.get("run_s", 1e9))
    lim = os.environ.get("RT_LIM", "0.8,8,160")

    def now():
        with lock:
            off = sync["offset"]
        return time.monotonic() + off - t0

    def cands():
        with lock:
            return [dict(o) for o in state["objs"]]

    def commit(k, U):
        send({"t": "cmd", "k": int(k), "U": np.asarray(U, float).round(9).tolist()})

    def suction(on, t):
        send({"t": "suc", "on": bool(on), "at": float(t)})

    def log(s):
        send({"t": "log", "msg": str(s)})

    try:
        ev, _buf, _cmds = S.run(V.M, V.SG, V.H, srv, cands, home, now, commit, suction, log,
                                run_s, lim=lim)
        cyc = [x for x in ev.get("nlp", []) if "cycle" in x.get("name", "")]
        send({"t": "done", "summary": dict(
            contacts=len(ev.get("contacts", [])), releases=len(ev.get("releases", [])),
            picked=list(ev.get("picked", [])), cycle_solves=len(cyc),
            cut_at_deadline=sum(1 for x in cyc if x.get("status") == "User_Requested_Stop"),
            command_pieces=len(ev.get("inv", [])),
            not_solved=sum(1 for x in ev.get("inv", []) if not x.get("ok", True)),
            error=ev.get("error"))})
    except Exception:  # noqa: BLE001 - report every planner failure to the app
        send({"t": "error", "msg": traceback.format_exc()[-1500:]})
    finally:
        srv.close()


if __name__ == "__main__":
    main()
