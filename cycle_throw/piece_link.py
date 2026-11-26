"""Child-process side of the piece mode: one realized timeline, one command fit.

Started by ``cycle_throw.piece_runner.PieceRunner`` like ``planner_link.py``
(cwd = the vendored planner directory, its ``env.sh`` sourced, possibly on
robot6 over ssh). It does NOT run ``rt_cont.run``: the app decides what the
arm does next, as a list of STEPS, and this process plans them as one planned
REALIZED motion (what the arm does after the controller filter) and turns that
into commands with the vendored ``rt_inv`` fit. Vendored parts, unchanged:

  rn2.py   one throw cycle  A contact -> lift -> release -> B contact
  p2p.py   time-optimal realized move between two states (pose, velocity,
           acceleration), inside the planner's limits, TCP z >= zmin
  add_rest.py  throw end -> the fixed wait point (``end``)
  rt_inv   realized target -> commands for the identified controller filter

Steps (app -> "plan" request):
  throw   the object on the cup (arm "obj"); always first when holding one
  push    a push as the app's PushSkill would stream it: COMMANDS from the
          backswing at rest, plus the fire time. Its realized motion is the
          identified filter applied to them (F(U)), i.e. exactly what the real
          arm does with those commands today, shape and timing.
  pick    go to an object and take it (arm ends "obj": contact, dwell)
Between steps the arm moves with p2p from the state it is in, moving or not:
release window end / push end -> (lift) -> above the next target -> down.
There is no rest-to-rest hand-off with a settle hold between steps. The
commands are fitted over the whole span and split only at contact keys (and
in long holds), the way rt_cont splits its cycles.

Timing glue that is rt_cont's (constants read from rt_cont itself): the dwell
on an object, the direct cycle A -> B, release keys, suction times, solve
deadlines, the first-move command lead DW after rest, the end-at-rest hold.

This file must NOT import the app's ``gp8_control`` package (the planner's
own ``gp8_control`` fork must win the import).

Protocol: JSON lines. App -> child on stdin:
  {"t": "start", "t0": <monotonic>, "home": [6]}
  {"t": "pong", "c", "a"}                             (clock sync, as planner_link)
  {"t": "plan", "rid", "steps": [...]}
      {"kind": "throw"}
      {"kind": "pick", "obj": {id, cls, x, y0, b[, z]}}
      {"kind": "push", "id", "U": [[6]..] (robot sign, 2 ms, row 0 = backswing at
       rest), "t_f": fire time of row 0, "i_contact", "i_cut", "q_hover": [6],
       "q_lift": [6] (above the stroke end; both at the travel height H)}
  {"t": "stop"}
Child -> app (original stdout fd):
  {"t": "ready", "H": travel height} {"t": "ping", "c"} {"t": "cmd", "k", "U"}
  {"t": "suc", "on", "at"} {"t": "log", "msg"} {"t": "error", "msg"} {"t": "done", "summary"}
  {"t": "planned", "rid", "ok", "msg", "arm", "taken": [ids], "skipped": [[id, why]..]}
"arm": {"kind": "rest", "q": [6] (robot sign), "t_free", "t_rest"} or
{"kind": "obj", "id", "tc": contact s, "ts": cycle start s}.
"""
import json
import os
import queue
import sys
import threading
import time
import traceback

PLANNER_DIR = os.path.abspath(os.environ.get("GT_PLANNER_DIR", os.getcwd()))

_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
_proto_lock = threading.Lock()


def send(msg):
    line = json.dumps(msg, default=float)
    with _proto_lock:
        _proto.write(line + "\n")
        _proto.flush()


def _p2p_job(a):
    """One vendored p2p solve, in a pool worker (forked after the planner imports)."""
    import db_vin
    import p2p
    lim, q0, v0, a0, q1, v1, zmin, xmin, T0, N = a
    r = p2p.solve(db_vin.M, lim, q0, v0, a0, q1, v1, q0 * 0.0, T0=T0, N=N, zmin=zmin, xmin=xmin)
    return {k: r[k] for k in ("ok", "status", "viol", "T", "tt", "Q", "V", "A", "J")}


def main():
    sys.path.insert(0, PLANNER_DIR)
    os.chdir(PLANNER_DIR)
    os.environ["LPP_VIN"] = "1"
    import numpy as np
    import db_vin as V
    import ordsim as O
    import rt_core as C
    import rt_inv as INV
    import rt_cont as S

    M, SG, H = V.M, np.asarray(V.SG, float), float(V.H)
    DT = C.DT
    lock = threading.Lock()
    requests: "queue.Queue" = queue.Queue()
    started = threading.Event()
    start = {}
    sync = {"samples": [], "offset": 0.0, "rtt": float("inf")}
    # Virtual time (RL twin, backends/sim_tick.py): the app's clock jumps
    # forward by "warp" messages while nothing is being planned. The clock is
    # synced once; a warp moves the offset, commits what falls due and acks.
    VIRTUAL = os.environ.get("GP8_PIECE_VIRTUAL", "0") not in ("0", "", "false", "no")
    hooks = {}
    synced = threading.Event()
    stopped = threading.Event()

    # --- link plumbing (same as planner_link) ------------------------------
    def on_pong(msg):
        t_back = time.monotonic()
        c, a = float(msg["c"]), float(msg["a"])
        rtt = t_back - c
        with lock:
            sync["samples"] = (sync["samples"] + [(rtt, a - (c + 0.5 * rtt))])[-30:]
            sync["rtt"], sync["offset"] = min(sync["samples"])
            if len(sync["samples"]) >= 8:
                synced.set()

    def pinger():
        n = 0
        while not stopped.is_set():
            if VIRTUAL and synced.is_set():
                return                    # offsets move only by warps from here on
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
            elif kind == "pong":
                on_pong(msg)
            elif kind == "warp" and "warp" in hooks:
                hooks["warp"](msg)
            else:
                requests.put(msg)
        requests.put({"t": "stop"})       # stdin closed: the app is gone
        started.set()

    INV.setup(M)
    INV.pool()
    # Moves are solved in parallel (one request has several). Fork this pool
    # before any thread starts, like rt_inv's (see the stdin note below).
    import multiprocessing as mp
    MPOOL = mp.get_context("fork").Pool(int(os.environ.get("GP8_PIECE_MOVE_WORKERS", "6")))
    MPOOL.map(abs, range(6))
    srv = C.Srv(sys.executable)
    # stdin reader only AFTER the fork pool exists (see planner_link).
    threading.Thread(target=reader, name="link_reader", daemon=True).start()
    threading.Thread(target=pinger, name="link_ping", daemon=True).start()
    send({"t": "ready", "H": H})
    started.wait()
    if "t0" not in start or not synced.wait(10.0):
        send({"t": "error", "msg": "no start / clock sync with the app"})
        srv.close()
        return
    send({"t": "log", "msg": f"clock sync: offset {sync['offset']:+.6f} s, "
                             f"best round trip {1e3 * sync['rtt']:.2f} ms"})
    t0 = float(start["t0"])

    def now():
        with lock:
            off = sync["offset"]
        return time.monotonic() + off - t0

    def log(s):
        send({"t": "log", "msg": str(s)})

    def suction(on, t):
        send({"t": "suc", "on": bool(on), "at": float(t)})

    # --- planner parts and constants ---------------------------------------
    # Reach ring for picks (contact radius): the planner's reach_r.json r_hi
    # unless GP8_PIECE_REACH_HI overrides it here (ordsim.yr reads O.R_HI at
    # call time; the vendored file stays as it is). Its models were built
    # inside r_hi = 0.65.
    if os.environ.get("GP8_PIECE_REACH_HI"):
        O.R_HI = float(os.environ["GP8_PIECE_REACH_HI"])
    vy = float(M.VB[1])
    R = json.load(open(os.path.join(C.HERE, "logs", "reach_r.json")))
    yin = lambda x: float(np.floor(np.sqrt(R["r_hi"] ** 2 - x * x) * 100 + 1e-9) / 100)  # noqa: E731
    pos = lambda o, t: (o["x"], o["y0"] + vy * t)                                      # noqa: E731
    D, L = INV._C["D"], INV._C["L"]
    hk = np.asarray(INV._C["hk"], float)
    LAG_S = int(round(float(np.arange(L) @ hk)))          # filter group delay [samples]
    tmp = os.path.join(C.HERE, "logs", "rt_tmp_pieces")
    os.makedirs(tmp, exist_ok=True)
    fn = lambda s: os.path.join(tmp, s)                                                # noqa: E731
    lim_s = os.environ.get("RT_LIM", "0.8,8,160")
    xw_end = 0.40
    env = {"RN_AT_CONTACT": "1", "RN_LIM": lim_s, "AR_XW": str(xw_end), "AR_LIM": lim_s,
           "RN_A0": "0,0,0,0,0,0"}
    top = 0.02
    DW, DWC, NK, IR, IH = S.DW, S.DWC, S.NK, S.IR, S.IH
    IW1 = sum(NK[:4])                                     # release window end (add_rest endto)
    Tdw = DWC * DT
    z6 = np.zeros(6)
    # p2p limits exactly as add_rest builds them (realized motion).
    VU = np.asarray(M.V.QD, float) / float(os.environ.get("GP8_LIM_SCALE", "0.9"))
    KV, KA, KJ = [float(x) for x in lim_s.split(",")]
    QLO, QHI = np.asarray(M.V.Q_LO, float) + 0.02, np.asarray(M.V.Q_HI, float) - 0.02
    # A push's realized motion runs up to the command speed limit (the stream
    # it comes from is tuned to 0.9 of the datasheet = rt_inv's QD).
    QD = np.asarray(INV._C["QD"], float)
    QDD = np.asarray(INV._C["QDD"], float)
    tm = O.TM(os.environ.get("RT_TMODEL") or os.path.join(C.HERE, "logs", "tmodel.npz"))
    cmds = {}
    st = dict(k_def=None, buf=None)
    jq: "queue.Queue" = queue.Queue()
    ev = dict(nlp=[], inv=[], picked=[], releases=[], pushes=0, p2p=[], rest_defaults={})
    zo = lambda o: float(o.get("z", top))                                              # noqa: E731
    load = S.load
    CHMAX = int(os.environ.get("GP8_PIECE_CHUNK", "500"))                # samples per command chunk at most
    # Soft landing on a pick: the last SOFT_DZ above the contact take SOFT_T,
    # decelerating evenly to zero vertical speed at the contact while
    # following the belt (the time-optimal move braked hard at the very end).
    SOFT_DZ = float(os.environ.get("GP8_PIECE_PICK_SOFT_DZ", "0.03"))
    SOFT_T = float(os.environ.get("GP8_PIECE_PICK_SOFT_T", "0.25"))

    def lim_for(v0=None, a0=None):
        """add_rest's limits, widened only as far as a given start state needs."""
        kv, ka, kj = KV, KA, KJ
        if v0 is not None:
            kv = max(kv, min(float(np.max(np.abs(v0) / VU)) + 0.02, float(np.min(QD / VU))))
        if a0 is not None:
            ka = max(ka, min(float(np.max(np.abs(a0) / VU)) + 0.5, float(np.min(QDD / VU))))
        return (kv * VU, ka * VU, kj * VU, QLO, QHI)

    batch_s = []                                          # measured move-batch durations

    def batch_est():
        return max(batch_s[-8:]) if batch_s else 0.8

    def solve_moves(jobs):
        """jobs: (name, q0, v0, a0, q1, v1, zmin, xmin) -> realized p2p pieces, solved in
        parallel (None where a move does not solve)."""
        args = []
        for _nm, q0, v0, a0, q1, v1, zmin, xmin in jobs:
            down = bool(np.abs(np.asarray(v1, float)).max() > 0)
            args.append((lim_for(v0, a0), np.asarray(q0, float), np.asarray(v0, float),
                         np.asarray(a0, float), np.asarray(q1, float), np.asarray(v1, float),
                         float(zmin), float(xmin), 0.3 if down else 0.6, 20 if down else 40))
        w0 = time.monotonic()
        res = MPOOL.map(_p2p_job, args, chunksize=1) if len(args) > 1 else [_p2p_job(args[0])]
        batch_s.append(time.monotonic() - w0)
        out = []
        for (nm, *_r), r in zip(jobs, res):
            ev["p2p"].append(dict(name=nm, ok=r["ok"], T=r["T"], viol=r["viol"]))
            if not r["ok"]:
                log(f"move {nm}: no solution ({r['status']}, violation {r['viol']:.1e})")
            out.append(r if r["ok"] else None)
        return out

    def tcp(q):
        return np.asarray(M.F_TCP(q), float).ravel()[:3]

    # --- command thread (rt_cont do_chunk, copied) ---------------------------
    def do_chunk(ka, n, keys, name, look, still=False):
        for k in range(st["k_def"] + 1, ka + 1):
            cmds[k] = cmds[st["k_def"]]
        hist = np.array([cmds[k] for k in range(ka - L - 4, ka + 1)])
        w0 = time.time()
        nf = n + look
        Qt = st["buf"].rows(ka + D + 1, nf)
        K = [(k_[0] / DT - ka,) + tuple(k_[1:]) for k_ in keys]
        soft = None
        if S.SMOOTH > 0 or ((S.CMD_WIN or S.CMD_SOFT) and K and S.SOFT_W > 0):
            U, inf = INV.chunk(hist, Qt, K, nf, still=(n if still else None),
                               soft=(S.SOFT_W if K else None), smooth=S.SMOOTH)
        else:
            U, inf = INV.chunk(hist, Qt, K, nf, still=(n if still else None))
        if S.SMOOTH <= 0 and not inf["ok"] and S.SOFT_W > 0 and K:
            U2, inf2 = INV.chunk(hist, Qt, K, nf, still=(n if still else None), soft=S.SOFT_W)
            if inf2["ok"]:
                U, inf = U2, inf2
                soft = [round(1e3 * float(np.linalg.norm(tcp(qk) - tcp(q))), 2)
                        for qk, q in zip(inf["Qk"], [k_[1] for k_ in keys])]
                log(f"commands {name}: exact key points not possible inside the command "
                    f"limits -> closest: TCP off by {soft} mm at the key points")
        U = U[:n]
        err = float(np.abs(inf["Qr"][:n] - Qt[:n]).max())
        for i in range(n):
            cmds[ka + 1 + i] = U[i]
        st["k_def"] = ka + n
        send({"t": "cmd", "k": int(ka + 1), "U": np.asarray(U * SG, float).round(9).tolist()})
        m = (ka + 1) * DT - now()
        ev["inv"].append(dict(name=name, s=round(time.time() - w0, 3), ok=bool(inf["ok"]),
                              err_rad=err, n=n, margin=round(m, 3), soft_mm=soft))
        log(f"commands {name}: {time.time() - w0:.2f} s, "
            f"{'ok' if inf['ok'] else 'NO SOLUTION'}, fit {err:.4f} rad, start margin {m:+.2f} s")

    def worker():
        while True:
            j = jq.get()
            try:
                if j is None:
                    return
                do_chunk(**j)
            except Exception:  # noqa: BLE001
                ev["error"] = traceback.format_exc()[-800:]
                send({"t": "error", "msg": "command thread FAILED " + ev["error"]})
                return
            finally:
                jq.task_done()

    def nlp(key, name, t_dead):
        budget = t_dead - now()
        e = dict(env)
        e["RN_MAXT"] = "%.3f" % max(0.05, budget - 0.03)
        out = fn("c.npz")
        if os.path.exists(out):
            os.remove(out)
        r = srv.call("rn2.py", ["%.5f,%.5f,%.5f,%.5f,%d,%.4f,%.4f" % key, out], e)
        if not os.path.exists(out):
            raise RuntimeError(f"plan {name} failed: {r}")
        P = load(out)
        I = json.loads(str(P["info"]))
        ev["nlp"].append(dict(name=name, s=r["s"], T=float(P["tt"][-1]), budget=round(budget, 3),
                              status=I["status"], viol=I["viol"]))
        log(f"plan {name}: {r['s']:.2f} s (deadline in {budget:.2f} s), {P['tt'][-1]:.3f} s long, "
            f"{I['status']}, violation {I['viol']:.1e}")
        return P

    def soft_landing(o, tc):
        """Realized rows on the 2 ms grid from tc - SOFT_T to tc: TCP over the
        object at the belt speed, z = top + SOFT_DZ (1 - s)^2 (constant
        deceleration to rest in z at the contact). Returns (rows, q0, v0)."""
        n = max(2, int(round(SOFT_T / DT)))
        t0_ = tc - n * DT
        rows = []
        for i in range(n + 1):
            t_ = t0_ + i * DT
            s_ = i / n
            rows.append(M.ik(np.array([o["x"], o["y0"] + vy * t_, zo(o) + SOFT_DZ * (1 - s_) ** 2])))
        rows = np.asarray(rows, float)
        v_in = np.array([0.0, vy, -2.0 * SOFT_DZ / (n * DT)])
        return rows, rows[0], M.qd_of(rows[0], v_in), t0_

    def ride(o, t):
        q = M.ik(np.array([o["x"], o["y0"] + vy * t, zo(o)]))
        v = M.qd_of(q, np.asarray(M.VB, float))
        return q, v, dict(tt=np.array([0.0, Tdw]), Q=np.array([q, q + v * Tdw]),
                          V=np.array([v, v]), A=np.zeros((2, 6)), J=np.zeros((1, 6)))

    def samples_piece(R_):
        """Exact realized rows as a Buf piece (zero V/A/J returns row m on [m, m+1))."""
        R_ = np.vstack([R_, R_[-1:]])
        return dict(tt=np.arange(len(R_)) * DT, Q=R_, V=np.zeros_like(R_), A=np.zeros_like(R_),
                    J=np.zeros((len(R_) - 1, 6)))

    # Every command chunk goes through cq. "asap" chunks are handed to the
    # command thread at once, in order. "lazy" chunks (the end of a plan that
    # comes to rest) only when they are due: until then a new request can
    # replace them and carry on from the moving state instead of from rest.
    cq_lock = threading.Lock()
    cq = {"jobs": []}
    COMMIT_AHEAD = S.CMD_MARGIN + 0.15             # a lazy chunk is fitted this long before it starts
    TAIL_CH = int(os.environ.get("GP8_PIECE_TAIL_CHUNK", "50"))   # lazy chunk length = branch spacing

    def cq_replace_from(k):
        """Drop the unsent lazy chunks from sample k on, if the one starting at k
        is still unsent (then a new plan can take over at k). False if the
        default already started going out at k."""
        with cq_lock:
            if not any(j["lazy"] and j["ka"] == k for j in cq["jobs"]):
                return False
            cq["jobs"] = [j for j in cq["jobs"] if not (j["lazy"] and j["ka"] >= k)]
            return True

    def cq_add(jobs, lazy):
        with cq_lock:
            cq["jobs"] += [dict(j, lazy=lazy) for j in jobs]

    def commit_due():
        with cq_lock:
            while cq["jobs"]:
                j = cq["jobs"][0]
                if j["lazy"] and j["ka"] * DT - now() > COMMIT_AHEAD:
                    break
                cq["jobs"].pop(0)
                jq.put({k: v for k, v in j.items() if k != "lazy"})

    def committer():
        while not stopped.is_set() and "error" not in ev:
            commit_due()
            time.sleep(0.005)

    def on_warp(msg):
        """Virtual time: move the clock, then hand out (and fit) every chunk
        that is due by then, so the app never steps past an unsent command."""
        dt = float(msg.get("dt", 0.0))
        with lock:
            sync["offset"] += dt
        commit_due()
        jq.join()
        send({"t": "warped", "seq": msg.get("seq")})

    hooks["warp"] = on_warp
    if VIRTUAL:
        send({"t": "vready"})             # synced, warps are handled from here on

    def ensure_buf(ka, q_rest):
        if st["buf"] is None:
            for k in range(ka - L - 6, ka + 1):
                cmds[k] = np.asarray(q_rest, float)
            st["k_def"] = ka
            st["buf"] = C.Buf(ka, q_rest)
            threading.Thread(target=worker, name="link_cmds", daemon=True).start()
            threading.Thread(target=committer, name="link_commit", daemon=True).start()

    def t_enter(o):
        return (float(O.yr(o["x"])) - o["y0"]) / vy

    def in_reach(o, t):
        return abs(o["y0"] + vy * t) <= float(O.yr(o["x"]))

    def cycle_ok(oa, ob, ts):
        allo = [oa, ob]
        Sx = O.Stream([o["x"] for o in allo], [o["y0"] for o in allo], [o["b"] for o in allo],
                      code=tm.bin_code, z=[zo(o) for o in allo])
        ok, _passed = O.cands(Sx, tm, 0, ts, [1])
        return bool(ok)

    def keys_of(c, ts):
        if S.CMD_WIN:
            return [(ts + float(c["tt"][n_]), c["Q"][n_], c["V"][n_],
                     1.0 if (S.CMD_WIN_REL and n_ == IR) else S.CMD_WIN_W / (NK[2] + NK[3] + 1))
                    for n_ in range(IR - NK[2], IR + NK[3] + 1)]
        return [(ts + float(c["tt"][IR]), c["Q"][IR], c["V"][IR])]

    # --- arm state -----------------------------------------------------------
    home = np.asarray(start["home"], float) * SG
    ka0 = int(np.ceil((now() + 0.3) / DT))
    ensure_buf(ka0, home)
    arm = {"kind": "rest", "q": home, "k_free": ka0, "t_rest": ka0 * DT}

    def arm_out():
        if arm["kind"] == "rest":
            b = open_branch()
            return {"kind": "rest", "q": (arm["q"] * SG).tolist(),
                    "t_free": arm["k_free"] * DT, "t_rest": arm["t_rest"],
                    "t_open": (b + L) * DT if b is not None else None}
        return {"kind": "obj", "id": arm["o"]["id"], "tc": arm["tc"], "ts": arm["tc"] + Tdw}

    # --- planning one request --------------------------------------------------
    class Plan:
        """Pieces of one request, written to the Buf only when all of it is planned."""

        def __init__(self):
            self.pieces = []          # (t_abs, P, t_cut)
            self.keys = []            # (t, q, v[, w])
            self.splits = []          # sample indices: chunk ends (contact keys)
            self.suc = []             # (on, t)

        def add(self, t_abs, P, t_cut=None):
            self.pieces.append((t_abs, P, t_cut))
            return t_abs + float(P["tt"][-1] if t_cut is None else t_cut)

    def chunks(k_a, k_b, keys, splits, last_look, last_still, chmax=None):
        """Command chunks [k_a, k_b): ends at every contact split, long spans
        split again; a key never lands at the start of a chunk."""
        bounds = sorted({int(s) for s in splits if k_a + L < s < k_b} | {k_b})
        out = []
        prev = k_a
        for b in bounds:
            span = b - prev
            parts = max(1, int(np.ceil(span / (chmax or CHMAX))))
            cuts = [prev + int(round(span * (i + 1) / parts)) for i in range(parts)]
            for c_ in cuts[:-1]:                            # keep keys off chunk starts
                for kk in (t / DT for t, *_ in keys):
                    if kk - D - (L // 2 + 1) < c_ <= kk - D:
                        c_ = int(np.floor(kk)) - D
                out.append(c_)
            out.append(b)
            prev = b
        out = sorted(set(x for x in out if x > k_a))
        jobs = []
        a_ = k_a
        for i, b in enumerate(out):
            last = i == len(out) - 1
            look = last_look if last else L - D
            n = b - a_
            kk = [k_ for k_ in keys
                  if a_ + D + 1 + L // 2 <= k_[0] / DT <= a_ + D + n + look]
            jobs.append(dict(ka=a_, n=n, keys=kk, look=look, still=bool(last and last_still)))
            a_ = b
        return jobs

    def filtered(U):
        """Realized motion of commands U (row 0 at rest held before): rows 0 .. n+L-1."""
        n = len(U)
        seq = np.vstack([np.repeat(U[:1], L, axis=0), U, np.repeat(U[-1:], L, axis=0)])
        return np.array([hk @ seq[L + i - np.arange(L)] for i in range(n + L)])

    def q_lift(q):
        """Same TCP x, y at the travel height (planner IK)."""
        p = tcp(q)
        return M.ik(np.array([p[0], p[1], H]))

    def q_hold(q, why=None):
        """Rest at the travel height above where the arm is -- or, when that is
        beyond the planner IK's reach (a throw ends 0.70-0.77 m out), at the
        closest reachable point in the same direction (radius r_hi 0.65, then
        5 cm steps in). None if none lands (the caller uses the wait point)."""
        p = tcp(q)
        r = float(np.hypot(p[0], p[1]))
        ang = float(np.arctan2(p[1], p[0]))
        for rr in [min(r, float(R["r_hi"]))] + [x for x in (0.60, 0.55, 0.50) if x < min(r, R["r_hi"])]:
            target = np.array([rr * np.cos(ang), rr * np.sin(ang), H])
            if target[0] < 0.25:
                continue
            try:
                qh = np.asarray(M.ik(target), float)
                ph = tcp(qh)
            except Exception as e:  # noqa: BLE001
                if why is not None:
                    why.append(f"IK failed at r {rr:.2f}: {type(e).__name__}")
                continue
            if np.all(np.isfinite(qh)) and np.linalg.norm(ph - target) <= 0.01:
                return qh
            if why is not None:
                why.append(f"IK off at r {rr:.2f}")
        return None

    def post_throw_rest(F, nxt, ob, why):
        """Where the arm waits after a throw that does not go straight on, in
        order of preference: above the next object where it first becomes
        pickable, above the next push's backswing (its hover pose), above where
        the throw ends (q_hold). Yields (q, label) candidates."""
        if nxt is not None and nxt.get("kind") == "push":
            yield np.asarray(nxt["q_hover"], float) * SG, f"above the next push ({nxt.get('id')})"
        elif ob is not None:
            t_w = max(F[0] + 0.5, t_enter(ob))
            y_w = ob["y0"] + vy * t_w
            if abs(y_w) <= float(O.yr(ob["x"])):
                try:
                    q_o = np.asarray(M.ik(np.array([ob["x"], y_w, H])), float)
                    if np.all(np.isfinite(q_o)):
                        yield q_o, f"above the next object ({ob['id']})"
                except Exception as e:  # noqa: BLE001
                    why.append(f"no pose above object {ob['id']}: {type(e).__name__}")
            else:
                why.append(f"object {ob['id']} out of reach")
        qh = q_hold(F[1], why)
        if qh is not None:
            yield qh, "where the throw ended"

    def push_geo(stp):
        U = np.asarray(stp["U"], float) * SG
        Sr = filtered(U)
        i_k = min(int(stp["i_contact"]) + LAG_S, len(Sr) - 3)
        i_c = min(int(stp["i_cut"]) + LAG_S, len(Sr) - 3)
        v_c = (Sr[i_c + 1] - Sr[i_c - 1]) / (2 * DT)
        a_c = (Sr[i_c + 1] - 2 * Sr[i_c] + Sr[i_c - 1]) / DT ** 2
        p_bs, p_c = tcp(U[0]), tcp(Sr[i_c])
        return dict(id=int(stp["id"]), k_f=int(round(float(stp["t_f"]) / DT)), Sr=Sr, i_k=i_k,
                    i_c=i_c, cut=(Sr[i_c], v_c, a_c), q_bs=U[0],
                    q_hv=np.asarray(stp["q_hover"], float) * SG,
                    q_up=(np.asarray(stp["q_lift"], float) * SG if "q_lift" in stp else q_lift(Sr[i_c])),
                    z_bs=float(p_bs[2]), x_bs=float(p_bs[0]), z_c=float(p_c[2]), x_c=float(p_c[0]),
                    vmax=float(np.max(np.abs(np.diff(Sr, axis=0)) / DT / QD)))

    def zlo(q):
        return min(H, float(tcp(q)[2])) - 0.003

    def plan_steps(pl, F, steps, name, t0_fn, deadline_fn):
        """Plan pushes and a pick from F = (_, q, v, a); t0_fn() gives F's time once
        the first move batch is solved. Returns (end, taken, skipped) with
        end = ("obj", o, tc, t_app) or ("rest", q, t_end), or None when the moves
        could not be planned before deadline_fn()."""
        taken, skipped = [], []
        geos, pick = [], None
        r_free = None
        for stp in steps:
            kind = stp.get("kind")
            if kind == "push":
                try:
                    geos.append(push_geo(stp))
                except Exception as e:  # noqa: BLE001 - one bad push is skipped, not the plan
                    skipped.append([int(stp.get("id", -1)), f"push geometry: {e}"[:120]])
            elif kind == "pick":
                pick = stp["obj"]
                break
            else:
                skipped.append([-1, f"unknown step {kind!r}"])

        def out_of_time():
            return now() + batch_est() > deadline_fn()

        # Batch 1: every push's moves at once. Each push's approach assumes the
        # step before it is done (it then ends at rest above its stroke end).
        jobs = []
        prev = (F[1], F[2], F[3])
        for g in geos:
            nm = f"{name} push {g['id']}"
            jobs += [(f"{nm} -> above", *prev, g["q_hv"], z6, zlo(prev[0]), 0.249),
                     (f"{nm} down", g["q_hv"], z6, z6, g["q_bs"], z6, g["z_bs"] - 0.003,
                      min(0.249, g["x_bs"] - 0.01)),
                     (f"{nm} lift", *g["cut"], g["q_up"], z6, g["z_c"] - 0.003,
                      min(0.249, g["x_c"] - 0.01))]
            prev = (g["q_up"], z6, z6)
        if jobs and out_of_time():
            return None
        res = solve_moves(jobs) if jobs else []
        t = t0_fn()
        state = (F[1], F[2], F[3])
        chained = True                        # state is what the batch assumed
        for j, g in enumerate(geos):
            a1, a2, lift = res[3 * j: 3 * j + 3]
            if not chained:
                if out_of_time():
                    return None
                a1 = solve_moves([(f"{name} push {g['id']} -> above", *state, g["q_hv"], z6,
                                   zlo(state[0]), 0.249)])[0]
            if a1 is None or a2 is None:
                skipped.append([g["id"], "no move to the backswing"])
                chained = False
                continue
            t_ready = t + float(a1["T"]) + float(a2["T"])
            if t_ready > g["k_f"] * DT - 0.004:   # at rest on the backswing when U starts
                skipped.append([g["id"], f"too late by {t_ready - g['k_f'] * DT:.2f} s"])
                chained = False
                continue
            pl.add(t + 1e-9, a1)
            pl.add(t + float(a1["T"]), a2)
            Sr, i_k, i_c = g["Sr"], g["i_k"], g["i_c"]
            pl.add((g["k_f"] - 1e-6) * DT, samples_piece(Sr[:i_c + 2]))
            t_k = (g["k_f"] + i_k) * DT
            pl.keys.append((t_k, Sr[i_k], (Sr[i_k + 1] - Sr[i_k - 1]) / (2 * DT)))
            pl.splits.append(int(np.floor(t_k / DT)) - D)
            taken.append(g["id"])
            log(f"{name} push {g['id']}: at the backswing {g['k_f'] * DT - t_ready:.2f} s early, "
                f"contact at {t_k:.2f}, realized speed up to {g['vmax']:.2f} of the command limit")
            t_c = (g["k_f"] + i_c) * DT
            r_free = g["k_f"] + i_c + 1               # after the stroke: free to replace
            if lift is None:
                state, t, chained = g["cut"], t_c, False
            else:
                pl.add(t_c + 1e-9, lift)
                state, t, chained = (g["q_up"], z6, z6), t_c + float(lift["T"]), True

        if pick is not None:
            o = pick
            tc = max(t_enter(o) + 0.02, t + 0.6)
            for _ in range(4):
                if not in_reach(o, tc):
                    skipped.append([o["id"], f"out of reach at {tc:.2f}"])
                    break
                qw = M.ik(np.array([o["x"], o["y0"] + vy * tc, H]))
                qc, vc, RD = ride(o, tc)
                soft = soft_landing(o, tc) if SOFT_DZ > 0 and SOFT_T > 0 else None
                if out_of_time():
                    return None
                if soft is not None:            # down to SOFT_DZ above, already moving like the landing
                    q_in, v_in, t_in = soft[1], soft[2], soft[3]
                    down = (f"{name} pick {o['id']} down", qw, z6, z6, q_in, v_in,
                            zo(o) + 0.5 * SOFT_DZ, 0.249)
                else:
                    t_in = tc
                    down = (f"{name} pick {o['id']} down", qw, z6, z6, qc, vc, zo(o), 0.249)
                a1, ap = solve_moves([(f"{name} pick {o['id']} -> above", *state, qw, z6,
                                       zlo(state[0]), 0.249), down])
                if a1 is None or ap is None:
                    skipped.append([o["id"], "no move to the object"])
                    break
                need = t + float(a1["T"]) + float(ap["T"]) + (tc - t_in)
                if tc >= need - 0.004:
                    t_app = t_in - float(ap["T"])
                    log(f"{name} pick {o['id']}: above it {t_app - (t + float(a1['T'])):.2f} s early, "
                        f"contact at {tc:.2f} y {o['y0'] + vy * tc:+.3f}"
                        + (f", soft landing {SOFT_DZ * 100:.0f} cm in {tc - t_in:.2f} s" if soft else ""))
                    pl.add(t + 1e-9, a1)
                    pl.add(t_app, ap)
                    if soft is not None:
                        pl.add((int(round(t_in / DT)) - 1e-6) * DT, samples_piece(soft[0]))
                    pl.add(tc, RD)
                    pl.keys.append((tc, qc, vc))
                    pl.suc.append((True, t_app))
                    taken.append(o["id"])
                    return ("obj", o, tc, t_app), taken, skipped
                tc = need + 0.01
            else:
                skipped.append([o["id"], "pick timing did not settle"])
        # No pick. After a push the arm already rests above its stroke end (the
        # lift): stay there for the next push or pick. Otherwise (nothing done,
        # or the lift did not solve) rest above where the arm is (q_hold); the
        # fixed wait point only if that cannot be reached.
        if taken and chained:
            return ("rest", np.asarray(state[0], float), t, r_free), taken, skipped
        q_wp = q_lift(state[0]) if taken else q_hold(state[0])
        if q_wp is None:
            q_wp = M.ik(np.array([xw_end, yin(xw_end), H]))
        if out_of_time():
            return None
        mv = solve_moves([(f"{name} -> rest", *state, q_wp, z6, zlo(state[0]), 0.249)])[0]
        if mv is None:
            return None
        r_free = int(np.floor(t / DT)) + 1
        t_end = pl.add(t + 1e-9, mv)
        return ("rest", q_wp, t_end, r_free), taken, skipped

    def open_branch():
        """First unsent end-chunk boundary the arm can still branch at, given the
        time a plan takes; None if the last plan's end is already on its way."""
        if arm["kind"] != "rest" or arm.get("r_free") is None:
            return None
        t_need = now() + 3 * batch_est() + 0.1 + COMMIT_AHEAD
        k_min = arm["r_free"] - L                 # the plan leaves the old end at b + L
        with cq_lock:
            for j in cq["jobs"]:
                if j["lazy"] and j["ka"] >= k_min and j["ka"] * DT >= t_need:
                    return int(j["ka"])
        return None

    def do_plan(steps, name):
        if not steps:
            return False, "no steps", [], []
        pl = Plan()
        buf = st["buf"]
        steps = list(steps)
        step_ids = [int(stp["id"]) if stp.get("kind") == "push" else int(stp["obj"]["id"])
                    for stp in steps if stp.get("kind") in ("push", "pick")]
        thrown = None
        fallback = None
        b_open = open_branch() if arm["kind"] == "rest" else None
        if arm["kind"] == "obj":
            if steps[0].get("kind") != "throw":
                return False, "an object is on the cup: the first step must be throw", [], []
            steps.pop(0)
            oa, tc, kc, t_prev = arm["o"], arm["tc"], arm["kc"], arm["t_prev"]
            ts = tc + Tdw
            t_dead = min(t_prev, kc * DT - S.CMD_MARGIN)
            nxt = steps[0] if steps else None
            ob = nxt["obj"] if nxt is not None and nxt.get("kind") == "pick" else None
            if ob is not None and in_reach(ob, ts + 0.3) and cycle_ok(oa, ob, ts):
                # rt_cont's direct cycle A -> B: unchanged.
                c = nlp(pos(oa, ts) + pos(ob, ts) + (oa["b"], zo(oa), zo(ob)), f"{name} throw", t_dead)
                keys = keys_of(c, ts)
                t_rel = ts + float(c["tt"][IR])
                ev["releases"].append(t_rel)
                suction(False, t_rel)
                tcn = ts + float(c["tt"][-1])
                buf.add(ts, c)
                _qn, _vn, RDn = ride(ob, tcn)
                buf.add(tcn, RDn)
                kn = int(np.floor(tcn / DT)) - D
                cq_add([dict(ka=kc, n=kn - kc, keys=keys + [(tcn, c["Q"][-1], c["V"][-1])],
                             name=f"{name} throw -> object {ob['id']}", look=DWC)], lazy=False)
                suction(True, ts + float(c["tt"][IH]))
                arm.clear()
                arm.update(kind="obj", o=ob, tc=tcn, kc=kn, t_prev=tc)
                ev["picked"].append(ob["id"])
                log(f"{name}: object {oa['id']} thrown, straight on to object {ob['id']} "
                    f"(contact at {tcn:.2f})")
                return True, "", [ob["id"]], []
            # The throw, leaving the release window moving.
            if nxt is not None and nxt.get("kind") == "push":
                xw = float(tcp(np.asarray(nxt["q_hover"], float) * SG)[0])
            elif ob is not None:
                xw = ob["x"]
            else:
                xw = xw_end
            xw = min(max(round(xw, 2), 0.25), 0.55)
            # The steps after the throw are planned in the same budget: keep the
            # measured time of their move rounds free (pushes 1, a pick up to 2,
            # the wait point 1), unless that leaves the solve almost nothing.
            n_rounds = (1 if any(x.get("kind") == "push" for x in steps) else 0) + \
                (2 if ob is not None else 0) + 1
            t_dead_nlp = t_dead - n_rounds * batch_est()
            if t_dead_nlp < now() + 0.3:
                t_dead_nlp = t_dead
            c = nlp(pos(oa, ts) + (xw, yin(xw) + 0.06) + (oa["b"], zo(oa), top), f"{name} throw",
                    t_dead_nlp)
            keys = keys_of(c, ts)
            t_rel = ts + float(c["tt"][IR])
            ev["releases"].append(t_rel)
            tcut = float(c["tt"][IW1])
            F = (ts + tcut, c["Q"][IW1], c["V"][IW1], c["A"][IW1])
            # The safe default goes out at once, lazily (one move from the
            # release-window end). A better plan replaces it only if it is
            # ready before the default's first chunk is due; however long
            # planning takes, the arm never waits on it. Where it waits: above
            # the next object (pick) or the next push's backswing, else where
            # the throw ends (post_throw_rest). The fixed upstream wait point
            # (rt_cont's own end) only if none of those can be reached.
            fallback, rest_at = None, None
            why_r: list = []
            for q_r, label in post_throw_rest(F, nxt, ob, why_r):
                mv = solve_moves([(f"{name} -> {label}", F[1], F[2], F[3], q_r, z6,
                                   zlo(F[1]), 0.249)])[0]
                if mv is not None:
                    fallback, rest_at = {k: mv[k] for k in ("tt", "Q", "V", "A", "J")}, label
                    kind_ = label.split(" (")[0]
                    ev["rest_defaults"][kind_] = ev["rest_defaults"].get(kind_, 0) + 1
                    break
                why_r.append(f"no move to {label}")
            if fallback is None:
                log(f"{name}: no rest near the work ({'; '.join(why_r)}); the fixed wait point")
                ev["rest_defaults"]["the fixed wait point"] = ev["rest_defaults"].get("the fixed wait point", 0) + 1
                rest_at = "the fixed wait point"
                srv.call("add_rest.py", ["end", fn("c.npz")], env)
                E = load(fn("c_end.npz"))
                tag = "%02d" % round(xw_end * 100)
                fallback = dict(tt=E["tt_" + tag], Q=E["Q_" + tag], V=E["V_" + tag],
                                A=E["A_" + tag], J=E["J_" + tag])
            buf.add(ts, c, t_cut=tcut)
            suction(False, t_rel)
            t_fb = buf.add(ts + tcut, fallback)
            k_cut = int(np.floor((ts + tcut) / DT))
            k1 = min(int(np.floor(min(k_[0] for k_ in keys) / DT)) - D - L // 2 - 2, k_cut - L - 1)
            k_a = kc
            if k1 - kc >= 20:                          # the throw up to just before its release keys
                cq_add([dict(ka=kc, n=k1 - kc, keys=[], name=f"{name} throw start", look=L - D)],
                       lazy=False)
                k_a = k1
            k_fb = int(np.floor(t_fb / DT)) + L + 20
            fb = chunks(k_a, k_fb, keys, [], 0, True, chmax=TAIL_CH)
            cq_add([dict(j, name=f"{name} default {i + 1}/{len(fb)}") for i, j in enumerate(fb)],
                   lazy=True)
            arm_default = dict(kind="rest", q=np.asarray(fallback["Q"][-1], float), k_free=k_fb,
                               t_rest=t_fb, r_free=k_cut + 1)
            pl.keys += keys
            thrown = oa["id"]
            # nothing after the throw: the default (post_throw_rest) is the plan
            r = None if not steps else plan_steps(
                pl, F, steps, name, lambda: F[0], lambda: k_a * DT - COMMIT_AHEAD)
            useful = r is not None and (r[1] or r[0][0] == "obj")
            if not (useful and cq_replace_from(k_a)):
                arm.clear()
                arm.update(arm_default)
                why = ("planning ran out of time" if steps and (r is None or useful) else None)
                skipped = ([[i, why] for i in step_ids] if why else (r[2] if r else []))
                log(f"{name}: object {oa['id']} thrown; waiting {rest_at} at {t_fb:.2f}"
                    + (f" (steps not ready in time, moves take {batch_est():.2f} s)" if why else "")
                    + (f"; skipped {skipped}" if skipped else ""))
                return True, "", [], skipped
            with buf.lock:                             # the plan takes over from F
                del buf.Q[k_cut - buf.k0 + 1:]
        elif b_open is not None:
            # Carry on from the end of the last plan, which is not sent yet.
            b = b_open
            # The commands sent up to b still shape the realized motion for a
            # filter length (L samples) after them, toward the old end. So the
            # new plan follows the old end for that long and leaves it only
            # then, from its state there: a target the commands can meet (and,
            # after a push, part of the way up the lift). Those L samples of new
            # commands are also the command lead a move from rest needs.
            r_b = b + L
            R3 = buf.rows(r_b - 1, 3)
            qb, vb = R3[1], (R3[2] - R3[0]) / (2 * DT)
            ab = (R3[2] - 2 * R3[1] + R3[0]) / DT ** 2
            moving = float(np.max(np.abs(vb))) > 1e-3
            t_b = r_b * DT
            r = plan_steps(pl, (t_b, qb, vb, ab), steps, name, lambda: t_b,
                           lambda: b * DT - COMMIT_AHEAD)
            if r is None or not r[1] or not cq_replace_from(b):   # the old end goes on
                why = ("planning ran out of time" if (r is None or r[1])
                       else "; ".join(w for _i, w in r[2]))
                return False, why or "nothing planned", [], (
                    [[i, why] for i in step_ids] if (r is None or r[1]) else r[2])
            with buf.lock:
                del buf.Q[r_b - buf.k0 + 1:]
            k_a = b
            log(f"{name}: carried on from {r_b * DT:.2f} ({'moving' if moving else 'at rest'}) "
                "instead of coming to rest")
        else:
            box = {}

            def t0_fn():                          # rt_cont's first-move command lead after rest
                # Not urgent from rest: room for the pick's move rounds and the
                # wait point after this, then the command margin.
                ka = max(int(np.ceil((now() + 3 * batch_est() + S.CMD_MARGIN + 0.1) / DT)),
                         arm["k_free"])
                box["ka"] = ka
                return (ka + D + 1 + DW) * DT

            r = plan_steps(pl, (None, arm["q"], z6, z6), steps, name, t0_fn,
                           lambda: box["ka"] * DT - S.CMD_MARGIN if "ka" in box else float("inf"))
            if r is None or not r[1]:
                why = "planning ran out of time" if r is None else "; ".join(w for _i, w in r[2])
                return False, why or "nothing planned", [], (
                    [[i, why] for i in step_ids] if r is None else r[2])
            k_a = box["ka"]
        end, taken, skipped = r

        # ---- commit: realized pieces, suction, command chunks ----
        for t_abs, P, t_cut in pl.pieces:
            buf.add(t_abs, P, t_cut=t_cut)
        for on, t in pl.suc:
            suction(on, t)
        r_free = None
        if end[0] == "obj":
            _k, ob, tcn, t_app = end
            k_b = int(np.floor(tcn / DT)) - D
            jobs = chunks(k_a, k_b, pl.keys, pl.splits, DWC, False)
            cq_add([dict(j, name=f"{name} part {i + 1}/{len(jobs)}") for i, j in enumerate(jobs)],
                   lazy=False)
        else:
            _k, q_end, t_end, r_free = end
            k_b = int(np.floor(t_end / DT)) + L + 20
            # Commands for everything before r_free are fixed; the way to rest
            # after it goes out lazily in short chunks (branch points).
            k_s = max(k_a, r_free - D - 1)
            jobs = chunks(k_a, k_s, pl.keys, pl.splits, L - D, False) if k_s - k_a > 10 else []
            if not jobs:
                k_s = k_a
            tail = chunks(k_s, k_b, [k_ for k_ in pl.keys if k_[0] / DT > k_s + D + 1], [],
                          0, True, chmax=TAIL_CH)
            cq_add([dict(j, name=f"{name} part {i + 1}/{len(jobs)}") for i, j in enumerate(jobs)],
                   lazy=False)
            cq_add([dict(j, name=f"{name} end {i + 1}/{len(tail)}") for i, j in enumerate(tail)],
                   lazy=True)
            jobs = jobs + tail
        arm.clear()
        if end[0] == "obj":
            arm.update(kind="obj", o=ob, tc=tcn, kc=k_b, t_prev=t_app)
            ev["picked"].append(ob["id"])
        else:
            arm.update(kind="rest", q=np.asarray(q_end, float), k_free=k_b, t_rest=t_end,
                       r_free=r_free)
        pushed = [i for i in taken if end[0] != "obj" or i != ob["id"]]
        ev["pushes"] += len(pushed)
        log(f"{name}: " + (f"object {thrown} thrown; " if thrown is not None else "")
            + (f"pushed {pushed}; " if pushed else "")
            + (f"on to object {ob['id']} (contact at {tcn:.2f})" if end[0] == "obj"
               else f"at rest at {t_end:.2f}")
            + (f"; skipped {skipped}" if skipped else "")
            + f"; {len(jobs)} command part(s) from {k_a * DT:.2f}, now {now():.2f}, "
              f"moves {batch_est():.2f} s")
        return True, "", taken, skipped

    # --- main request loop ---------------------------------------------------
    n_req = 0
    try:
        while "error" not in ev:
            msg = requests.get()
            kind = msg.get("t")
            if kind == "stop":
                if arm["kind"] == "obj":                # never stop holding an object
                    do_plan([{"kind": "throw"}], "stop")
                break
            n_req += 1
            name = f"r{n_req}"
            try:
                if kind == "plan":
                    ok, why, taken, skipped = do_plan(msg.get("steps", []), name)
                else:
                    ok, why, taken, skipped = False, f"unknown request {kind!r}", [], []
            except Exception as e:  # noqa: BLE001 - one failed plan is reported, not fatal
                ok, why, taken, skipped = False, f"{type(e).__name__}: {e}"[:300], [], []
                log(f"{name} FAILED: {traceback.format_exc()[-600:]}")
                if arm["kind"] == "obj":
                    raise                               # holding an object with no plan: stop
            send({"t": "planned", "rid": msg.get("rid"), "ok": bool(ok), "msg": why,
                  "arm": arm_out(), "taken": taken, "skipped": skipped})
        while "error" not in ev:                  # the last pieces' commands, lazy ones too
            with cq_lock:
                left = len(cq["jobs"])
            if not left and not jq.unfinished_tasks:
                break
            time.sleep(0.02)
        send({"t": "done", "summary": dict(
            requests=n_req, picked=list(ev["picked"]), releases=len(ev["releases"]),
            pushes=ev["pushes"], cycle_solves=len(ev["nlp"]),
            cut_at_deadline=sum(1 for x in ev["nlp"] if x.get("status") == "User_Requested_Stop"),
            moves=len(ev["p2p"]), moves_failed=sum(1 for x in ev["p2p"] if not x["ok"]),
            command_pieces=len(ev["inv"]),
            not_solved=sum(1 for x in ev["inv"] if not x.get("ok", True)),
            max_fit_rad=max([x["err_rad"] for x in ev["inv"]] or [0.0]),
            rest_defaults=dict(ev["rest_defaults"]),
            error=ev.get("error"))})
    except Exception:  # noqa: BLE001
        send({"t": "error", "msg": traceback.format_exc()[-1500:]})
    finally:
        stopped.set()
        jq.put(None)
        srv.close()


if __name__ == "__main__":
    main()
