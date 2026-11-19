"""rt_cont: continuous operation with the realized-trajectory cycles (same pieces and rules as rt_seq, objects chosen while running).
 rest pose -> wait above the first contact point -> contact -> [dwell on the object, cycle] ... -> when no next object (or the run ends): throw -> wait point -> again.
 Planning thread (this function): cycle optimizations one after another, each started as soon as the one before is in; a cycle optimization is stopped when the
   cycle before it starts executing and whatever it has is used (user rule 11-11).
 Command thread: the commands of each planned piece (rt_inv.chunk), in order, at the same time as the next optimization.
 Next object: first in, first out — of the objects inside the reach at the expected contact time, the one furthest downstream."""
import os, json, time, threading, queue
import numpy as np
import rt_core as C
import rt_inv as INV
import ordsim as O
DT = C.DT; DW = int(os.environ.get("RT_DW", "89")); DWC = int(os.environ.get("RT_DWC", "60")); NK = [10, 10, 6, 6, 30, 10]; IL = NK[0]; IR = sum(NK[:3]); IH = sum(NK[:5])
SOFT_W = float(os.environ.get("RT_SOFT_W", "1e6")); SMOOTH = float(os.environ.get("RT_CMD_SMOOTH", "0") or 0); T_END_EST = 0.55; AHEAD = float(os.environ.get("RT_AHEAD", "3.0")); LEAD0 = float(os.environ.get("RT_LEAD0", "1.2")); T_SEG = float(os.environ.get("RT_TSEG", "2.2"))
CMD_WIN = os.environ.get("RT_CMD_WIN", "0") == "1"                              # 11-17 (user): the command fit weights the realized state at every node of the release window (1/13 of a key instant each) instead of the release instant alone; setting_now.sh turns it on.  before: rt_cont_pre_win.py.bak
CMD_WIN_W = float(os.environ.get("RT_CMD_WIN_W", "1"))                          # 11-17 (user): factor on the weight of the release window in the command fit (1 = the window as a whole weighs as much as the contact)
CMD_WIN_REL = os.environ.get("RT_CMD_WIN_REL", "0") == "1"                      # 11-18 test: the release instant keeps the full weight of a key instant, RT_CMD_WIN_W / 13 on the other window nodes
CMD_SOFT = os.environ.get("RT_CMD_SOFT", "0") == "1"                            # 11-18 (user: release instant alone, no smoothing, tracking weight 1): key points as weights at once (no exact attempt first)
CMD_MARGIN = float(os.environ.get("RT_CMD_MARGIN", "0.35"))                   # 11-17 (user: a solve that runs long is cut at the deadline): a cycle solve ends at least this long before the first sample of that cycle's command
#   = making the command (measured: median 0.17 s, max 0.25) + the send lead (0.08) + 0.02.  Ordinary cycles are not affected (their deadline, the contact before, is earlier);
#   the first cycle after a wait above an object had its deadline at the start of the descent, 0.05 s before the command was due.  before: rt_cont_pre_cmdmargin.py.bak


def load(f):
    Z = np.load(f, allow_pickle=True); return {k: Z[k] for k in Z.files}


def run(M, SG, H, srv, cands, home_ext, now, commit, suction, log, t_run, lim="0.8,8,160", xw_end=0.40):
    vy = float(M.VB[1]); VB = np.asarray(M.VB, float); R = json.load(open(os.path.join(C.HERE, "logs", "reach_r.json"))); yin = lambda x: float(np.floor(np.sqrt(R["r_hi"] ** 2 - x * x) * 100 + 1e-9) / 100)
    pos = lambda o, t: (o["x"], o["y0"] + vy * t); D = INV._C["D"]; L = INV._C["L"]; tmp = os.path.join(C.HERE, "logs", "rt_tmp"); os.makedirs(tmp, exist_ok=True); fn = lambda s: os.path.join(tmp, s)
    env = {"RN_AT_CONTACT": "1", "RN_LIM": lim, "AR_XW": str(xw_end), "AR_LIM": lim, "RN_A0": "0,0,0,0,0,0"}; top = 0.02; Tdw = DWC * DT; z6 = np.zeros(6)
    ev = dict(nlp=[], inv=[], contacts=[], releases=[], segments=[], picked=[]); cmds = {}; st = dict(k_def=None, buf=None, t_stop=None); taken = set(); jq = queue.Queue()

    def do_chunk(ka, n, keys, name, look, still=False):
        for k in range(st["k_def"] + 1, ka + 1):                               # hold the last command up to the start of this piece
            cmds[k] = cmds[st["k_def"]]
        hist = np.array([cmds[k] for k in range(ka - L - 4, ka + 1)]); w0 = time.time(); nf = n + look; Qt = st["buf"].rows(ka + D + 1, nf)
        K = [(k_[0] / DT - ka,) + tuple(k_[1:]) for k_ in keys]; soft = None
        if SMOOTH > 0 or ((CMD_WIN or CMD_SOFT) and K and SOFT_W > 0):                       # 11-18: also the window weights without the smoothing penalty (weights mean nothing for exact conditions; one solve instead of two) | smooth command (user 11-13): smoothness term + key points as heavy weights; what is lost at the key points is kept in ev["inv"]
            U, inf = INV.chunk(hist, Qt, K, nf, still=(n if still else None), soft=(SOFT_W if K else None), smooth=SMOOTH)
            if K:
                soft = [round(1e3 * float(np.linalg.norm(np.asarray(M.F_TCP(qk)).ravel() - np.asarray(M.F_TCP(q)).ravel())), 2) for qk, q in zip(inf["Qk"], [k_[1] for k_ in keys])]
        else:
            U, inf = INV.chunk(hist, Qt, K, nf, still=(n if still else None))
        if SMOOTH <= 0 and not inf["ok"] and SOFT_W > 0 and K:                                 # no command inside the limits meets the key points exactly (user 11-13): this piece only, key points as close as possible
            U2, inf2 = INV.chunk(hist, Qt, K, nf, still=(n if still else None), soft=SOFT_W)
            if inf2["ok"]:
                U, inf = U2, inf2; soft = [round(1e3 * float(np.linalg.norm(np.asarray(M.F_TCP(qk)).ravel() - np.asarray(M.F_TCP(q)).ravel())), 2) for qk, q in zip(inf["Qk"], [k_[1] for k_ in keys])]
                log(f"commands {name}: exact key points not possible inside the command limits -> closest: TCP off by {soft} mm at the key points (joint {inf['key_err']:.4f} rad, {inf['key_verr']:.3f} rad/s)")
        U = U[:n]; err = float(np.abs(inf["Qr"][:n] - Qt[:n]).max())
        for i in range(n):
            cmds[ka + 1 + i] = U[i]
        st["k_def"] = ka + n; commit(ka + 1, U * SG); m = (ka + 1) * DT - now(); ev["inv"].append(dict(name=name, s=round(time.time() - w0, 3), ok=bool(inf["ok"]), err_rad=err, n=n, margin=round(m, 3), soft_mm=soft, key_rad=inf.get("key_err"), key_rads=inf.get("key_verr")))
        log(f"commands {name}: {time.time() - w0:.2f} s, {'ok' if inf['ok'] else 'NO SOLUTION'}, fit {err:.4f} rad, start margin {m:+.2f} s")

    def worker():
        while True:
            j = jq.get()
            if j is None:
                return
            try:
                do_chunk(**j)
            except Exception as e:  # noqa: BLE001
                import traceback
                log("command thread FAILED " + traceback.format_exc()[-500:]); ev["error"] = str(e); return

    zo = lambda o: float(o.get("z", top))                                      # object top: the perceived one when the object carries it (real robot; Isaac with RT_USE_Z=1), else 0.02

    def nlp(key, name, t_dead):
        budget = t_dead - now(); e = dict(env); e["RN_MAXT"] = "%.3f" % max(0.05, budget - 0.03); out = fn("c.npz")
        if os.path.exists(out):
            os.remove(out)
        r = srv.call("rn2.py", ["%.5f,%.5f,%.5f,%.5f,%d,%.4f,%.4f" % key, out], e)
        if not os.path.exists(out):
            raise RuntimeError(f"plan {name} failed: {r}")
        P = load(out); I = json.loads(str(P["info"])); ev["nlp"].append(dict(name=name, s=r["s"], T=float(P["tt"][-1]), iters=I["iters"], budget=round(budget, 3), status=I["status"], viol=I["viol"], dist=I["start_dist"]))
        log(f"plan {name}: {r['s']:.2f} s (deadline in {budget:.2f} s), {P['tt'][-1]:.3f} s long, {I['status']}, violation {I['viol']:.1e}"); return P

    def reach(q_from, q_to, v_to, out, zmin=top):
        np.savez(fn("p_in.npz"), q_from=q_from, q_to=q_to, v_to=v_to, zmin=zmin); srv.call("add_rest.py", ["move", fn("p_in.npz"), out], env); return load(out)

    def ride(o, t):                                                            # dwell on the object from its contact at t: the pose follows the belt
        q = M.ik(np.array([o["x"], o["y0"] + vy * t, zo(o)])); v = M.qd_of(q, VB); return q, v, dict(tt=np.array([0.0, Tdw]), Q=np.array([q, q + v * Tdw]), V=np.array([v, v]), A=np.zeros((2, 6)), J=np.zeros((1, 6)))

    def choose(t):                                                             # first in, first out: inside the reach at time t, furthest downstream
        c = [o for o in cands() if o["id"] not in taken and abs(o["y0"] + vy * t) <= float(O.yr(o["x"]))]        # contact point inside the reach ring (the whole ring)
        return min(c, key=lambda o: o["y0"] + vy * t) if c else None

    tm = O.TM(os.environ.get("RT_TMODEL") or os.path.join(C.HERE, "logs", "tmodel.npz")); order = os.environ.get("RT_ORDER", "fifo"); pol = O.make_nn(O.MLP(os.environ["RT_ORDER_MODEL"])) if order == "nn" else O.choose_fifo; ev["order"] = []

    def choose_next(oa, ts):                                                   # next object for the cycle that starts at ts holding oa: candidates and choice exactly as in the flow simulator (ordsim)
        c = [o for o in cands() if o["id"] not in taken]
        if not c:
            return None
        allo = [oa] + c; S = O.Stream([o["x"] for o in allo], [o["y0"] for o in allo], [o["b"] for o in allo], code=tm.bin_code, z=[zo(o) for o in allo]); vis = list(range(1, len(allo))); ok, passed = O.cands(S, tm, 0, ts, vis)
        if not ok:
            return None
        i = pol(S, tm, 0, ts, vis, ok, passed); ev["order"].append(dict(ts=round(ts, 3), n=len(ok[:O.K]), i=int(i), T=round(ok[i][1], 4))); return allo[ok[i][0]]

    def first(t_min):                                                          # first object of a segment and its contact time (>= t_min)
        o = choose(t_min)
        if o is not None:
            return o, t_min
        c = [o for o in cands() if o["id"] not in taken and o["y0"] + vy * t_min > float(O.yr(o["x"]))]
        if not c:
            return None, None
        c = sorted(c, key=lambda o: (float(O.yr(o["x"])) - o["y0"]) / vy)[:3]; te = [(float(O.yr(o["x"])) - o["y0"]) / vy for o in c]        # nobody inside yet: start with the first to enter, late enough that the next two are inside at their turn
        return c[0], O.start_time([o["x"] for o in c], [o["y0"] for o in c], [o["b"] for o in c], te, tm, [zo(o) for o in c]) + 0.05                                # their turns predicted with the cycle-time model (no constant cycle time)

    T_GO = (L + 22 + D + DW) * DT + 0.01                                      # at rest above the contact point -> earliest start of the descent: the commands of the move settle (L + 20 samples) and the descent piece starts D + 2 + DW samples before the descent
    BRIDGE = os.environ.get("RT_HOVER_NEXT", "1") != "0"

    def upcoming(t_free):                                                      # known object that cannot be reached in the coming cycle: the one that can be touched first once the arm is free at t_free (user 11-13)
        best = None
        for o in cands():
            if o["id"] in taken:
                continue
            w = float(O.yr(o["x"])); te = (w - o["y0"]) / vy; t1 = max(te + 0.02, t_free)                     # te: the moment it enters the reach ring
            if o["y0"] + vy * (t1 + 1.0) >= -w and (best is None or t1 < best[1]):                             # still well inside one second later
                best = (o, t1, te)
        return best

    def segment(q_rest, k_free, n_seg):
        while True:                                                            # first object
            t_min = max(now() + LEAD0, k_free * DT) + T_SEG; o1, tc1 = first(t_min)
            if o1 is not None or (st["t_stop"] is not None and now() > st["t_stop"]):
                break
            time.sleep(0.05)
        if o1 is None or (st["t_stop"] is not None and tc1 > st["t_stop"]):
            return None
        taken.add(o1["id"]); qw = M.ik(np.array([o1["x"], o1["y0"] + vy * tc1, H])); MV = reach(q_rest, qw, z6, fn("mv.npz")); ka = max(int(np.ceil((now() + 0.8) / DT)), k_free)
        if st["buf"] is None:
            for k in range(ka - L - 6, ka + 1):
                cmds[k] = np.asarray(q_rest, float)
            st["k_def"] = ka; st["buf"] = C.Buf(ka, q_rest); threading.Thread(target=worker, daemon=True).start()
        buf = st["buf"]; buf.add((ka + D + 1 + DW) * DT, MV); n0 = int(float(MV["T"]) / DT) + L + 20 + DW; jq.put(dict(ka=ka, n=n0, keys=[], name=f"s{n_seg} rest -> wait", look=0, still=True)); k_last = ka + n0
        qc, vc, RD = ride(o1, tc1); AP = reach(qw, qc, vc, fn("ap.npz"), zmin=zo(o1)); t_app = tc1 - float(AP["T"]); buf.add(t_app, AP); buf.add(tc1, RD); kA = int(np.floor(t_app / DT)) - D - 2 - DW
        if kA < k_last:
            log(f"segment {n_seg}: the dwell at the wait pose is cut short by {(k_last - kA) * DT:.3f} s"); kA = k_last
        kc = int(np.floor(tc1 / DT)) - D; jq.put(dict(ka=kA, n=kc - kA, keys=[(tc1, qc, vc)], name=f"s{n_seg} wait -> contact", look=DWC)); suction(True, t_app)
        if st["t_stop"] is None:
            st["t_stop"] = float(os.environ["RT_STOP_SIM"]) if os.environ.get("RT_STOP_SIM") else tc1 + t_run; ev["t_first"] = tc1        # RT_STOP_SIM: keep throwing until this sim time (user 11-13: fixed sim-time window, cut in the analysis)
        ev["contacts"].append(tc1); ev["picked"].append(o1["id"]); ev["segments"].append(dict(t_app=t_app, tc1=tc1)); log(f"segment {n_seg}: object {o1['id']} ({o1['cls']}, x {o1['x']:.3f}) contact at {tc1:.2f} (now {now():.2f})")
        oa = o1; tc = tc1; t_prev = t_app; i = 0

        def bridge(nx, c, ts, keys, i):
            """Last throw of a run of cycles, a later object known: end at rest above its first contact point, wait, go down. False = not possible (old way)."""
            on, _t1, te = nx; tcut = float(c["tt"][sum(NK[:4])]); T_ap = 0.20; t_end = ts + T_END_EST; EP = None       # first guess: at rest 0.55 s after the cycle starts, descent 0.20 s (measured 0.50 and 0.17)
            for _ in range(3):                                                 # the contact time and the end piece depend on each other
                tc1 = max(te + 0.02, t_end + T_GO + T_ap); qw = M.ik(np.array([on["x"], on["y0"] + vy * tc1, H])); np.savez(fn("qto.npz"), q_to=qw)
                r = srv.call("add_rest.py", ["endto", fn("c.npz"), fn("qto.npz"), fn("ep.npz")], env); EP = load(fn("ep.npz")) if os.path.exists(fn("ep.npz")) else None
                if EP is None or not bool(EP["ok"]):
                    log(f"wait above object {on['id']}: no end piece ({r.get('tail', '')[-80:]}) -> fixed wait point"); return False
                t_end = ts + tcut + float(EP["T"]); qc, vc, RD = ride(on, tc1); AP = reach(qw, qc, vc, fn("ap.npz"), zmin=zo(on)); T_ap = float(AP["T"])
                if tc1 - T_ap >= t_end + T_GO - 0.004:
                    break
            else:
                log(f"wait above object {on['id']}: timing did not settle -> fixed wait point"); return False
            if abs(on["y0"] + vy * tc1) > float(O.yr(on["x"])) or tc1 > st["t_stop"]:
                return False
            nonlocal kc
            taken.add(on["id"]); buf = st["buf"]; buf.add(ts, c, t_cut=tcut); buf.add(ts + tcut, EP); k_mid = int(np.floor((ts + tcut) / DT)); k_end = int(np.floor(t_end / DT)) + L + 20
            jq.put(dict(ka=kc, n=k_mid - kc, keys=keys, name=f"s{n_seg} cycle {i} (throw)", look=L - D)); jq.put(dict(ka=k_mid, n=k_end - k_mid, keys=[], name=f"s{n_seg} -> above object {on['id']}", look=0, still=True))
            t_app = tc1 - T_ap; buf.add(t_app, AP); buf.add(tc1, RD); kA = max(int(np.floor(t_app / DT)) - D - 2 - DW, k_end); kc = int(np.floor(tc1 / DT)) - D
            jq.put(dict(ka=kA, n=kc - kA, keys=[(tc1, qc, vc)], name=f"s{n_seg} wait -> contact", look=DWC)); suction(True, t_app)
            ev["contacts"].append(tc1); ev["picked"].append(on["id"]); ev.setdefault("bridges", []).append(dict(t_rest=t_end, t_app=t_app, tc1=tc1, t_enter=te, id=on["id"]))
            log(f"segment {n_seg}: object {on['id']} ({on['cls']}, x {on['x']:.3f}) contact at {tc1:.2f} — waiting above it from {t_end:.2f} (it enters the reach at {te:.2f}; now {now():.2f})")
            st["br"] = (on, tc1, t_app, kc); return True
        while True:                                                            # cycles: [dwell on the object] -> lift -> release -> next contact
            ts = tc + Tdw; i += 1
            while now() < ts - AHEAD:                                          # do not choose the next object earlier than this far ahead
                time.sleep(0.01)
            ob = None if ts > st["t_stop"] else choose_next(oa, ts); last = ob is None; nx = upcoming(ts + T_END_EST + T_GO) if (last and BRIDGE and ts <= st["t_stop"]) else None
            xw = round(nx[0]["x"], 2) if nx else xw_end; bpos = pos(ob, ts) if not last else (xw, yin(xw) + 0.06)
            if not last:
                taken.add(ob["id"])
            c = nlp(pos(oa, ts) + bpos + (oa["b"], zo(oa), zo(ob) if not last else top), f"s{n_seg} cycle {i}", min(t_prev, kc * DT - CMD_MARGIN)); t_prev = tc; t_rel = ts + float(c["tt"][IR]); keys = [(ts + float(c["tt"][n_]), c["Q"][n_], c["V"][n_], 1.0 if (CMD_WIN_REL and n_ == IR) else CMD_WIN_W / (NK[2] + NK[3] + 1)) for n_ in range(IR - NK[2], IR + NK[3] + 1)] if CMD_WIN else [(t_rel, c["Q"][IR], c["V"][IR])]; ev["releases"].append(t_rel); suction(False, t_rel)
            if not last:
                tcn = ts + float(c["tt"][-1]); buf.add(ts, c); qn, vn, RDn = ride(ob, tcn); buf.add(tcn, RDn); kn = int(np.floor(tcn / DT)) - D
                jq.put(dict(ka=kc, n=kn - kc, keys=keys + [(tcn, c["Q"][-1], c["V"][-1])], name=f"s{n_seg} cycle {i}", look=DWC)); suction(True, ts + float(c["tt"][IH]))
                kc = kn; tc = tcn; oa = ob; ev["contacts"].append(tcn); ev["picked"].append(ob["id"])
            elif nx is not None and bridge(nx, c, ts, keys, i):                # wait above the next object and go on (no new start)
                on, tc1, t_app, kc = st["br"]; oa = on; tc = tc1; t_prev = t_app
            else:
                srv.call("add_rest.py", ["end", fn("c.npz")], env); E = load(fn("c_end.npz")); tag = "%02d" % round(xw_end * 100); EP = dict(tt=E["tt_" + tag], Q=E["Q_" + tag], V=E["V_" + tag], A=E["A_" + tag], J=E["J_" + tag])
                tcut = float(E["t_cut"]); buf.add(ts, c, t_cut=tcut); t_end = buf.add(ts + tcut, EP); k_mid = int(np.floor((ts + tcut) / DT)); k_end = int(np.floor(t_end / DT)) + L + 20
                jq.put(dict(ka=kc, n=k_mid - kc, keys=keys, name=f"s{n_seg} cycle {i} (throw)", look=L - D)); jq.put(dict(ka=k_mid, n=k_end - k_mid, keys=[], name=f"s{n_seg} -> wait point", look=0))
                log(f"segment {n_seg} ends: {i} throws, at the wait point at {t_end:.2f}"); return np.asarray(EP["Q"][-1], float), k_end, t_end

    q_rest = np.asarray(home_ext, float) * SG; k_free = 0; n_seg = 0; t_end = None
    while "error" not in ev:
        n_seg += 1; r = segment(q_rest, k_free, n_seg)
        if r is None:
            break
        q_rest, k_free, t_end = r
    jq.put(None); ev["t_end"] = t_end; ev["t_stop"] = st["t_stop"]
    while not jq.empty() and "error" not in ev:
        time.sleep(0.05)
    return ev, st["buf"], cmds
