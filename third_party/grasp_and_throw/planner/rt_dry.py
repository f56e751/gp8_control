"""rt_dry: the real-time pipeline WITHOUT a simulator and WITHOUT the robot (robot PC check, 2026-11-13).
Same planning thread, command thread and order rule as rt_run2 / rt_cont (rt_cont.run is called unchanged); only the world is replaced:
 clock    = this machine's monotonic clock (0 at start),
 objects  = a spawn schedule file (logs/pat/*.json, as given to Isaac): object on the belt line y(t) = 1.40 - 0.12 (t - t_spawn), visible from y = 1.30,
 commands = kept in memory (the same 'already sent' bookkeeping as the Isaac sender: a piece committed after its first sample was due counts as late),
 suction  = recorded only.
Nothing is sent anywhere.  It shows whether THIS machine plans fast enough: solve times, time left at each deadline, solves cut at the deadline, late commands.
usage: RT_ORDER=nn RT_ORDER_MODEL=logs/ord_full/m1.npz RT_STOP_SIM=80 python rt_dry.py NAME SCHEDULE.json"""
import os, sys, json, time, threading
os.environ["LPP_VIN"] = "1"
import numpy as np
import db_vin as V
import rt_core as C, rt_inv as INV, rt_cont as S
M = V.M; SG = V.SG; H = V.H; DT = C.DT; name = sys.argv[1]; sched = json.load(open(sys.argv[2])); LIM = os.environ.get("RT_LIM", "0.8,8,160"); LEAD = 0.08; SPAWN_Y = 1.40; VIS_Y = float(os.environ.get("RT_DRY_VIS", "1.30")); T_OFF = 3.6
run = time.strftime("%Y%m%d_%H%M%S") + "_" + name; out = os.path.join("logs", run); os.makedirs(out, exist_ok=True)
log = lambda s: print(time.strftime("%H:%M:%S ") + s, flush=True)
BIN = {"metal": 0, "paper": 1, "transparent": 2}                                # class -> bin index (cyc2.BINS order: y = +0.33, 0, -0.33), as in the Isaac runs
VBELT = float(os.environ.get("RGE_BELT_V", "0.12")); objs = [dict(id=d["id"], cls=d["cls"], x=float(d["x"]), y0=SPAWN_Y + VBELT * (d["t"] + T_OFF), b=BIN[d["cls"]]) for d in sched]
if os.environ.get("RT_DRY_TOPS"):                                             # perceived tops: one of the list per object (seeded)
    import random
    _r = random.Random(7); _t = [float(v) for v in os.environ["RT_DRY_TOPS"].split(",")]
    for o in objs:
        o["z"] = _r.choice(_t)
q0 = np.array([0.0, 0.28890832, -0.52880029, 0.0, -0.75308772, -1.13446401]); INV.setup(M); INV.pool(); srv = C.Srv(sys.executable); SLOW = float(os.environ.get("RT_DRY_SLOW", "1")); t00 = time.monotonic(); now = lambda: (time.monotonic() - t00) / SLOW        # RT_DRY_SLOW > 1: slowed clock (studies only)
cmd = {}; st = dict(k_sent=int(np.ceil(now() / DT)) + 1, late=0, late_s=[], stop=False); lk = threading.Lock(); suc = []


def sender():                                                                  # what the Isaac sender does to the bookkeeping: samples up to now + LEAD count as sent
    while not st["stop"]:
        k1 = int((now() + LEAD) / DT)
        with lk:
            st["k_sent"] = max(st["k_sent"], k1)
        time.sleep(0.001)


def commit(k, U):
    with lk:
        if k <= st["k_sent"]:
            st["late"] += 1; st["late_s"].append((st["k_sent"] - k) * DT); log(f"late commit: first sample {k} already sent up to {st['k_sent']} ({(st['k_sent'] - k) * DT:.3f} s)")
        for i, u in enumerate(U):
            cmd[k + i] = np.asarray(u, float)


def cands():
    t = now(); return [dict(o) for o in objs if -0.9 <= o["y0"] - VBELT * t <= VIS_Y]


threading.Thread(target=sender, daemon=True).start(); res = dict(run=run, lim=LIM, host=os.uname().nodename)
try:
    ev, buf, cmds = S.run(M, SG, H, srv, cands, q0, now, commit, lambda on, t: suc.append((bool(on), float(t), now())), log, float(os.environ.get("RT_RUN_S", "60")), lim=LIM); res.update(ev=ev)
except Exception as e:  # noqa: BLE001
    import traceback
    log("FAILED " + traceback.format_exc()[-800:]); res["error"] = str(e); ev = None
st["stop"] = True; srv.close(); res["late"] = st["late"]; res["late_s"] = st["late_s"]; res["suction"] = suc
json.dump(res, open(os.path.join(out, "rt_run.json"), "w"), indent=1, default=str); ks = sorted(cmd); np.savez(os.path.join(out, "cmds.npz"), k=np.array(ks), cmd=np.array([cmd[k] for k in ks]) if ks else np.zeros((0, 6)))
if ev:
    W0, W1 = 15.0, 75.0; c = np.array(ev["contacts"]); cw = c[(c >= W0) & (c < W1)]; cyc = [x for x in ev["nlp"] if "cycle" in x["name"]]; s = np.array([x["s"] for x in cyc]); bd = np.array([x["budget"] for x in cyc])
    cut = [x for x in cyc if x["status"] == "User_Requested_Stop"]; inv = ev["inv"]; mg = np.array([x["margin"] for x in inv]); od = [x for x in ev.get("order", []) if W0 <= x["ts"] < W1]
    U = np.array([cmd[k] for k in ks]); dv = np.abs(np.diff(U, axis=0)).max(0) / DT if len(U) > 1 else np.zeros(6)
    print("SUMMARY %s (%s, order %s): contacts in %.0f-%.0f s: %d (interval median %.3f s) | cycle solves %d: time median %.2f s, 90 %% %.2f, max %.2f | time left at the start of a solve: median %.2f s, 10 %% %.2f, min %.2f | cut at the deadline %d (largest violation %.1e) | "
          "late command pieces %d (max %.3f s) | command pieces %d: NOT SOLVED %d, start margin min %.2f s, fit error max %.3f rad | non-FIFO choices %d of %d | suction switches %d | largest command step per joint [rad/s] %s"
          % (run, res["host"], os.environ.get("RT_ORDER", "fifo"), W0, W1, len(cw), float(np.median(np.diff(cw))) if len(cw) > 2 else float("nan"), len(cyc), np.median(s), np.percentile(s, 90), s.max(), np.median(bd), np.percentile(bd, 10), bd.min(),
             len(cut), max([x["viol"] for x in cut] or [0.0]), st["late"], max(st["late_s"] or [0.0]), len(inv), sum(1 for x in inv if not x["ok"]), mg.min(), max(x["err_rad"] for x in inv), sum(1 for x in od if x["i"] > 0), len(od), len(suc), np.round(dv, 2).tolist()))
print("command pieces solved with the key points as close as possible (name, TCP error at each key point [mm]):", [(x["name"], x["soft_mm"]) for x in (ev["inv"] if ev else []) if x.get("soft_mm")])
print("RUN=" + run)
