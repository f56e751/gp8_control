"""plan_cycle: ONE grasp-and-throw cycle, from the contact with object A to the contact with the next object B (belt 0.12 m/s).

  hold on A (0.12 s)  ->  lift  ->  release (A flies to its bin)  ->  travel  ->  descent  ->  contact with B  ->  hold on B (0.12 s)

It runs the same three steps as the runtime (rt_cont.py), for a single cycle:
  1. plan     : the cycle optimization rn2.py (realized joint trajectory, cubic spline, 72 intervals) through the solver server srv.py,
  2. command  : the joint command whose filtered output is closest to the plan (rt_inv.chunk; the 13 nodes of the release window and the next contact
                weighted heavily, as rt_cont.py does with RT_CMD_WIN=1; tracking weight RT_TRACK_W, smoothing RT_CMD_SMOOTH — env.sh),
  3. realized : the motion that the controller-filter model makes of this command.
Everything is written into one .npz file and a one-line JSON summary is printed (line starting with CYCLE).

usage (after `source ../env.sh`, in this folder):
  python plan_cycle.py --a XA,YA[,ZA] --b XB,YB[,ZB] --bin BIN --out OUT.npz [--maxt SECONDS]
    --a    suction point of object A in the base frame [m] at the instant the cup touches it (t = 0); ZA = height of its top surface (default 0.02)
    --b    suction point of the next object B at the same instant t = 0 (it is still upstream, i.e. at larger y); ZB = its top (default 0.02)
    --bin  bin of object A: 0 | 1 | 2  or  metal | paper | transparent   (bins at x = GT_BIN_X (1.35 m), y = +0.33, 0, -0.33 m, aiming height -0.11 m)
    --maxt wall-clock limit of the optimization [s] (default 60: solve to convergence; the runtime uses the time left before its deadline)
The belt moves along -y at GT_BELT (0.12 m/s), so an object at (x, y) at t = 0 is at (x, y - 0.12 t) at time t.
Both objects must be touched inside the reach ring (horizontal distance 0.25 ... 0.65 m from the base axis, lanes x = 0.25 ... 0.60 m).

Output file (times in seconds, t = 0 = contact with A; joint angles in rad in the PLANNER sign convention — multiply by `sign` for the robot):
  t_release, t_hover, t_next_contact, t_cycle_start   instants (cycle start = end of the hold on A = 0.12 s)
  plan_t, plan_q, plan_v, plan_a (73 nodes x 6), plan_j (72 x 6)      the planned realized trajectory (constant jerk between nodes)
  cmd_t, cmd_q (n x 6), cmd_q_robot = cmd_q * sign                    the command, one sample per 2 ms; cmd_t[i] is the time at which sample i is sent
  real_t, real_q (n x 6), real_tcp (n x 3)                            the motion the filter model realizes from the command, and its TCP path
  sign, target (aiming point), info (JSON: status of the solve, iterations, constraint violation, ...)"""
import os, sys, json, time, argparse
os.environ["LPP_VIN"] = "1"; os.environ.setdefault("LPP_XMIN", "0.25")
import numpy as np
import db_vin as V
import rt_core as C
import rt_inv as INV

ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--bin", required=True); ap.add_argument("--out", required=True); ap.add_argument("--maxt", type=float, default=60.0)
a = ap.parse_args(); M = V.M; SG = np.asarray(V.SG, float); DT = C.DT; G = float(M._GRAV); HOLD = int(os.environ.get("RT_DWC", "60")) * DT; vy = float(M.VB[1]); f3 = lambda x: np.array(x, float).ravel()
BIN = {"metal": 0, "paper": 1, "transparent": 2}; b = BIN[a.bin] if a.bin in BIN else int(a.bin); pa = [float(v) for v in a.a.split(",")]; pb = [float(v) for v in a.b.split(",")]; zA = pa[2] if len(pa) > 2 else 0.02; zB = pb[2] if len(pb) > 2 else 0.02
LIM = os.environ.get("RT_LIM", "0.8,6,160"); SOFT = float(os.environ.get("RT_SOFT_W", "1e6")); SMOOTH = float(os.environ.get("RT_CMD_SMOOTH", "1e-5") or 0)
# ---- 1. plan: the cycle starts when the hold on A ends, HOLD after the contact; the solver takes the positions at that instant (as rt_cont.py)
key = "%.5f,%.5f,%.5f,%.5f,%d,%.4f,%.4f" % (pa[0], pa[1] + vy * HOLD, pb[0], pb[1] + vy * HOLD, b, zA, zB); out = os.path.abspath(a.out); tmp = out[:-4] + "_plan.npz"
srv = C.Srv(sys.executable); w0 = time.time(); r = srv.call("rn2.py", [key, tmp], {"RN_AT_CONTACT": "1", "RN_LIM": LIM, "RN_A0": "0,0,0,0,0,0", "RN_MAXT": "%.3f" % a.maxt}); srv.close()
if not os.path.exists(tmp):
    print("CYCLE " + json.dumps(dict(ok=False, error="the cycle optimization gave no result", detail=r))); raise SystemExit(1)
Z = np.load(tmp, allow_pickle=True); I = json.loads(str(Z["info"])); os.remove(tmp); tt, Q, Vv, A, J = Z["tt"], Z["Q"], Z["V"], Z["A"], Z["J"]; tg = np.array(Z["tg"], float); tc = I["t_cycle"]; tr = I["t_release"]; t_plan = time.time() - w0


def plan(t):                                                                   # planned realized state at time t after the cycle start; before it and after the next contact the TCP rides on the object
    if t <= 0:
        return Q[0] + Vv[0] * t, Vv[0]
    if t >= tc:
        return Q[-1] + Vv[-1] * (t - tc), Vv[-1]
    m = int(np.clip(np.searchsorted(tt, t, side="right") - 1, 0, len(tt) - 2)); h = t - tt[m]; return Q[m] + Vv[m] * h + A[m] * h ** 2 / 2 + J[m] * h ** 3 / 6, Vv[m] + A[m] * h + J[m] * h ** 2 / 2


# ---- 2. command: from the contact with A (the arm rides on A) to the end of the hold on B
INV.setup(M); INV.pool(); hk = INV._C["hk"]; L = INV._C["L"]; D = INV._C["D"]; LAG = float(np.sum(np.arange(L) * hk) / np.sum(hk)); T0 = -HOLD
N = int(np.ceil((tc + 2 * HOLD) / DT)); Qt = np.array([plan(T0 + (n + 1) * DT)[0] for n in range(N)]); qr, vr = plan(tr); qe, ve = plan(tc)
hist = np.array([Q[0] + Vv[0] * (T0 + (m - D + LAG) * DT) for m in range(-(L + 1), 1)])                    # the commands sent while the arm rides on A at the belt speed
WIN = os.environ.get("RT_CMD_WIN", "0") == "1"; WW = float(os.environ.get("RT_CMD_WIN_W", "1")); NK = [10, 10, 6, 6, 30, 10]; IR = sum(NK[:3])      # node IR of the plan = release
KR = [(D + (tt[n_] - T0) / DT, Q[n_], Vv[n_], WW / (NK[2] + NK[3] + 1)) for n_ in range(IR - NK[2], IR + NK[3] + 1)] if WIN else [(D + (tr - T0) / DT, qr, vr)]
w1 = time.time(); U, inf = INV.chunk(hist, Qt, KR + [(D + (tc - T0) / DT, qe, ve)], N, soft=SOFT, smooth=SMOOTH); t_cmd = time.time() - w1; Qr = inf["Qr"]
# ---- 3. realized motion and checks


def st(t):                                                                     # realized joint position and velocity at time t after the cycle start (cubic through 4 samples)
    n = int(np.clip(np.floor((t - T0) / DT) - 1, 1, N - 3)); ts = T0 + (np.arange(n - 1, n + 3) + 1) * DT - t; q = np.zeros(6); v = np.zeros(6)
    for j in range(6):
        cf = np.polyfit(ts, Qr[n - 1:n + 3, j], 3); q[j] = cf[3]; v[j] = cf[2]
    return q, v


def land(q, qd):                                                               # ballistic landing point of the launch point on the aiming plane, distance from the aiming point
    p = f3(M.F_P(q)); v = f3(M.F_V(q, qd)); disc = v[2] ** 2 + 2 * G * (p[2] - tg[2]); tau = (v[2] + np.sqrt(max(disc, 0.0))) / G; return float(np.hypot(*(p[:2] + v[:2] * tau - tg[:2]))), p, v


qR, vR = st(tr); eR, pR, vRl = land(qR, vR); eP, _, _ = land(qr, vr); eL40, _, _ = land(*st(tr + 0.04)); eP40, _, _ = land(*plan(tr + 0.04)); qC, _ = st(tc); con = float(np.linalg.norm(f3(M.F_TCP(qC)) - f3(M.F_TCP(qe)))); n0 = int(HOLD / DT); n1 = int((HOLD + tc) / DT)
real_t = (np.arange(N) + 1) * DT; cmd_t = (np.arange(N) + 1 - D) * DT; tcp = np.array([f3(M.F_TCP(q)) for q in Qr])
np.savez(out, t_release=HOLD + tr, t_hover=HOLD + I["t_hover"], t_next_contact=HOLD + tc, t_cycle_start=HOLD, plan_t=HOLD + tt, plan_q=Q, plan_v=Vv, plan_a=A, plan_j=J, cmd_t=cmd_t, cmd_q=U, cmd_q_robot=U * SG,
         real_t=real_t, real_q=Qr, real_tcp=tcp, sign=SG, target=tg, info=json.dumps(I))
ok = I["status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and I["viol"] <= 1e-5 and bool(inf["ok"])
print("CYCLE " + json.dumps(dict(ok=bool(ok), out=out, status=I["status"], iterations=I["iters"], violation=I["viol"], plan_s=round(t_plan, 2), command_s=round(t_cmd, 3), command_solved=bool(inf["ok"]),
                                 cycle_s=round(tc, 4), contact_to_contact_s=round(HOLD + tc, 4), t_release=round(HOLD + tr, 4), t_next_contact=round(HOLD + tc, 4), samples=int(N),
                                 release_point=np.round(pR, 4).tolist(), release_velocity=np.round(vRl, 3).tolist(), target=np.round(tg, 3).tolist(),
                                 landing_error_mm=dict(plan=round(1e3 * eP, 2), realized=round(1e3 * eR, 2)), landing_error_release_40ms_late_mm=dict(plan=round(1e3 * eP40, 1), realized=round(1e3 * eL40, 1)),
                                 command_fit=dict(release_window=WIN, tracking_weight=float(os.environ.get("RT_TRACK_W", "1")), smoothing=SMOOTH), next_contact_error_mm=round(1e3 * con, 3),
                                 largest_plan_vs_realized_deg=round(float(np.degrees(np.abs(Qr[n0:n1] - Qt[n0:n1]).max())), 2), lowest_tcp_z=round(float(tcp[:, 2].min()), 4))), flush=True)
