"""대기점 구간 붙이기 (11-11 사용자: 대기점에서 시작하는 사이클·대기점으로 끝나는 사이클도 DB 로).
대기점 = 그 x 의 진입점 (접근 범위에 처음 들어오는 y) 위 높이 h 에서 정지.
 start: 풀린 사이클 (A 가 진입점에서 닿는 칸) 앞에 '대기점 정지 → A 접촉' 을 붙인다 (끝 가속 = 사이클의 시작 가속 → 매끄럽게 이어짐).  → *_app.npz
 end  : 풀린 사이클의 릴리즈 창 끝에서 '→ 대기점 정지' 를 붙인다 (대기점 x = 0.25, 0.35, 0.40, 0.45, 0.55 각각).  → *_end.npz
사용: python add_rest.py start|end 칸파일.npz"""
import os, sys, json
os.environ["LPP_VIN"] = "1"
import numpy as np
import db_vin as V
import p2p
M = V.M; H = V.H; TOP = V.D.ZTOP; VU = np.asarray(M.V.QD, float) / float(os.environ.get("GP8_LIM_SCALE", "0.9")); KV, KA, KJ = [float(x) for x in os.environ.get("AR_LIM", os.environ.get("RN_LIM", "0.8,8,160")).split(",")]   # AR_LIM: limits for the pieces that start or end at rest
LIM = (KV * VU, KA * VU, KJ * VU, np.asarray(M.V.Q_LO, float) + 0.02, np.asarray(M.V.Q_HI, float) - 0.02); R = json.load(open("logs/reach_r.json"))
yin = lambda x: float(np.floor(np.sqrt(R["r_hi"] ** 2 - x * x) * 100 + 1e-9) / 100); wait_q = lambda x: M.ik(np.array([x, yin(x), H])); z6 = np.zeros(6)
mode, f = sys.argv[1], sys.argv[2]
if mode == "move":                                                           # 정지 → 정지: 자세 파일 (from, to) → 궤적
    P = np.load(f); vt = P["v_to"] if "v_to" in P.files else z6; down = bool(np.abs(vt).max() > 0)        # v_to given and nonzero: reach down to a contact (floor = object top)
    r = p2p.solve(M, LIM, P["q_from"], z6, z6, P["q_to"], vt, z6, T0=(0.3 if down else 0.6), N=(20 if down else 40), zmin=((float(P["zmin"]) if "zmin" in P.files else TOP) if down else H - 0.001), xmin=0.249)
    np.savez(sys.argv[3], ok=r["ok"], T=r["T"], tt=r["tt"], Q=r["Q"], V=r["V"], A=r["A"], J=r["J"]); print("MOVE", json.dumps(dict(ok=r["ok"], T=r["T"], viol=r["viol"]))); raise SystemExit
Z = np.load(f, allow_pickle=True); I = json.loads(str(Z["info"])); Q, Vv, A = Z["Q"], Z["V"], Z["A"]; NK = I["NK"]; iW1 = sum(NK[:4])
if mode == "endto":                                                          # from the end of the release window to a given pose at rest (11-13: wait above the next object's first contact point)
    qt = np.load(sys.argv[3])["q_to"]; r = p2p.solve(M, LIM, Q[iW1], Vv[iW1], A[iW1], qt, z6, z6, T0=0.5, N=40, zmin=H - 0.001, xmin=0.249)
    np.savez(sys.argv[4], ok=r["ok"], T=r["T"], tt=r["tt"], Q=r["Q"], V=r["V"], A=r["A"], J=r["J"], t_cut=float(Z["tt"][iW1])); print("ENDTO", json.dumps(dict(ok=r["ok"], T=r["T"], viol=r["viol"]))); raise SystemExit
XW = [float(x) for x in os.environ.get("AR_XW", "0.25,0.35,0.40,0.45,0.55").split(",")]
if mode == "start":
    xa, ya = I["key"][0], I["key"][1]; r = p2p.solve(M, LIM, M.ik(np.array([xa, ya, H])), z6, z6, Q[0], Vv[0], A[0], T0=0.25, N=20, zmin=TOP, xmin=0.249)   # wait pose = right above the contact point
    np.savez(f[:-4] + "_app.npz", ok=r["ok"], T=r["T"], tt=r["tt"], Q=r["Q"], V=r["V"], A=r["A"], J=r["J"], x_wait=xa, y_wait=ya)
    print("APP", json.dumps(dict(key=I["key"], ok=r["ok"], status=r["status"], T=r["T"], viol=r["viol"])), flush=True)
else:
    out = {}
    for xw in XW:
        r = p2p.solve(M, LIM, Q[iW1], Vv[iW1], A[iW1], wait_q(xw), z6, z6, T0=0.5, N=40, zmin=H - 0.001, xmin=0.249); tag = "%02d" % round(xw * 100)
        out.update({f"ok_{tag}": r["ok"], f"T_{tag}": r["T"], f"tt_{tag}": r["tt"], f"Q_{tag}": r["Q"], f"V_{tag}": r["V"], f"A_{tag}": r["A"], f"J_{tag}": r["J"]})
        print("END", json.dumps(dict(key=I["key"], x_wait=xw, ok=r["ok"], status=r["status"], T=r["T"], viol=r["viol"], t_window_end=float(Z["tt"][iW1]))), flush=True)
    np.savez(f[:-4] + "_end.npz", i_cut=iW1, t_cut=float(Z["tt"][iW1]), **out)
