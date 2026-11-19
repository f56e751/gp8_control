"""11-09 사용자 "다 실제 궤적이 지나도록 해야지" + "두 점 속도 고정하지 마, cycle 한 번에, 명령 모양·관절 동기 제한 없이 다 free":
하강 직전 hover·상승 끝 = 실제 관절 위치 (= 물체 위 h, 수직 자세) 만 등식, 속도 전부 자유. 접촉 = 위치 + 물체 대비 속도 0, 릴리즈 = 위치 + 던지기 속도.
 사이클 = 실제 (필터) 궤적이 정확히 지나는 점 4 개 (하강 직전 hover, 접촉, 상승 끝, 릴리즈),
점 사이는 관절별 최단 시간 (6 관절 한 LP: 2 ms 명령 + 두 점의 수직 속도 변수). 다음 물체 접촉까지 내다보고 계획, 실행은 다음 하강 직전 hover 까지.
시각 배치: 앞에서부터 가장 이른 시각 (greedy) → 앞 점 시각을 미뤄 보며 다음 hover 가 빨라지면 받아들이는 국소 탐색.
릴리즈: 빈별 최적화 값. 판정: 실행 명령 전체에 실행기 모델 (지연 0.04 + 평균 0.2·0.08).
사용: python lp_cycle2.py [n_cycles]"""
import os, sys, time, numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
import v3ref as V
from v3ref import ik, qd_of, SG, DT, H, VB, BINS, ZI, LAG, hk, tcp_batch, F_LAND, F_W, F_JT, F_TCP, F_AX, release_state
import cyc2 as C
import jcyc1 as _J1

F_LVEC = _J1.landing_vec()                                                     # (q, q̇, 조준점) → 착지 오차 xy (2)
from v3ref import F_P, F_V                                                     # 발사점 (TCP + 2 cm) 위치·속도
_GRAV = float(getattr(getattr(_J1, "T", None), "G", 9.81))


def relv_res(q, qd, tg):
    """릴리즈 속도 조건의 잔차 (3): 발사점 속도 − 농구식 최소 속력 발사 속도 v*(발사점 위치, 조준점).
    11-10 사용자: 조준점과 발사각 (θ* = 45° + ½ atan(h/R), v² = g(h + √(R² + h²))) 으로 xyz 속도가 정해진다 — 위치·각속도는 자유."""
    p = np.array(F_P(q)).ravel(); v = np.array(F_V(q, qd)).ravel(); tg = np.asarray(tg, float)
    d = tg[:2] - p[:2]; R = float(np.linalg.norm(d)); h = float(tg[2] - p[2])
    th = np.pi / 4 + 0.5 * np.arctan2(h, R); sp = np.sqrt(max(_GRAV * (h + np.hypot(R, h)), 1e-9))
    return v - np.r_[sp * np.cos(th) * d / R, sp * np.sin(th)]
L = len(hk)
O0 = [np.array(p) for p in ([0.40, 0.15, ZI], [0.50, 0.40, ZI], [0.35, 0.62, ZI], [0.45, 0.85, ZI], [0.40, 1.05, ZI], [0.50, 1.25, ZI])]
BIN = [0, 1, 2, 0, 1]
at = lambda k, t: O0[k] + VB * t
RELP = {0: (0.65, 0.40, 1.2, -2.0), 1: (0.60, 0.30, 0.8, -2.0), 2: (0.65, 0.40, 1.2, -2.0)}
REL = {b: release_state(np.array(BINS[b], float), *RELP[b]) for b in range(3)}
VZ = lambda q: qd_of(q, np.array([0.0, 0.0, 1.0])) - qd_of(q, np.zeros(3))
NLP = [0]


BAND = float(os.environ.get("LPC_BAND", "0.01"))                              # 통로 폭 [rad] (수직선 관절 경로에서 수직 방향 성분 밖)


def _ik_safe(p):
    """IK 실패 (도달 밖) 면 None — 호출한 사건은 None 이 되고 lp() 는 그 시각을 불가로 본다 (11-10)."""
    try:
        return ik(p)
    except Exception:                                                         # noqa: BLE001
        return None


ZRELAX = float(os.environ.get("LPC_ZRELAX", "0.01"))


def lp(q0, qd0, Sh, events, N, corridors=(), smooth=False, zcons=(), slp=None, win=None):
    """z 하한은 기준 자세 주변 1 차 선형화라, 종점이 하한 위에 딱 붙어 있으면(hover·준비점 z = h) 선형화 오차만으로 불가가 날 수 있다 (11-10 확인:
    1 cm 여유 해의 실제 z 최소는 정확히 h, 그 해 주변에서 다시 선형화하면 정확한 h 로 가능). → 불가면 하한을 ZRELAX 만 낮춰 기준 궤적을 얻고,
    그 실제 궤적 주변에서 **원래 하한 그대로** 다시 푼다 (제약은 바뀌지 않음, 선형화 점만 바뀜)."""
    r = _lp_core(q0, qd0, Sh, events, N, corridors, smooth, zcons, slp, win)
    if r is not None or not zcons or ZRELAX <= 0 or any(e is None for e in events):
        return r
    zc2 = [(c[0], c[1], c[2] - ZRELAX) + tuple(c[3:]) for c in zcons]
    r2 = _lp_core(q0, qd0, Sh, events, N, corridors, smooth, zc2, slp, win)
    if r2 is None:
        return None
    qr = realized_q(q0, Sh, r2[0])
    zc3 = [(c[0], qr[c[0] - 1] if 1 <= c[0] <= len(qr) else c[1], c[2]) + tuple(c[3:]) for c in zcons]
    return _lp_core(q0, qd0, Sh, events, N, corridors, smooth, zc3, slp, win)


def _lp_core(q0, qd0, Sh, events, N, corridors=(), smooth=False, zcons=(), slp=None, win=None):
    if any(e is None for e in events):
        NLP[0] += 1; return None
    """zcons: [(n, qref (6), zmin)] — 실제 TCP z(q_n) ≥ zmin 을 qref 주위 1 차 선형화 (11-09 사용자: 하강·상승은 윗면 이상, 그 외 h 이상).
    slp: dict(tr, gaps, ns) — 점마다 시각 이동 δ_i [s] 변수 추가 (|δ| ≤ tr). 사건에 dpos/dvel (6,) 이 있으면 위치·속도 등식 행에 δ_i 계수로 들어감
         (q[n] + q̇ref·δ = κ(n) + κ̇·δ → δ 계수 dpos = q̇ref − κ̇; 속도 행 dvel = q̈ref − v̇_target). 순서: (n_i + gap_{i+1})·DT + δ_i ≤ n_{i+1}·DT + δ_{i+1}.
         목적 = δ_3 (다음 hover 시각) 최소. 리턴 꼬리에 δ 들이 ż 뒤에 붙음."""
    NLP[0] += 1; N = N + 2
    ns = sum(1 for e in events if e.get("vd") is not None); nd = len(events) if slp else 0; nv = 6 * N + ns + nd
    I, Jc, Vv, bub = [], [], [], []; r = 0; n = np.arange(1, N + 1)
    for j in range(6):
        A, Vm = V.QDD[j], V.QD[j]; base = j * N
        for sgn in (1.0, -1.0):
            I += list(r + n - 1); Jc += list(base + n - 1); Vv += [sgn] * N
            I += list(r + n[1:] - 1); Jc += list(base + n[1:] - 2); Vv += [-sgn] * (N - 1)
            c = np.zeros(N); c[0] = q0[j]; bub += list(Vm * DT + sgn * c); r += N
        for sgn in (1.0, -1.0):
            I += list(r + n - 1); Jc += list(base + n - 1); Vv += [sgn] * N
            I += list(r + n[1:] - 1); Jc += list(base + n[1:] - 2); Vv += [-2 * sgn] * (N - 1)
            I += list(r + n[2:] - 1); Jc += list(base + n[2:] - 3); Vv += [sgn] * (N - 2)
            c = np.zeros(N); c[0] = -2.0 * q0[j] + (q0[j] - qd0[j] * DT); c[1] = q0[j]; bub += list(A * DT * DT - sgn * c); r += N
    for (na, nb, qref, dvec) in corridors:
        d = np.asarray(dvec, float).copy(); d[3] = 0.0; d /= np.linalg.norm(d)
        B = np.linalg.svd(np.eye(6) - np.outer(d, d))[0][:, :4]             # d 와 관절 4 축에 수직인 4 방향
        B = B[:, [i for i in range(B.shape[1]) if abs(B[3, i]) < 0.5]][:, :4]
        for nn in range(na + 2, nb, 4):                                     # 8 ms 간격 표본
            qr_ = qref(nn); rows = []
            for j in range(6):
                h_all = np.r_[Sh[-L - 2:-1, j], q0[j]]; cols = []; vals = []; cst = 0.0
                for kk in range(L):
                    m = nn - kk
                    if m >= 1:
                        cols.append(j * N + m - 1); vals.append(hk[kk])
                    else:
                        cst += hk[kk] * h_all[len(h_all) - 1 + m]
                rows.append((cols, vals, cst))
            for b in B.T:
                for sgn in (1.0, -1.0):
                    cst_tot = 0.0
                    for j in range(6):
                        if abs(b[j]) < 1e-12:
                            continue
                        cols, vals, cst = rows[j]
                        I += [r] * len(cols); Jc += cols; Vv += [sgn * b[j] * v for v in vals]; cst_tot += b[j] * (cst - qr_[j])
                    bub.append(BAND - sgn * cst_tot); r += 1
    for zc_ in zcons:                                                         # (표본, 기준 자세, 하한[, 축 번호 — 0 = x, 기본 2 = z])
        nn, qref, zmin = zc_[:3]; ax_ = int(zc_[3]) if len(zc_) > 3 else 2
        qref = np.asarray(qref, float); Jz = np.array(F_JT(qref))[ax_]; z0 = float(np.array(F_TCP(qref)).ravel()[ax_])
        cst_tot = 0.0
        for j in range(6):
            if abs(Jz[j]) < 1e-12:
                continue
            h_all = np.r_[Sh[-L - 2:-1, j], q0[j]]
            for kk in range(L):
                m = nn - kk
                if m >= 1:
                    I.append(r); Jc.append(j * N + m - 1); Vv.append(-Jz[j] * hk[kk])
                else:
                    cst_tot += Jz[j] * hk[kk] * h_all[len(h_all) - 1 + m]
        bub.append(-(zmin - z0 + float(Jz @ qref)) + cst_tot); r += 1                  # -Jz.q_n <= -(zmin - z0 + Jz.qref)
    if slp:                                                                   # 순서 제약 (δ 사이) + 첫 점 ≥ gap
        d0 = 6 * N + ns; nsl = list(slp["ns"]); gp = list(slp["gaps"])
        I.append(r); Jc.append(d0); Vv.append(-1.0); bub.append((nsl[0] - gp[0]) * DT); r += 1
        for i in range(len(events) - 1):
            I += [r, r]; Jc += [d0 + i, d0 + i + 1]; Vv += [1.0, -1.0]; bub.append((nsl[i + 1] - nsl[i] - gp[i + 1]) * DT); r += 1
    nv0 = nv; cobj = np.zeros(nv); _free = []
    if slp:
        cobj[6 * N + ns + int(slp.get("obj", 3))] = 1.0                         # 목적: 종점 (다음 hover / 준비점) 시각 이동 δ 최소
        if slp.get("w_other"):                                                 # (선택) 다른 사건도 가장 이르게 — 작은 가중
            for i_ in range(nd):
                if i_ != int(slp.get("obj", 3)):
                    cobj[6 * N + ns + i_] = float(slp["w_other"])
    if smooth:                                                                 # 보조 변수 a_jn ≥ |Δ²u_jn| (관절별 가속 한계로 정규화), 목적 = Σ a
        cs = 1.0 if isinstance(smooth, bool) else float(smooth)                # 가중치 (bool = 1): slp 와 같이 쓰면 목적 = δ_끝 + cs·Σa (시간 + 토크 대용 가중합)
        nv = nv0 + 6 * N; cobj = np.r_[cobj[:nv0], cs * np.ones(6 * N)]
        for j in range(6):
            base = j * N; sc = 1.0 / (V.QDD[j] * DT * DT)
            for sgn in (1.0, -1.0):
                I += list(r + n - 1); Jc += list(base + n - 1); Vv += [sgn * sc] * N
                I += list(r + n[1:] - 1); Jc += list(base + n[1:] - 2); Vv += [-2 * sgn * sc] * (N - 1)
                I += list(r + n[2:] - 1); Jc += list(base + n[2:] - 3); Vv += [sgn * sc] * (N - 2)
                I += list(r + n - 1); Jc += list(nv0 + base + n - 1); Vv += [-1.0] * N
                c = np.zeros(N); c[0] = -2.0 * q0[j] + (q0[j] - qd0[j] * DT); c[1] = q0[j]; bub += list(-sgn * sc * c); r += N
    if win:                                                                   # 릴리즈 창 항 (11-10 사용자): Σ_m w_m (W_ACC·착지오차² + W_OMEGA·ω²) — 제곱은 접선 묶음 (s ≥ ±2a·x − a²)
        if slp:
            cobj[6 * N + ns + int(slp.get("obj", 3))] = float(win.get("w1", 100.0))  # 시간 항 가중 W1 (창 항과 같은 단위)
        tg = np.asarray(win["tgt"], float); d_rel = (6 * N + ns + int(win["irel"])) if slp else None
        A_E = (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32); A_W = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
        for (nm, wm, qb, qdb, de, dw) in win["nodes"]:
            qb = np.asarray(qb, float); qdb = np.asarray(qdb, float)
            e0 = np.array(F_LVEC(qb, qdb, tg)).ravel(); om0 = np.array(F_W(qb, qdb)).ravel(); Eq = np.zeros((5, 6)); Ed = np.zeros((5, 6)); eps = 1e-6
            x0 = np.r_[e0, om0]
            for j in range(6):
                dq = np.zeros(6); dq[j] = eps
                Eq[:, j] = (np.r_[np.array(F_LVEC(qb + dq, qdb, tg)).ravel(), np.array(F_W(qb + dq, qdb)).ravel()] - x0) / eps
                Ed[:, j] = (np.r_[np.array(F_LVEC(qb, qdb + dq, tg)).ravel(), np.array(F_W(qb, qdb + dq)).ravel()] - x0) / eps
            RC = []
            for j in range(6):
                h_all = np.r_[Sh[-L - 2:-1, j], q0[j]]; out = []
                for nn in (nm - 1, nm, nm + 1):
                    cols, vals, cst = [], [], 0.0
                    for k in range(L):
                        m = nn - k
                        if m >= 1:
                            cols.append(j * N + m - 1); vals.append(hk[k])
                        else:
                            cst += hk[k] * h_all[len(h_all) - 1 + m]
                    out.append((np.array(cols, int), np.array(vals, float), cst))
                RC.append(out)
            for c in range(5):                                                # 성분 x_c = x0_c + Eq·(q − q̄) + Ed·(q̇ − q̄̇) [+ 변화율·δ_릴리즈]
                acc = {}; const = x0[c] - Eq[c] @ qb - Ed[c] @ qdb
                for j in range(6):
                    (c0, v0, k0), (c1, v1, k1), (c2, v2, k2) = RC[j]
                    for cc, vv in zip(c1, v1 * Eq[c, j]):
                        acc[cc] = acc.get(cc, 0.0) + vv
                    for cc, vv in zip(c2, v2 * Ed[c, j] / (2 * DT)):
                        acc[cc] = acc.get(cc, 0.0) + vv
                    for cc, vv in zip(c0, -v0 * Ed[c, j] / (2 * DT)):
                        acc[cc] = acc.get(cc, 0.0) + vv
                    const += Eq[c, j] * k1 + Ed[c, j] * (k2 - k0) / (2 * DT)
                if d_rel is not None:
                    dcoef = (de[c] if (c < 2 and de is not None) else (dw[c - 2] if (c >= 2 and dw is not None) else 0.0))
                    if dcoef:
                        acc[d_rel] = acc.get(d_rel, 0.0) + float(dcoef)
                cols_x = list(acc.keys()); vals_x = np.array(list(acc.values()), float)
                x_col = nv; s_col = nv + 1; nv += 2; _free.append(x_col); cobj = np.r_[cobj, 0.0, float(wm) * float(win["wacc"] if c < 2 else win["wom"])]
                for sgn in (1.0, -1.0):                                       # x = (선형식) + const — 부등식 두 줄
                    I += [r] * (len(cols_x) + 1); Jc += cols_x + [x_col]; Vv += list(sgn * vals_x) + [-sgn]; bub.append(-sgn * const); r += 1
                for a_ in (A_E if c < 2 else A_W):
                    for sgn in (1.0, -1.0):                                   # sgn·2a·x − s ≤ a²
                        I += [r, r]; Jc += [x_col, s_col]; Vv += [sgn * 2 * a_, -1.0]; bub.append(a_ * a_); r += 1
    _wtr = [dict(n=nm_, trust=win["trust"], qref=qb_w, qdref=qdb_w) for (nm_, _w, qb_w, qdb_w, _de, _dw) in win["nodes"]] if (win and win.get("trust")) else []
    for e in list(events) + _wtr:                                             # 선형화 신뢰 구간 (비선형 사건·창 노드, (ρq, ρv)): |q[n] − q̄| ≤ ρq, |q̇[n] − q̄̇| ≤ ρv
        tr_ = e.get("trust")                                                  # — 순차 선형화가 한 번에 멀리 뛰어 진동하는 것을 막는 풀이 장치 (수렴하면 구속하지 않는다)
        if not tr_:
            continue
        qb_ = np.asarray(e["qref"], float); qdb_ = np.asarray(e["qdref"], float)
        for j in range(6):
            h_all = np.r_[Sh[-L - 2:-1, j], q0[j]]

            def rowc(nn):
                cols, vals, cst = [], [], 0.0
                for k in range(L):
                    m = nn - k
                    if m >= 1:
                        cols.append(j * N + m - 1); vals.append(hk[k])
                    else:
                        cst += hk[k] * h_all[len(h_all) - 1 + m]
                return cols, vals, cst
            c1, v1, k1 = rowc(e["n"]); c0, v0, k0 = rowc(e["n"] - 1); c2, v2, k2 = rowc(e["n"] + 1)
            for sgn in (1.0, -1.0):
                I += [r] * len(c1); Jc += c1; Vv += [sgn * x for x in v1]; bub.append(float(tr_[0]) + sgn * (qb_[j] - k1)); r += 1
                I += [r] * (len(c2) + len(c0)); Jc += c2 + c0; Vv += [sgn * x / (2 * DT) for x in v2] + [-sgn * x / (2 * DT) for x in v0]
                bub.append(float(tr_[1]) + sgn * (qdb_[j] - (k2 - k0) / (2 * DT))); r += 1
    Aub = sp.csr_matrix((Vv, (I, Jc)), shape=(r, nv))
    _hn = None                                                                # hold_end: 앞 hold 사건 시각부터 고정 (11-10 A1 진입점 정지)
    for e in events:
        if e.get("hold"):
            _hn = e["n"]
        if e.get("hold_end"):
            e["hold_from"] = _hn
    nrow = sum(12 if e.get("cmd") else 3 if e.get("relv") else 2 if e.get("land") else 11 if e.get("tcp") else ((6 + len(e.get("jxy", ()))) if e.get("free") else (6 * (L + 1) if e.get("hold") else
               (6 * max(0, e["n"] - e["hold_from"] - 1) if e.get("hold_end") else (18 if e.get("acc0") else 12)))) for e in events)
    Aeq = np.zeros((nrow, nv)); beq = np.zeros(nrow); row = 0; si = 6 * N
    for ei, e in enumerate(events):
        di = 6 * N + ns + ei
        if e.get("cmd"):                                                      # 명령 사건 (11-11): 명령 자체가 n 에서 pos, n-1 에서 pos1 (표준 수직 하강 직선의 마지막 두 표본)
            for j in range(6):
                Aeq[row, j * N + e["n"] - 1] = 1.0; beq[row] = e["pos"][j]
                if slp and e.get("dpos") is not None:
                    Aeq[row, di] = e["dpos"][j]
                row += 1
                Aeq[row, j * N + e["n"] - 2] = 1.0; beq[row] = e["pos1"][j]
                if slp and e.get("dpos1") is not None:
                    Aeq[row, di] = e["dpos1"][j]
                row += 1
            continue
        if e.get("hold"):                                                     # 정지 사건: 명령 자체를 q 에 고정 (필터 기억 길이만큼) → 실제가 n 에서 정지·유지
            for j in range(6):
                for m in range(e["n"] - L + 1, e["n"] + 2):
                    if 1 <= m <= N:
                        Aeq[row, j * N + m - 1] = 1.0; beq[row] = e["pos"][j]; row += 1
            continue
        if e.get("hold_end"):                                                 # 정지 유지: hold 다음 샘플부터 이 시각까지 명령 고정 (출발 = 이 시각 뒤)
            for j in range(6):
                for m in range(e["hold_from"] + 2, e["n"] + 1):
                    if 1 <= m <= N:
                        Aeq[row, j * N + m - 1] = 1.0; beq[row] = e["pos"][j]; row += 1
            continue
        if e.get("land") or e.get("relv"):                                    # 릴리즈 자유 (11-10): relv = 발사 속도 3 행 (위치·각속도 자유), land = 착지점 2 행만
            def realj2(j, nn):
                h_all = np.r_[Sh[-L - 2:-1, j], q0[j]]; rw = np.zeros(nv); cst = 0.0
                for k in range(L):
                    m = nn - k
                    if m >= 1:
                        rw[j * N + m - 1] += hk[k]
                    else:
                        cst += hk[k] * h_all[len(h_all) - 1 + m]
                return rw, cst
            R1 = [realj2(j, e["n"]) for j in range(6)]; R0 = [realj2(j, e["n"] - 1) for j in range(6)]; R2 = [realj2(j, e["n"] + 1) for j in range(6)]
            qb = np.asarray(e["qref"], float); qdb = np.asarray(e["qdref"], float); tg = np.asarray(e["tgt"], float)
            Fres = (lambda a_, b_: relv_res(a_, b_, tg)) if e.get("relv") else (lambda a_, b_: np.array(F_LVEC(a_, b_, tg)).ravel()); nres = 3 if e.get("relv") else 2
            e0 = Fres(qb, qdb); Eq = np.zeros((nres, 6)); Ed = np.zeros((nres, 6)); eps = 1e-6
            for j in range(6):
                dq = np.zeros(6); dq[j] = eps
                Eq[:, j] = (Fres(qb + dq, qdb) - e0) / eps; Ed[:, j] = (Fres(qb, qdb + dq) - e0) / eps
            for i in range(nres):
                Aeq[row] = sum(Eq[i, j] * R1[j][0] + Ed[i, j] * (R2[j][0] - R0[j][0]) / (2 * DT) for j in range(6))
                beq[row] = -e0[i] + Eq[i] @ qb + Ed[i] @ qdb - sum(Eq[i, j] * R1[j][1] + Ed[i, j] * (R2[j][1] - R0[j][1]) / (2 * DT) for j in range(6))
                if slp and e.get("tdland") is not None:
                    Aeq[row, di] = e["tdland"][i]
                row += 1
            continue
        if e.get("tcp"):                                                      # TCP 사건 (실제 궤적 기준, 기준 자세 주위 선형화) — 11-10
            def realj(j, nn):
                h_all = np.r_[Sh[-L - 2:-1, j], q0[j]]; rw = np.zeros(nv); cst = 0.0
                for k in range(L):
                    m = nn - k
                    if m >= 1:
                        rw[j * N + m - 1] += hk[k]
                    else:
                        cst += hk[k] * h_all[len(h_all) - 1 + m]
                return rw, cst
            R1 = [realj(j, e["n"]) for j in range(6)]; R0 = [realj(j, e["n"] - 1) for j in range(6)]; R2 = [realj(j, e["n"] + 1) for j in range(6)]
            qb = np.asarray(e["qref"], float); qdb = np.asarray(e["qdref"], float)
            Jt = np.array(F_JT(qb)); p_b = np.array(F_TCP(qb)).ravel(); a_b = np.array(F_AX(qb)).ravel(); tw0 = Jt @ qdb
            Ja = np.zeros((2, 6)); G = np.zeros((6, 6)); eps = 1e-6
            for j in range(6):
                dq = np.zeros(6); dq[j] = eps
                Ja[:, j] = (np.array(F_AX(qb + dq)).ravel()[:2] - a_b[:2]) / eps
                G[:, j] = (np.array(F_JT(qb + dq)) @ qdb - tw0) / eps
            for i in range(3):                                                # 위치
                Aeq[row] = sum(Jt[i, j] * R1[j][0] for j in range(6)); beq[row] = e["p"][i] - p_b[i] + Jt[i] @ qb - sum(Jt[i, j] * R1[j][1] for j in range(6))
                if slp and e.get("tdpos") is not None:
                    Aeq[row, di] = e["tdpos"][i]
                row += 1
            for i in range(2):                                                # 툴 축 수직
                Aeq[row] = sum(Ja[i, j] * R1[j][0] for j in range(6)); beq[row] = -a_b[i] + Ja[i] @ qb - sum(Ja[i, j] * R1[j][1] for j in range(6))
                if slp and e.get("tdpos") is not None:
                    Aeq[row, di] = e["tdpos"][3 + i]
                row += 1
            vt = np.r_[np.asarray(e["v"], float), 0.0, 0.0, 0.0]
            for i in range(6):                                                # 선속도·각속도 (중앙 차분)
                rw = sum(Jt[i, j] * (R2[j][0] - R0[j][0]) / (2 * DT) + G[i, j] * R1[j][0] for j in range(6))
                cst = sum(Jt[i, j] * (R2[j][1] - R0[j][1]) / (2 * DT) + G[i, j] * R1[j][1] for j in range(6))
                if i == 2 and e.get("vd") is not None:
                    rw[si] -= 1.0                                             # 수직 속도 자유 (ż 변수)
                Aeq[row] = rw; beq[row] = vt[i] + G[i] @ qb - cst
                if slp and e.get("tdvel") is not None:
                    Aeq[row, di] = e["tdvel"][i]
                row += 1
            if e.get("vd") is not None:
                si += 1
            continue
        for j in range(6):
            h_all = np.r_[Sh[-L - 2:-1, j], q0[j]]

            def real(nn):
                rw = np.zeros(nv); cst = 0.0
                for k in range(L):
                    m = nn - k
                    if m >= 1:
                        rw[j * N + m - 1] += hk[k]
                    else:
                        cst += hk[k] * h_all[len(h_all) - 1 + m]
                return rw, cst
            r1, c1 = real(e["n"]); r0, c0 = real(e["n"] - 1); r2, c2 = real(e["n"] + 1)
            Aeq[row] = r1; beq[row] = e["pos"][j] - c1
            if slp and e.get("dpos") is not None:
                Aeq[row, di] = e["dpos"][j]
            row += 1
            if e.get("free"):                                                 # 위치 (속도는 아래 xy 행만)
                e.setdefault("_vr", {})[j] = ((r2 - r0) / (2 * DT), (c2 - c0) / (2 * DT))
                continue
            rv = (r2 - r0) / (2 * DT)                                         # 중앙 차분 속도
            if e.get("vd") is not None:
                rv[si] -= e["vd"][j]
            Aeq[row] = rv; beq[row] = e["vb"][j] - (c2 - c0) / (2 * DT)
            if slp and e.get("dvel") is not None:
                Aeq[row, di] = e["dvel"][j]
            row += 1
            if e.get("acc0"):
                Aeq[row] = (r2 - 2 * r1 + r0) / DT ** 2; beq[row] = -(c2 - 2 * c1 + c0) / DT ** 2; row += 1   # 실제 가속도 0 (선택)
        if e.get("free") and e.get("jxy") is not None:                         # 툴 xy 속도 = 벨트: Σ_j J[a, j]·(실제 관절 속도_j) = vb_a
            for a_, (Jrow, vtar) in enumerate(e["jxy"]):
                rw = np.zeros(nv); cst = 0.0
                for j in range(6):
                    rv, cv = e["_vr"][j]; rw += Jrow[j] * rv; cst += Jrow[j] * cv
                Aeq[row] = rw; beq[row] = vtar - cst; row += 1
        if e.get("vd") is not None:
            si += 1
    J4 = float(q0[3]); fix4 = os.environ.get("LPC_FIX4", "1") == "1"         # 11-09 사용자 "joint4 전체 고정 / 전체 free 비교": LPC_FIX4=1 고정, 0 자유
    QM = float(os.environ.get("LPC_QMARGIN", "0.02"))                           # 11-09: 관절 상자 안쪽 여유 (앱 게이트가 자리표시 뒤 정지 구간에서 한계 0.4° 초과를 거부)
    bnds = [((J4, J4) if (j == 3 and fix4) else (V.Q_LO[j] + QM, V.Q_HI[j] - QM)) for j in range(6) for _ in range(N)] + [(e["lo"], e["hi"]) for e in events if e.get("vd") is not None]
    if slp:
        bnds += [(-float(slp["tr"]), float(slp["tr"]))] * nd
    bnds += [(0.0, None)] * (nv - nv0)
    for c_ in _free:                                                          # 창 성분 변수 x 는 부호 자유
        bnds[c_] = (None, None)
    # --- 4 점 허용 띠 (LPC_BAND_POS / LPC_BAND_VEL): 등식 행에 여유 변수 s (|s| ≤ ε). 정지·가속 행은 등식 그대로. 여유 변수는 변수 벡터 맨 뒤.
    EP = float(os.environ.get("LPC_BAND_POS", "0")); EV = float(os.environ.get("LPC_BAND_VEL", "0"))
    Aeq_m = sp.csr_matrix(Aeq)
    if EP > 0 or EV > 0:
        rt = []
        for e in events:
            if e.get("cmd"):
                rt += ["h"] * 12
            elif e.get("hold"):
                rt += ["h"] * (6 * (L + 1))
            elif e.get("hold_end"):
                rt += ["h"] * (6 * max(0, e["n"] - e["hold_from"] - 1))
            elif e.get("free"):
                rt += ["h"] * (6 + len(e.get("jxy", ())))
            elif e.get("relv"):
                rt += ["tv"] * 3
            elif e.get("land"):
                rt += ["tl"] * 2
            elif e.get("tcp"):
                rt += ["tp"] * 3 + ["ta"] * 2 + ["tv"] * 3 + ["tw"] * 3
            else:
                rt += (["p", "v", "a"] if e.get("acc0") else ["p", "v"]) * 6
        tol = {"p": EP, "v": EV, "tp": float(os.environ.get("LPC_BAND_TP", "0.002")) if EP > 0 else 0.0, "ta": EP,
               "tv": float(os.environ.get("LPC_BAND_TV", "0.02")) if EV > 0 else 0.0, "tw": EV, "tl": float(os.environ.get("LPC_BAND_LAND", "0.01")) if EP > 0 else 0.0}
        idx = [i for i, t in enumerate(rt) if tol.get(t, 0.0) > 0]; nsl = len(idx)
        Aeq_m = sp.hstack([Aeq_m, sp.csr_matrix((-np.ones(nsl), (idx, np.arange(nsl))), shape=(Aeq.shape[0], nsl))]).tocsr()
        Aub = sp.hstack([Aub, sp.csr_matrix((Aub.shape[0], nsl))]).tocsr()
        bnds = list(bnds) + [(-tol[rt[i]], tol[rt[i]]) for i in idx]; cobj = np.r_[cobj, np.zeros(nsl)]
    nvt = len(cobj)
    KBS = int(os.environ.get("LPC_BS", "0"))                                    # B-spline 매듭 간격 (샘플); 0 = 샘플마다 변수. 차수 LPC_BS_DEG (기본 2)
    if KBS > 0:                                                               # 명령 u_j = B c_j — 제약은 2 ms 표본 그대로, 변수만 제어점
        Bm = bs_basis(N, KBS); M = Bm.shape[1]
        T = sp.block_diag([Bm] * 6 + [sp.identity(nvt - 6 * N, format="csr")], format="csr")
        Aub2 = (Aub @ T).tocsr(); Aeq2 = (Aeq_m @ T).tocsr()
        # 제어점 상자 (분할 단위성 → u 도 상자 안, 약간 보수적); 관절 4 고정은 제어점 전부 = q0[3]
        bnds2 = [((J4, J4) if (j == 3 and fix4) else (V.Q_LO[j] + QM, V.Q_HI[j] - QM)) for j in range(6) for _ in range(M)] + list(bnds[6 * N:])
        res = linprog(np.r_[np.zeros(6 * M), cobj[6 * N:]], A_ub=Aub2, b_ub=np.array(bub), A_eq=Aeq2, b_eq=beq, bounds=bnds2, method="highs", options={"time_limit": float(os.environ.get("LPC_TLIM", "20"))})
        if res.status != 0:
            return None
        Cc = res.x[:6 * M].reshape(6, M).T
        return (np.asarray(Bm @ Cc), res.x[6 * M:6 * M + nv0 - 6 * N])
    res = linprog(cobj, A_ub=Aub, b_ub=np.array(bub), A_eq=Aeq_m, b_eq=beq, bounds=bnds, method="highs", options={"time_limit": float(os.environ.get("LPC_TLIM", "20"))})
    return (res.x[:6 * N].reshape(6, N).T, res.x[6 * N:nv0]) if res.status == 0 else None


_BS_CACHE = {}


def bs_basis(N, K):
    """표본 1..N 에서 균일 3차 B-spline 기저 행렬 (N × M), 매듭 간격 K 샘플, 양끝 3 개 여분 매듭 (끝값 자유)."""
    dg = int(os.environ.get("LPC_BS_DEG", "2")); key = (N, K, dg)             # 11-09 사용자: 2 차 (C¹) 가 기본 — 가속이 매듭에서 뛸 수 있어 bang-bang 표현
    if key not in _BS_CACHE:
        from scipy.interpolate import BSpline
        m = -(-(N - 1) // K); t = (np.arange(-dg, m + dg + 1) * K + 1).astype(float)
        _BS_CACHE[key] = sp.csr_matrix(BSpline.design_matrix(np.arange(1, N + 1, dtype=float), t, dg))
    return _BS_CACHE[key]


Z_STEP = int(os.environ.get("LPC_ZSTEP", "4"))                                  # z 조건 표본 간격 (샘플)


def z_cons_from(q_start, ns, poses, floors, qref_traj=None):
    """구간별 z 하한: ns = 사건 시각 (샘플), poses = 사건 자세 (q_start 는 n=0 자세), floors[i] = 구간 (ns[i-1], ns[i]) 의 z 하한.
    기준 자세 = qref_traj[n] (재선형화) 아니면 자세들 사이 관절 공간 선형 보간."""
    out = []; pts = [0] + list(ns); qs = [np.asarray(q_start, float)] + [np.asarray(p, float) for p in poses]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        for nn in range(a + 1, b, Z_STEP):
            if qref_traj is not None and nn - 1 < len(qref_traj):
                qr = qref_traj[nn - 1]
            else:
                w = (nn - a) / max(b - a, 1); qr = (1 - w) * qs[i - 1] + w * qs[i]
            out.append((nn, qr, floors[i - 1]))
    return out


def realized_q(q0, Sh, u):
    """명령 u (n = 1..N, 시작 q0 제외) 의 실제 관절 궤적 (n = 1..N)."""
    S = np.vstack([Sh[:-1], q0[None], u]); Q = np.stack([np.convolve(S[:, j], hk)[:len(S)] for j in range(6)], -1)
    return Q[len(Sh):]


def slp_core(q0, qd0, Sh, mk, gaps, zc, ns0, sol0=None, tr0=None, verbose=False, ob=None):
    """순차 LP (11-09 사용자): 가능해 시각 ns0 에서 시작, 점 시각 이동 δ 를 변수로 넣은 1 차 전개 LP → 2 ms 반올림 → 정수 시각 확인 LP 를 반복.
    mk[i](n) 사건, gaps 최소 간격, zc(ns, evs, qref) z 하한. 리턴 (ns, (u, sc), 반복 수)."""
    NE = len(mk); OB = min(3, NE - 1) if ob is None else int(ob)             # 목적 사건: 다음 hover (5 점) / 준비점 정지 (4 점) / A1 대기 (7 점: 5)
    W_OTHER = float(os.environ.get("LPC_WAIT_LEX", "0")) * 1e-3

    def exact(ns, qr=None):
        evs = [mk[j](ns[j]) for j in range(NE)]
        return None if any(e is None for e in evs) else lp(q0, qd0, Sh, evs, ns[-1], zcons=zc(ns, evs, qr))
    ns = list(ns0); sol = sol0 if sol0 is not None else exact(ns)
    if sol is None:
        raise RuntimeError("SLP 시작 시각 불가")
    TR = float(tr0 if tr0 is not None else os.environ.get("LPC_SLP_TR", "0.02")); it = 0
    while it < int(os.environ.get("LPC_SLP_IT", "40")) and TR >= DT * 0.99:
        it += 1; u, sc = sol; qr = realized_q(q0, Sh, u); evs = [mk[j](ns[j]) for j in range(NE)]; zi = 0
        if any(e is None for e in evs):
            TR /= 2; continue
        for i, e in enumerate(evs):
            if e.get("hold") or e.get("hold_end"):
                continue
            ep, em = mk[i](ns[i] + 1), mk[i](ns[i] - 1)
            if ep is None or em is None:
                e["dpos"] = None; e["dvel"] = None; continue
            n = ns[i]; qd_ref = (qr[n] - qr[n - 2]) / (2 * DT); qdd_ref = (qr[n] - 2 * qr[n - 1] + qr[n - 2]) / DT ** 2
            kd = (ep["pos"] - em["pos"]) / (2 * DT); vt = (ep["vb"] - em["vb"]) / (2 * DT)
            if e.get("vd") is not None:
                vt = vt + float(sc[zi]) * (ep["vd"] - em["vd"]) / (2 * DT); zi += 1
            e["dpos"] = qd_ref - kd; e["dvel"] = qdd_ref - vt
        r = lp(q0, qd0, Sh, evs, ns[-1], zcons=zc(ns, evs, qr), slp=dict(tr=TR, gaps=gaps, ns=ns, obj=OB, w_other=W_OTHER))
        if r is None:
            TR /= 2; continue
        d = r[1][-NE:]; sh = [int(round(x / DT)) for x in d]
        if verbose:
            print(f"   [slp {it}] δ {np.round(np.array(d) * 1e3, 1).tolist()} ms → 이동 {sh} (TR {TR * 1e3:.0f} ms)", flush=True)
        if all(x == 0 for x in sh):
            if TR <= DT * 1.01:
                break
            TR /= 2; continue
        ns_new = [a + b for a, b in zip(ns, sh)]
        if ns_new[0] < gaps[0] or any(ns_new[i + 1] - ns_new[i] < gaps[i + 1] for i in range(NE - 1)) or ns_new[OB] > ns[OB]:
            TR /= 2; continue
        s2 = exact(ns_new)
        if s2 is None:
            TR /= 2; continue
        ns, sol = ns_new, s2
    for _it in range(int(os.environ.get("LPC_ZITER", "2"))):
        qr = realized_q(q0, Sh, sol[0]); r = exact(ns, qr)
        if r is None:
            break
        sol = r
    return ns, sol, it


def greedy_generic(q0, qd0, Sh, mk, gaps, zc):
    """국소 탐색 없는 탐욕 시각 배치 (사건 수 무관, 풀 없음 — DB 빌드 워커 안에서 씀). 리턴 (ns, (u, sc)) 또는 None."""
    NE = len(mk)

    def earliest(ns, i, n_from):
        evs = [mk[j](ns[j]) for j in range(len(ns))]; n = n_from
        while n < n_from + 400:
            e = mk[i](n)
            if e is None:
                n += 4; continue
            r = lp(q0, qd0, Sh, evs + [e], n, zcons=zc(ns + [n], evs + [e]))
            if r is not None:
                e2 = mk[i](n - 2)
                r2 = lp(q0, qd0, Sh, evs + [e2], n - 2, zcons=zc(ns + [n - 2], evs + [e2])) if (e2 is not None and n - 2 > (ns[-1] if ns else 0)) else None
                return (n - 2, r2) if r2 is not None else (n, r)
            n += 4
        return None
    ns = []; sol = None; i = 0; bumps = 0; n_start = NLP[0]; JOBMAX = int(os.environ.get("LPC_JOB_MAX", "100000"))
    while i < NE:
        if NLP[0] - n_start > JOBMAX:                                          # LP 예산 초과 → 불가 (DB 빌드에서 한 건이 수천 LP 먹는 것 방지)
            return None
        got = earliest(ns, i, (ns[-1] if ns else 0) + gaps[i])
        if got is not None:
            ns.append(got[0]); sol = got[1]; i += 1; continue
        if i == 0 or bumps >= 40:
            return None
        bumps += 1; prev = ns.pop(); i -= 1
        got = earliest(ns, i, prev + 8)
        if got is None:
            return None
        ns.append(got[0]); sol = got[1]; i += 1
    return ns, sol


Q_READY_TCP = np.array([float(v) for v in os.environ.get("GP8_READY_TCP", "0.45,0.35,0.12").split(",")])


def ev_ready(n):
    """준비점 정지 사건 (11-09 사용자 "준비점 종점"): 명령을 준비점 자세에 고정 → 실제가 n 에서 정지, 그 뒤 유지."""
    q = ik(Q_READY_TCP); return dict(n=n, pos=q, hold=True, vb=None, vd=None)


def slp_start(q0, qd0, Sh, mk, zc, ns_prev, verbose=False):
    """앞 사이클 시각을 ×1.0, 1.1, … 로 늘려 가며 가능해 시작점 (LP 1 번씩). 리턴 (ns0, sol0, 배율) 또는 None."""
    NE = len(mk)
    for f in (1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.8, 2.1, 2.5, 3.0):                 # 11-10: 멀리 있는 다음 물체 (대기 포함) 까지
        cand = [int(round(n * f / 2) * 2) for n in ns_prev]
        evs = [mk[j](cand[j]) for j in range(NE)]
        if any(e is None for e in evs):
            continue
        r = lp(q0, qd0, Sh, evs, cand[-1], zcons=zc(cand, evs))
        if verbose:
            print(f"   시작점: 앞 사이클 시각 × {f:.2f} = {cand} → {'가능' if r is not None else '불가'}", flush=True)
        if r is not None:
            return cand, r, f
    return None


def ev_contact(k, t0, n):
    q = _ik_safe(np.r_[at(k, t0 + n * DT)[:2], ZI])
    return None if q is None else dict(n=n, pos=q, vb=qd_of(q, VB), vd=None)


def ev_h(k, t0, n, up):
    """11-09 사용자: 위치·자세 + 툴 xy 속도 = 컨베이어 + 회전 속도 0, 수직 속도만 자유 (범위 제한 없음).
    관절 속도 = qd_of(q, [VBx, VBy, 0]) + ż·VZ(q) (ż = LP 변수)."""
    q = _ik_safe(np.r_[at(k, t0 + n * DT)[:2], H])
    return None if q is None else dict(n=n, pos=q, vb=qd_of(q, np.r_[VB[:2], 0.0]), vd=VZ(q), lo=None, hi=None)


def ev_rel(b, n):
    q, qd, _, _ = REL[b]; return dict(n=n, pos=q, vb=qd, vd=None)


_G = {}


def _greedy_job(fx):
    return _G["greedy"](fx)


def plan(q0, qd0, Sh, k, t0, first):
    """first: 이 사이클이 hover 에서 시작 (True) — 첫 점은 접촉. 점 순서: 접촉 k, 상승끝 k, 릴리즈 k, hover k+1, 접촉 k+1 (내다봄)."""
    mk = [lambda n: ev_contact(k, t0, n), lambda n: ev_h(k, t0, n, True), lambda n: ev_rel(BIN[k], n),
          lambda n: ev_h(k + 1, t0, n, False), lambda n: ev_contact(k + 1, t0, n)]
    gaps = [20, 16, 30, 30, 16]

    def corr(ns):
        if os.environ.get("LPC_CORR", "0") != "1":                            # 11-09 사용자: 통로 조건 빼
            return []
        """하강·상승 통로: 물체 위 수직선 관절 경로 qref(n) = IK(물체(t_n), 윗면) + VZ·(높이), 방향 = VZ."""
        out = []

        def mkc(kk, na, nb):
            """기준선 = 접촉 자세와 h 자세를 잇는 관절 공간 현 (두 끝 자세가 정확히 위에 있음), 벨트 따라 이동."""
            qc0 = ik(np.r_[at(kk, t0 + na * DT)[:2], ZI]); qh0 = ik(np.r_[at(kk, t0 + na * DT)[:2], H]); qb = qd_of(qc0, VB)
            return (na, nb, lambda nn, qc0=qc0, qb=qb, na=na: qc0 + qb * (nn - na) * DT, qh0 - qc0)
        if len(ns) >= 1 and first:
            pass
        if len(ns) >= 2:
            out.append(mkc(k, ns[0], ns[1]))                                  # 상승 k
        if len(ns) >= 5:
            out.append(mkc(k + 1, ns[3], ns[4]))                              # 하강 k+1
        if len(ns) >= 1:
            out.append(mkc(k, max(1, ns[0] - 40), ns[0]))                     # 하강 k 끝 0.08 s (앞 사이클이 정한 hover 뒤)
        return out

    FLOORS = [ZI, ZI, H, H, ZI]                                               # 하강 k, 상승 k, 아크, 복귀, 다음 하강
    q_start = Sh[-1]

    def zc(ns, evs, qref_traj=None):
        return z_cons_from(q_start, ns, [e["pos"] for e in evs], FLOORS[:len(ns)], qref_traj)

    def earliest(ns, i, n_from):
        """앞 시각 ns 고정, 점 i 의 가장 이른 가능 시각 (n_from 부터 4 ms 훑고 2 ms 다듬기). 리턴 (n, 해) 또는 None."""
        evs = [mk[j](ns[j]) for j in range(len(ns))]; n = n_from
        while n < n_from + 400:
            e = mk[i](n); r = lp(q0, qd0, Sh, evs + [e], n, corr(ns + [n]), zcons=zc(ns + [n], evs + [e]))
            if r is not None:
                e2 = mk[i](n - 2); r2 = lp(q0, qd0, Sh, evs + [e2], n - 2, corr(ns + [n - 2]), zcons=zc(ns + [n - 2], evs + [e2])) if n - 2 > (ns[-1] if ns else 0) else None
                return (n - 2, r2) if r2 is not None else (n, r)
            n += 4
        return None

    def greedy(fixed):
        """fixed: 앞 점 시각. 나머지를 앞에서부터 가장 이른 시각으로; 뒤 점이 안 되면 바로 앞 점을 8 ms 씩 늦춰 다시 (되돌림, 최대 40 번)."""
        ns = list(fixed); sol = None; n_start = NLP[0]; JOBMAX = int(os.environ.get("LPC_JOB_MAX", "100000"))
        if ns:
            evs0 = [mk[j](ns[j]) for j in range(len(ns))]
            sol = lp(q0, qd0, Sh, evs0, ns[-1], corr(ns), zcons=zc(ns, evs0))
            if sol is None:
                return None
        nfix = len(ns); i = len(ns); bumps = 0
        while i < 5:
            if NLP[0] - n_start > JOBMAX:                                      # 탐색 예산 (B-spline 비교용: 한 배치가 수천 LP 를 먹는 것 방지)
                return None
            got = earliest(ns, i, (ns[-1] if ns else 0) + gaps[i])
            if got is not None:
                ns.append(got[0]); sol = got[1]; i += 1; continue
            if i <= nfix or bumps >= 40:                                       # 고정된 점까지 되돌아가야 하면 불가
                if os.environ.get("LPC_DBG"):
                    print(f"   [greedy] 점 {i} 불가, 앞 시각 {ns}", flush=True)
                return None
            bumps += 1; prev = ns.pop(); i -= 1                                # 앞 점을 늦춤
            got = earliest(ns, i, prev + 8)
            if got is None:
                return None
            ns.append(got[0]); sol = got[1]; i += 1
        return ns, sol

    import multiprocessing as mp
    best = greedy([])
    if best is None:
        raise RuntimeError(f"사이클 {k + 1} 불가")
    g0 = best[0][3]; _G["greedy"] = greedy
    import contextlib
    _ls = os.environ.get("LPC_LS", "1") == "1"                                   # LPC_LS=0: 국소 탐색 없음 (풀 생성도 안 함 — DB 빌드 워커 안에서 호출)
    with (mp.get_context("fork").Pool(int(os.environ.get("LPC_W", "11"))) if _ls else contextlib.nullcontext()) as pool:   # 국소 탐색: 앞 점 시각을 옮긴 배치들을 병렬로
        while _ls:
            cands = []
            for i in range(3):
                for d in (4, 8, 16, 24, -2, -4):
                    fx = list(best[0][:i]) + [best[0][i] + d]
                    if fx[-1] <= (fx[-2] + 4 if i > 0 else 4):
                        continue
                    cands.append(fx)
            res = [g for g in pool.map(_greedy_job, cands) if g is not None]
            better = [g for g in res if g[0][3] < best[0][3]]
            if not better:
                break
            best = min(better, key=lambda g: g[0][3])
    ns = best[0]; evs = [mk[j](ns[j]) for j in range(5)]; sol = best[1]
    for _it in range(int(os.environ.get("LPC_ZITER", "2"))):                    # z 조건 재선형화: 해의 실제 관절 궤적 주위에서 다시
        qr = realized_q(q0, Sh, sol[0]); r = lp(q0, qd0, Sh, evs, ns[4], corr(ns), zcons=zc(ns, evs, qr))
        if r is None:
            break
        sol = r
    return ns, sol, g0                                                        # 11-09 사용자: 목적함수 없이 순수 time-optimal (가능해 그대로)


def plan_slp(q0, qd0, Sh, k, t0, ns0, u0=None, tr0=None, verbose=False):
    """순차 LP: 시작 시각 ns0 (가능해) 에서 점 시각 이동 δ 를 변수로 넣은 LP 를 반복. 각 반복 = 1 차 전개 LP 1 번 + 정수 시각 확인 LP 1 번.
    리턴 (ns, (u, sc), 반복 수)."""
    mk = [lambda n: ev_contact(k, t0, n), lambda n: ev_h(k, t0, n, True), lambda n: ev_rel(BIN[k], n),
          lambda n: ev_h(k + 1, t0, n, False), lambda n: ev_contact(k + 1, t0, n)]
    gaps = [20, 16, 30, 30, 16]; FLOORS = [ZI, ZI, H, H, ZI]; q_start = Sh[-1]

    def zc(ns, evs, qref_traj=None):
        return z_cons_from(q_start, ns, [e["pos"] for e in evs], FLOORS[:len(ns)], qref_traj)

    return slp_core(q0, qd0, Sh, mk, gaps, zc, ns0, u0, tr0, verbose)
    TR = 0.0; it = 0; hist = []; exact = None; ns = list(ns0); sol = u0                   # (아래는 더 이상 쓰이지 않음)
    while it < int(os.environ.get("LPC_SLP_IT", "40")) and TR >= DT * 0.99:
        it += 1; u, sc = sol; qr = realized_q(q0, Sh, u); evs = [mk[j](ns[j]) for j in range(5)]; zi = 0
        for i, e in enumerate(evs):
            n = ns[i]; qd_ref = (qr[n] - qr[n - 2]) / (2 * DT); qdd_ref = (qr[n] - 2 * qr[n - 1] + qr[n - 2]) / DT ** 2
            kd = (ep["pos"] - em["pos"]) / (2 * DT); vt = (ep["vb"] - em["vb"]) / (2 * DT)
            if e.get("vd") is not None:
                vt = vt + float(sc[zi]) * (ep["vd"] - em["vd"]) / (2 * DT); zi += 1
            e["dpos"] = qd_ref - kd; e["dvel"] = qdd_ref - vt
        r = lp(q0, qd0, Sh, evs, ns[4], zcons=zc(ns, evs, qr), slp=dict(tr=TR, gaps=gaps, ns=ns))
        if r is None:
            TR /= 2; hist.append(("불가", TR)); continue
        d = r[1][-5:]; sh = [int(round(x / DT)) for x in d]
        if verbose:
            print(f"   [slp {it}] δ {np.round(np.array(d) * 1e3, 1).tolist()} ms → 이동 {sh} (TR {TR * 1e3:.0f} ms)", flush=True)
        if all(x == 0 for x in sh):
            if TR <= DT * 1.01:
                break
            TR /= 2; continue                                                    # 선형 모델이 움직일 수 없다 → 반경 줄여 2 ms 수준 확인
        ns_new = [a + b for a, b in zip(ns, sh)]
        if ns_new[0] < gaps[0] or any(ns_new[i + 1] - ns_new[i] < gaps[i + 1] for i in range(4)) or ns_new[3] > ns[3]:
            TR /= 2; continue
        s2 = exact(ns_new)
        if s2 is None:
            TR /= 2; hist.append(("정수 불가", ns_new)); continue
        if ns_new[3] == ns[3] and ns_new == ns:
            break
        ns, sol = ns_new, s2
    evs = [mk[j](ns[j]) for j in range(5)]
    for _it in range(int(os.environ.get("LPC_ZITER", "2"))):
        qr = realized_q(q0, Sh, sol[0]); r = exact(ns, qr)
        if r is None:
            break
        sol = r
    return ns, (sol[0], sol[1][:len(sol[1])]), it


NC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NH = 300
S = np.array([ik(np.r_[at(0, (j - NH) * DT + LAG)[:2], H]) for j in range(NH + 1)])   # 시작: 물체 1 위 명령 hover (지금 계획기 검사와 같음)
t = 0.0; log = []; t_start = time.time()
for k in range(NC):
    q0 = S[-1]; qd0 = (S[-1] - S[-2]) / DT; tt0 = time.time(); NLP[0] = 0
    ns, (u, sc), g0 = plan(q0, qd0, S, k, t, k == 0)
    nh = ns[3]; S = np.vstack([S, u[:nh]]); last = (ns, u)
    log.append(dict(k=k, t0=t, ns=ns, sc=sc))
    print(f"사이클 {k + 1}: 접촉 +{ns[0] * DT:.3f} 상승끝 +{ns[1] * DT:.3f} 릴리즈 +{ns[2] * DT:.3f} hover +{ns[3] * DT:.3f} "
          f"다음접촉 +{ns[4] * DT:.3f} | 사이클 {nh * DT:.3f} s (앞에서부터 이른 시각만이면 {g0 * DT:.3f}) | LP {NLP[0]} 번 {time.time() - tt0:.0f}s", flush=True)
    t += nh * DT
ns_l, u_l = last                                                              # 판정용: 마지막 사이클이 내다본 명령 (hover → 다음 접촉) 을 그대로 이어 붙임
S = np.vstack([S, u_l[ns_l[3]:ns_l[4] + 2], np.repeat(u_l[ns_l[4] + 1:ns_l[4] + 2], L, 0)])
Qr = np.stack([np.convolve(S[:, j], hk)[:len(S)] for j in range(6)], -1); Qdr = np.gradient(Qr, DT, axis=0)
P = tcp_batch(Qr * SG); n0 = NH; tt = (np.arange(len(S)) - n0) * DT; Vt = np.gradient(P, DT, axis=0)
print("\n=== 실행기 (지연 0.04 + 평균 0.2·0.08) 실제 궤적 판정 ===")
for c in log:
    k = c["k"]; t0 = c["t0"]; ns = c["ns"]; j_k = n0 + int(round(t0 / DT)) + ns[0]; j_l = n0 + int(round(t0 / DT)) + ns[1]
    j_r = n0 + int(round(t0 / DT)) + ns[2]; j_h = n0 + int(round(t0 / DT)) + ns[3]
    e = P[j_k] - at(k, tt[j_k]); vr = Vt[j_k] - VB; e2, _ = F_LAND(Qr[j_r], Qdr[j_r], np.array(BINS[BIN[k]], float))
    el = P[j_l] - np.r_[at(k, tt[j_l])[:2], H]; eh = P[j_h] - np.r_[at(k + 1, tt[j_h])[:2], H]
    lift = np.linalg.norm(P[j_k:j_l + 1, :2] - np.stack([at(k, x)[:2] for x in tt[j_k:j_l + 1]]), axis=1).max()
    seg = slice(n0 + int(round(t0 / DT)), j_h + 1)
    box = bool(np.all(Qr[seg] >= V.Q_LO - 1e-6) and np.all(Qr[seg] <= V.Q_HI + 1e-6)); qd = float(np.max(np.abs(Qdr[seg]) / V.QD))
    lands = [100 * np.sqrt(float(F_LAND(Qr[j_r + d], Qdr[j_r + d], np.array(BINS[BIN[k]], float))[0])) for d in (-1, 1)]
    print(f"사이클 {k + 1}: 릴리즈 ±2 ms 착지 {lands[0]:.2f}/{lands[1]:.2f} cm | 접촉 전후 2 ms 수직 속도 {Vt[j_k - 1, 2]:+.3f}/{Vt[j_k + 1, 2]:+.3f} m/s")
    q4 = Qr[seg, 3]; print(f"사이클 {k + 1}: 관절4 실제 범위 [{np.degrees(q4.min()):+.1f}, {np.degrees(q4.max()):+.1f}]° (시작 {np.degrees(q4[0]):+.1f}°)")
    print(f"사이클 {k + 1}: 접촉 {1e3*np.linalg.norm(e):.2f} mm 상대속도 {1e3*np.linalg.norm(vr):.1f} mm/s | 상승끝 {1e3*np.linalg.norm(el):.2f} mm vz {Vt[j_l, 2]:+.2f} | "
          f"릴리즈 착지 {100*np.sqrt(float(e2)):.2f} cm | hover {1e3*np.linalg.norm(eh):.2f} mm vz {Vt[j_h, 2]:+.2f} | 상승 중 옆 {1e3*lift:.1f} mm | 상자 {box} 속도비 {qd:.2f} x최소 {P[seg, 0].min():.3f}")
cyc = [log[i + 1]["t0"] - log[i]["t0"] for i in range(NC - 1)] + [t - log[-1]["t0"]]
print(f"\n실제 4 점 LP 사이클 (hover→hover): {np.round(cyc, 3).tolist()} 평균 {np.mean(cyc):.3f} s ({60 / np.mean(cyc):.1f}/분) | 계산 {time.time() - t_start:.0f}s")
