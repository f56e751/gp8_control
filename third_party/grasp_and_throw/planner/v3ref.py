"""version3 참조 구현 (11-08): NLP 없이 사이클 = [하강, 접촉 창, 상승, 아크, 릴리즈 창, 복귀] 를 명령 공간 관절별 bang-bang 으로.
  릴리즈: 목표 방위각 수직면, 툴 수직, ω = 0, 농구식 최소 속력 각 θ* = 45° + ½·atan(h/R), v² = g(h + √(R²+h²)) (항력 0 닫힌 식).
  정확점 (접촉, 릴리즈) 앞 MEM = 0.28 s 는 명령 등속 (접촉: 벨트 추종 접촉 자세 + LAG 리드) → 실제가 정확히 지남.
  자유 변수: (ż_h, v_out, r_rel, z_rel) 격자 탐색. 평가: 2 ms 실제 궤적 (box 필터) 에서 접촉 xy·z·ż, 하강 띠, 착지 오차, x_min, 관절 상자.
사용 (robot10 ~/rge_real): python v3ref.py g52c_db.npz [칸 번호 ...]"""
import os, sys, time, json
os.environ.setdefault("GP8_CYC_NCTRL", "52")
import numpy as np
import casadi as cs
import cyc2 as C
import jcyc1 as J
from cyc2 import SG, DT, hk, LAG, H, ZI, BINS, VB
from gp8_control.throwing.nlp import throw_nlp as T
from gp8_control.grasping.workspace_gate import tcp_batch

MEM = len(hk) * DT
KT = np.arange(len(hk)) * DT; VAR = float(hk @ (KT - LAG) ** 2)                 # 커널 2차 모멘트: 등가속 명령 창 → 실제(t) = 명령(t−LAG) + ½ a VAR
WIN_MODE = os.environ.get("V3_WIN", "opt")                                      # opt: 창 없음 (실제 종단 조건으로 bang-bang)                                    # phase: 창 = 등가속 한 구간 (시간최적) | const: 창 = 등속
Q_LO = np.maximum(T.GP8_Q_MIN, T.Q_LO_REAL) + C.CY.Q_MARGIN; Q_HI = np.minimum(T.GP8_Q_MAX, T.Q_HI_REAL) - C.CY.Q_MARGIN
QD = np.asarray(T.QD_LIM, float); QDD = np.asarray(T.QDD_LIM, float)
_q = cs.MX.sym("q", 6); _qd = cs.MX.sym("qd", 6); _pts = T.fk_col_points_sym(_q); _tcp = _pts[3]; _wr = _pts[0]
_pl = _tcp + (T.GRIP_OFF / T._CHAIN[5][0][0]) * (_tcp - _wr)                   # 발사점 = landing_closed 와 같은 식
F_P = cs.Function("P", [_q], [_pl]); F_V = cs.Function("V", [_q, _qd], [cs.jtimes(_pl, _q, _qd)])
F_W = cs.Function("W", [_q, _qd], [T.fk_omega_sym(_q, _qd)]); F_TCP = cs.Function("tcp", [_q], [_tcp])
F_JM = cs.Function("Jm", [_q], [cs.jacobian(cs.vertcat(cs.jtimes(_pl, _q, _qd), T.fk_omega_sym(_q, _qd)), _qd)])
F_JT = cs.Function("Jt", [_q], [cs.jacobian(cs.vertcat(cs.jtimes(_tcp, _q, _qd), T.fk_omega_sym(_q, _qd)), _qd)])
F_LAND = J.landing_closed()
F_AX = cs.Function("ax", [_q], [T.fk_tool_axis_sym(_q)])
_REL = {}


def F_JREL(q):
    """릴리즈 뉴턴 자코비안 (발사점 e_r·z 성분, 툴 축 z) — 목표 방위각마다 한 번 생성."""
    global _psi_cur
    k = round(_psi_cur, 6)
    if k not in _REL:
        er = cs.DM([np.cos(_psi_cur), np.sin(_psi_cur), 0.0])
        _REL[k] = cs.Function("jrel", [_q], [cs.jacobian(cs.vertcat(cs.dot(_pl, er), _pl[2], T.fk_tool_axis_sym(_q)[2]), _q)])
    return _REL[k](q)


_psi_cur = 0.0
ik = J.ik_pl                                                                   # (xyz) → 수직 툴 관절 (계획 규약)


def qd_of(q, v, tcp=True):
    """TCP(또는 발사점) 속도 v (3) 와 ω = 0 을 주는 q̇ (6×6 역)."""
    Jm = np.array((F_JT if tcp else F_JM)(q)); return np.linalg.solve(Jm, np.r_[np.asarray(v, float), 0.0, 0.0, 0.0])


# ------------------------------------------------------------ 관절별 이중적분기 bang-bang (속도 한계, 양 끝 속도 임의)
def min_time(x0, v0, x1, v1, A, V):
    """최소 시간 프로파일 [(dt, a), ...] 또는 None."""
    best = None
    for s in (1.0, -1.0):
        vp2 = s * A * (x1 - x0) + 0.5 * (v0 ** 2 + v1 ** 2)
        if vp2 < 0:
            continue
        vp = s * np.sqrt(vp2)
        if abs(vp) > V:
            vp = s * V
        t1 = (vp - v0) / (s * A); t2 = (v1 - vp) / (-s * A)
        if t1 < -1e-12 or t2 < -1e-12:
            continue
        d = (vp ** 2 - v0 ** 2) / (2 * s * A) + (vp ** 2 - v1 ** 2) / (2 * s * A); rest = (x1 - x0) - d
        t3 = 0.0 if abs(vp) < 1e-12 else rest / vp
        if abs(vp) < 1e-12 and abs(rest) > 1e-9:
            continue
        if t3 < -1e-9:
            continue
        prof = [(max(t1, 0.0), s * A), (max(t3, 0.0), 0.0), (max(t2, 0.0), -s * A)]; tt = sum(p[0] for p in prof)
        if best is None or tt < best[0]:
            best = (tt, prof)
    return best


def _vp_of(prof_t, v0):
    """프로파일 (T, [(t1,a1), (t3,0), (t2,a2)]) 의 코스트 속도."""
    pr = prof_t[1]; return v0 + pr[0][1] * pr[0][0]


def fixed_time(x0, v0, x1, v1, A, V, Tf):
    """시간 Tf 에 맞춘 프로파일. 1) 가속도는 최대로 두고 중간 (코스트) 속도 v_p 를 2차식 근으로 (끝 속도가 커도 한계 밖으로 안 돎),
    2) 안 되면 가속도 크기를 줄여 (이분법). 못 맞추면 가장 가까운 것."""
    lo, hi = 1e-6 * A, A; r = min_time(x0, v0, x1, v1, hi, V)
    if r is None or r[0] > Tf + 1e-9:
        return r
    cand = []
    for sgn in (1.0, -1.0):                                                   # 2 v_p² − (2(v0+v1) + 2 s A Tf) v_p + (v0² + v1² + 2 s A Δx) = 0
        bq = -(2 * (v0 + v1) + 2 * sgn * A * Tf); cq = v0 ** 2 + v1 ** 2 + 2 * sgn * A * (x1 - x0); disc = bq * bq - 8 * cq
        if disc < 0:
            continue
        for vp in ((-bq + np.sqrt(disc)) / 4, (-bq - np.sqrt(disc)) / 4):
            if abs(vp) > V or abs(vp) < 1e-9:
                continue
            t1 = (vp - v0) / (sgn * A); t2 = (v1 - vp) / (-sgn * A)
            d = (vp ** 2 - v0 ** 2) / (2 * sgn * A) + (vp ** 2 - v1 ** 2) / (2 * sgn * A); t3 = ((x1 - x0) - d) / vp
            if min(t1, t2, t3) < -1e-9:
                continue
            cand.append((abs(vp), (t1 + t2 + t3, [(max(t1, 0), sgn * A), (max(t3, 0), 0.0), (max(t2, 0), -sgn * A)])))
    for a in (A, -A):                                                         # 같은 부호 두 구간 (a, 코스트, a): v_p 1차식
        den = Tf - (v1 - v0) / a
        if abs(den) < 1e-12:
            continue
        vp = ((x1 - x0) - (v1 ** 2 - v0 ** 2) / (2 * a)) / den
        t1 = (vp - v0) / a; t2 = (v1 - vp) / a; t3 = Tf - t1 - t2
        if min(t1, t2, t3) < -1e-9 or abs(vp) > V:
            continue
        cand.append((abs(vp), (Tf, [(max(t1, 0), a), (max(t3, 0), 0.0), (max(t2, 0), a)])))
    if cand:                                                                  # 11-08: 중간 속도는 평균 속도 Δx/Tf 에 가장 가까운 것 (최소 |v_p| 를 고르면 감속-정지-재가속, 영상의 "멈췄다 던지기")
        vavg = (x1 - x0) / max(Tf, 1e-9)
        return min(cand, key=lambda c: abs(c[1][1][0][1] * 0 + _vp_of(c[1], v0) - vavg))[1]
    for _ in range(80):
        m = 0.5 * (lo + hi); rm = min_time(x0, v0, x1, v1, m, V)
        if rm is None or rm[0] > Tf:
            lo = m
        else:
            hi = m; r = rm
    return r


def _segs(x0, v0, prof):
    T0, X, Vv, Aa = [0.0], [x0], [v0], []; x, v, t0 = x0, v0, 0.0
    for dt, a in prof:
        Aa.append(a); x = x + v * dt + 0.5 * a * dt * dt; v = v + a * dt; t0 += dt; T0.append(t0); X.append(x); Vv.append(v)
    Aa.append(0.0)
    return np.array(T0), np.array(X), np.array(Vv), np.array(Aa)


def sample(x0, v0, prof, ts):
    """프로파일을 시각 ts (0 부터) 에서 위치로 (벡터)."""
    T0, X, Vv, Aa = _segs(x0, v0, prof); ts = np.asarray(ts, float)
    j = np.clip(np.searchsorted(T0, ts, side="right") - 1, 0, len(T0) - 1); tau = ts - T0[j]
    return X[j] + Vv[j] * tau + 0.5 * Aa[j] * tau * tau


def ptp(q0, qd0, q1, qd1):
    """6 관절 동기화 점대점: (T, 프로파일 리스트). 관절별 최소 시간 최대 T 에 맞춤."""
    r = [min_time(q0[j], qd0[j], q1[j], qd1[j], QDD[j], QD[j]) for j in range(6)]
    if any(v is None for v in r):
        return None
    Tm = max(v[0] for v in r); Tm = np.ceil(Tm / DT) * DT
    pr = [fixed_time(q0[j], qd0[j], q1[j], qd1[j], QDD[j], QD[j], Tm)[1] for j in range(6)]
    return Tm, pr


def ptp_win(q0, qd0, q1r, qd1r):
    """실제가 창 끝 (t = T_pre + MEM) 에 정확히 (q1r, q̇1r) 을 지나게: 창 [T_pre, T_pre+MEM] 은 관절별 등가속 a ∈ {−A, 0, +A},
    그 앞은 시간최적 (동기화). 리턴 (T_pre, Q (N,6), q_end, qd_end) — q_end 는 창 끝 명령 상태."""
    if WIN_MODE == "const":
        r = ptp(q0, qd0, q1r + qd1r * (LAG - MEM), qd1r)
        if r is None:
            return None
        Tm, pr = r; ts = np.arange(0.0, Tm + 1e-9, DT); Q = np.stack([sample(q0[j], qd0[j], pr[j], ts) for j in range(6)], 1)
        tw = np.arange(DT, MEM + 1e-9, DT); Qw = np.stack([q1r + qd1r * (LAG - MEM + t) for t in tw])
        return Tm, np.vstack([Q, Qw]), q1r + qd1r * LAG, qd1r.copy()
    opts = []
    for j in range(6):
        best = None
        amax = min(QDD[j], (QD[j] - abs(qd1r[j])) / MEM)                      # 창 전체에서 속도 한계 안
        for a in np.linspace(-amax, amax, 21) if amax > 0 else (0.0,):
            uc = q1r[j] - 0.5 * a * VAR; udc = qd1r[j]; dl = MEM - LAG
            uws = uc - udc * dl + 0.5 * a * dl * dl; udws = udc - a * dl
            if max(abs(udws), abs(udws + a * MEM)) > QD[j]:
                continue
            r = min_time(q0[j], qd0[j], uws, udws, QDD[j], QD[j])
            if r is not None and (best is None or r[0] < best[0]):
                best = (r[0], a, uws, udws)
        if best is None:
            return None
        opts.append(best)
    Tp = np.ceil(max(o[0] for o in opts) / DT) * DT; Q = []; qe = np.zeros(6); qde = np.zeros(6)
    for j in range(6):
        _, a, uws, udws = opts[j]; pr = fixed_time(q0[j], qd0[j], uws, udws, QDD[j], QD[j], Tp)[1] + [(MEM, a)]
        ts = np.arange(0.0, Tp + MEM + 1e-9, DT); Q.append(sample(q0[j], qd0[j], pr, ts))
        qe[j] = uws + udws * MEM + 0.5 * a * MEM * MEM; qde[j] = udws + a * MEM
    return Tp, np.stack(Q, 1), qe, qde


def _vel_samples(x0, v0, prof, ts):
    """프로파일 속도 (정확, 벡터)."""
    T0, X, Vv, Aa = _segs(x0, v0, prof); ts = np.asarray(ts, float)
    j = np.clip(np.searchsorted(T0, ts, side="right") - 1, 0, len(T0) - 1)
    return Vv[j] + Aa[j] * (ts - T0[j])


def _real_end(x0, v0, prof, T, hx, hv):
    """실제 (필터) 위치·속도 at T: Σ h_k u(T − kDT) — 음수 시각은 이력 (hx 위치, hv 속도, 마지막이 t=0 직전)."""
    tq = T - KT; m = tq >= -1e-12
    u = np.empty(len(KT)); ud = np.empty(len(KT))
    u[m] = sample(x0, v0, prof, tq[m]); ud[m] = _vel_samples(x0, v0, prof, tq[m])
    if (~m).any():
        idx = np.clip(np.round(tq[~m] / DT).astype(int), -len(hx), -1); u[~m] = hx[idx]; ud[~m] = hv[idx]
    return float(hk @ u), float(hk @ ud)


C_ENDS = (LAG, 0.5 * LAG, 0.1 * LAG, 0.7 * MEM, 0.85 * MEM, MEM)              # 마지막 등속 길이 후보 — 끝 속도가 한계에 가까우면 길어야 (MEM 이면 등속 창)


def _solve_sw(x0, v0, A, V, T, sgn, x1, v1, hx, hv, c_end=LAG):
    """T·부호·c_end 고정: (t1, t2) 뉴턴 (프로파일 [t1 +sA, 코스트, t2 −sA, c_end 등속]). 리턴 prof 또는 None."""
    best = None; inits = []; Tm_ = T - c_end
    if Tm_ <= 1e-9:
        return None
    a1, a2 = sgn * A, -sgn * A
    g = fixed_time(x0, v0, x1, v1, A, V, Tm_)                                 # 초기값·가속도 크기: 실제 ≈ 명령(t−LAG) → T−c_end 에 목표를 맞추는 명령 공간 해
    if g is not None and g[1] and (g[1][0][1] * sgn > 0 or (g[1][0][1] == 0.0 and g[1][2][1] * sgn < 0)):
        a1 = g[1][0][1] if g[1][0][1] != 0.0 else -g[1][2][1]; a2 = g[1][2][1] if g[1][2][1] != 0.0 else -a1
        inits.append((g[1][0][0], g[1][2][0]))
    inits += [(0.02 * Tm_, 0.02 * Tm_), (0.1 * Tm_, 0.1 * Tm_)]
    for init in inits:
        t1, t2 = init
        for _ in range(30):
            def res(t1, t2):
                pr = [(t1, a1), (max(Tm_ - t1 - t2, 0.0), 0.0), (t2, a2), (c_end, 0.0)]
                p, v = _real_end(x0, v0, pr, T, hx, hv); return np.array([p - x1, v - v1])
            r0 = res(t1, t2)
            if np.linalg.norm(r0 * [1.0, 0.05]) < 1e-7:
                break
            e = 1e-5; Jm = np.stack([(res(t1 + e, t2) - r0) / e, (res(t1, t2 + e) - r0) / e], 1)
            try:
                d = np.linalg.solve(Jm, r0)
            except np.linalg.LinAlgError:
                break
            n = np.linalg.norm(d); d = d * min(1.0, 0.2 * Tm_ / max(n, 1e-12)); t1 -= d[0]; t2 -= d[1]
            t1 = min(max(t1, 0.0), Tm_); t2 = min(max(t2, 0.0), Tm_ - t1)
        else:
            continue
        if np.linalg.norm(res(t1, t2) * [1.0, 0.05]) > 1e-6:
            continue
        vp = v0 + a1 * t1
        if abs(vp) > V * (1 + 1e-3) or abs(vp + a2 * t2) > V * (1 + 1e-3):
            continue
        pr = [(t1, a1), (max(Tm_ - t1 - t2, 0.0), 0.0), (t2, a2), (c_end, 0.0)]
        if best is None:
            best = pr; break
    return best


def _solve_any(x0, v0, A, V, T, x1, v1, hx, hv):
    """c_end = LAG, 명령 해의 부호부터; 안 되면 나머지."""
    g = fixed_time(x0, v0, x1, v1, A, V, max(T - LAG, 1e-3)); sg0 = 1.0 if (g is None or not g[1] or g[1][0][1] >= 0) else -1.0
    for c_end in C_ENDS:
        for sg in (sg0, -sg0):
            pr = _solve_sw(x0, v0, A, V, T, sg, x1, v1, hx, hv, c_end)
            if pr is not None:
                return pr
    return None


def ptp_real(q0, qd0, hist, q1r, qd1r, T_hi=0.9):
    """창 없는 시간최적: 관절별 bang-bang 명령, 종단 = 실제 (필터) 상태 (q1r, q̇1r) at T. T 는 이분법 (2 ms), 동기화 = 공통 T 에서 다시 풀기.
    hist: 이 구간 앞 명령 (N,6). 리턴 (T, Q (N+1,6) 시작 포함, q_end 명령, qd_end 명령)."""
    hv = np.gradient(hist, DT, axis=0) if len(hist) > 1 else np.zeros_like(hist)
    Tj = []; sol = [None] * 6
    for j in range(6):
        r = min_time(q0[j], qd0[j], q1r[j], qd1r[j], QDD[j], QD[j])
        lo = 0.1 * LAG if r is None else max(0.1 * LAG, r[0]); hi = lo + LAG + 0.05; found = None
        while hi <= T_hi + 1e-9:                                              # 상한: 가능해질 때까지 늘린다
            found = _solve_any(q0[j], qd0[j], QDD[j], QD[j], hi, q1r[j], qd1r[j], hist[:, j], hv[:, j])
            if found is not None:
                break
            hi *= 1.4
        if found is None:
            if os.environ.get("V3_DBG"):
                r1 = _real_end(q0[j], qd0[j], [(hi, 0.0)], hi, hist[:, j], hv[:, j])
                print(f"ptp_real: 관절 {j+1} T_hi {hi} 불가: q0 {q0[j]:.3f} qd0 {qd0[j]:.2f} → 목표 {q1r[j]:.3f} {qd1r[j]:.2f}; min_time {None if r is None else round(r[0],3)}", flush=True)
            return None
        a_, b_ = lo, hi
        while b_ - a_ > DT:
            m = 0.5 * (a_ + b_); ok = _solve_any(q0[j], qd0[j], QDD[j], QD[j], m, q1r[j], qd1r[j], hist[:, j], hv[:, j]) is not None
            a_, b_ = (a_, m) if ok else (m, b_)
        Tj.append(b_)
    T = np.ceil(max(Tj) / DT) * DT
    for _ in range(50):
        ok = True
        for j in range(6):
            pr = _solve_any(q0[j], qd0[j], QDD[j], QD[j], T, q1r[j], qd1r[j], hist[:, j], hv[:, j])
            if pr is None:
                ok = False; break
            sol[j] = pr
        if ok:
            break
        T += DT
    else:
        if os.environ.get("V3_DBG"):
            print("ptp_real: 동기화 실패 T", T, flush=True)
        return None
    ts = np.arange(0.0, T + 1e-9, DT); Q = np.stack([sample(q0[j], qd0[j], sol[j], ts) for j in range(6)], 1)
    qe = Q[-1]; qde = np.array([_vel_samples(q0[j], qd0[j], sol[j], [T])[0] for j in range(6)])
    return T, Q, qe, qde


_HIST = {"u": None}                                                            # 현재 구간 앞 명령 스트림 (opt 모드 ptp_real 용)


def seg_win_to(q0, qd0, end_fn, n_fix=6):
    """실제 정확점 종단의 벨트 고정점 판: end_fn(t_end) → 실제 목표 (q, q̇). opt: ptp_real (창 없음), 그 밖: ptp_win."""
    T_ = 0.3
    for _ in range(n_fix):
        if WIN_MODE == "opt":
            q1, qd1 = end_fn(T_); r = ptp_real(q0, qd0, _HIST["u"], q1, qd1)
        else:
            q1, qd1 = end_fn(T_ + MEM); r = ptp_win(q0, qd0, q1, qd1)
        if r is None:
            return None
        if abs(r[0] - T_) < 1e-9:
            break
        T_ = r[0]
    return r


def seg_to(q0, qd0, end_fn, n_fix=6):
    """끝 상태가 시간에 따라 움직일 때 (벨트): T 고정점 반복. end_fn(T) → (q1, qd1). 리턴 (T, Q (N+1,6), q1, qd1)."""
    T_ = 0.3
    for _ in range(n_fix):
        q1, qd1 = end_fn(T_); r = ptp(q0, qd0, q1, qd1)
        if r is None:
            return None
        if abs(r[0] - T_) < 1e-9:
            break
        T_ = r[0]
    Tm, pr = r; ts = np.arange(0.0, Tm + 1e-9, DT)
    Q = np.stack([sample(q0[j], qd0[j], pr[j], ts) for j in range(6)], 1)
    return Tm, Q, q1, qd1


N_VIA = int(os.environ.get("V3_NVIA", "1")); N_CORR = int(os.environ.get("V3_NCORR", "0"))   # 11-08 사용자: 하강도 점 두 개 (h, 표면) 만 — 중간 경유점·보정 없음
N_VIA_LIFT = int(os.environ.get("V3_NVIA_LIFT", "1"))                          # 11-08: 상승은 한 구간 (경유점 사이 관절 지나침 → 실제 정지·되내림)
A_Z, V_Z = 12.0, 2.0                                                          # 경유점 속도 배분용 z 1차원 프로파일 한계 (관절 한계는 ptp 가 본다)


def line_segs(q0, qd0, t0, xy_fn, z0, zd0, z1, zd1, n_via, off=None, win_last=False, xy_real=None):
    """벨트 추종 수직선 (z0, ż0) → (z1, ż1) 를 경유점 n_via 개의 관절별 bang-bang 으로. 경유점 (z, ż, t) 는 z 1차원 bang-bang 에서.
    off: 경유점별 xy 보정 (실제 오차 피드포워드). win_last: 마지막 경유점은 실제 정확점 (창 포함, ptp_win).
    리턴 (Q (N,6) 시작 포함, t 끝 절대시각, q 끝, q̇ 끝, 경유 시각 리스트)."""
    r = min_time(z0, zd0, z1, zd1, A_Z, V_Z)
    if r is None:
        return None
    Tz, pz = r; tz = np.linspace(0.0, Tz, 400); zz = sample(z0, zd0, pz, tz); zd = np.gradient(zz, tz)
    zs = np.linspace(z0, z1, n_via + 1)[1:]; Qs = [q0[None]]; q, qd, t = q0, qd0, t0; tv = []
    for k, zk in enumerate(zs):
        i = int(np.argmin(np.abs(zz - zk))); zdk = zd1 if k == n_via - 1 else float(zd[i]); o_k = np.zeros(2) if off is None else off[k]
        last = win_last and k == n_via - 1
        fn = xy_real if last else xy_fn                                       # 정확점 (실제) 목표에는 리드 없음
        tgt_fn = lambda T_, zk=zk, zdk=zdk, o_k=o_k, fn=fn: (ik(np.r_[fn(t + T_)[:2] + o_k, zk]), qd_of(ik(np.r_[fn(t + T_)[:2] + o_k, zk]), np.r_[VB[:2], zdk]))
        rr = seg_win_to(q, qd, tgt_fn) if last else seg_to(q, qd, tgt_fn)
        if rr is None:
            if os.environ.get("V3_DBG"):
                print(f"line_segs: 경유 {k} (z {zk:.3f}, ż {zdk:.2f}, last {last}) 실패", flush=True)
            return None
        Tm, Q, q1, qd1 = rr; Qs.append(Q[1:]); q, qd = q1, qd1; t = t + Tm + (MEM if (last and WIN_MODE != "opt") else 0.0); tv.append(t)
        _HIST["u"] = np.vstack([_HIST["u"], Q[1:]])[-len(hk) - 2:]
    return np.vstack(Qs), t, q, qd, tv


def release_state(tgt, r_rel, z_rel, phi=0.0, phid=0.0):
    """릴리즈 (q, q̇): 발사점이 목표 방위각 반직선 위 (r, z), 툴 축이 수직에서 목표 쪽으로 phi 기울어짐, 최소 속력 각, ω = 0.
    q1 = ψ, q4 = 0, q6 = IK 기본; (q2, q3, q5) 는 뉴턴 (잔차: 발사점 r·z, 툴 축 z 성분). q̇ 은 평면 3×3 (v_r, v_z, ω_n = 0)."""
    global _psi_cur
    psi = np.arctan2(tgt[1], tgt[0]); _psi_cur = psi; er = np.array([np.cos(psi), np.sin(psi), 0.0]); en = np.array([-np.sin(psi), np.cos(psi), 0.0])
    p = np.array([r_rel * np.cos(psi), r_rel * np.sin(psi), z_rel])
    q0 = ik(np.array([0.35 * np.cos(psi), 0.35 * np.sin(psi), 0.2])).copy()  # 시작 추정 (수직 툴)
    I = [1, 2, 4]; q = q0.copy(); ok_ = False
    for st in (np.zeros(3), np.array([-0.4, 0.0, -0.6]), np.array([-0.6, 0.5, -0.6]), np.array([0.3, -0.5, 0.3])):   # 다중 시작
        q = q0.copy(); q[I] += st
        for _ in range(80):
            P = np.array(F_P(q)).ravel(); ax = np.array(F_AX(q)).ravel()
            res = np.array([P @ er - r_rel, P[2] - z_rel, ax[2] + np.cos(phi)])
            if np.linalg.norm(res) < 1e-10:
                ok_ = True; break
            Jp = np.array(F_JREL(q))[:, I]
            try:
                d = np.linalg.solve(Jp, res)
            except np.linalg.LinAlgError:
                break
            q[I] -= d * min(1.0, 0.3 / max(np.linalg.norm(d), 1e-12))           # 감쇠 (한 스텝 ≤ 0.3 rad)
        if ok_:
            break
    if not ok_:
        raise ValueError("릴리즈 IK 미수렴")
    ax = np.array(F_AX(q)).ravel()
    if abs(ax @ en) > 1e-6 or (phi > 1e-6 and ax @ er < 0):
        raise ValueError("릴리즈 툴 축 방향")
    if np.any(q < Q_LO) or np.any(q > Q_HI):
        raise ValueError("릴리즈 관절 상자 밖")
    R = np.hypot(tgt[0] - p[0], tgt[1] - p[1]); h = tgt[2] - p[2]
    th = np.pi / 4 + 0.5 * np.arctan2(h, R); v = np.sqrt(T.G * (h + np.hypot(R, h)))
    vv = v * (np.cos(th) * er + np.array([0.0, 0.0, np.sin(th)]))
    Jm = np.array(F_JM(q)); A = np.stack([Jm[0:3, I].T @ er, Jm[2, I], Jm[3:6, I].T @ en])
    qd = np.zeros(6); qd[I] = np.linalg.solve(A, np.array([v * np.cos(th), v * np.sin(th), phid]))   # 11-08 사용자: 릴리즈 각속도 0 조건 뺌 → ω_n = φ̇ (탐색 변수); 속도는 발사점 (흡착 오프셋 포함) 기준이라 착지는 그대로 정확
    if np.any(np.abs(qd) > QD):
        raise ValueError("릴리즈 관절 속도 밖")
    return q, qd, p, vv


def plan_cycle(o_i, o_n, tgt, zd_h, v_out, r_rel, z_rel, phi=0.0, top=ZI):
    """명령 스트림 (계획 규약, 2 ms) 과 구간 시각. 시작 = o_i 위 (xy, H) 하강 속도 zd_h 의 명령 상태 (t = 0)."""
    oi = C.moving(o_i); on = C.moving(o_n); seg = {}
    lead = lambda t: oi(t + LAG)                                               # 명령은 벨트 리드 v_b·LAG 를 싣는다 (실제 = 명령 지연)
    q_h = ik(np.r_[lead(0.0)[:2], H]); qd_h = qd_of(q_h, np.r_[VB[:2], -abs(zd_h)])
    # 1 하강 (경유점, 마지막 = 접촉 정확점 창) + 2 상승 (경유점): 실제 xy 오차를 경유점 목표에 피드포워드 (반복 N_CORR)
    nh = len(hk) + 5; Uh = np.stack([q_h - qd_h * k * DT for k in range(nh, 0, -1)])
    off1 = np.zeros((N_VIA, 2)); off2 = np.zeros((N_VIA, 2))
    for it in range(N_CORR + 1):
        _HIST["u"] = Uh.copy()
        r1 = line_segs(q_h, qd_h, 0.0, lead, H, -abs(zd_h), top, 0.0, N_VIA, off1, win_last=True, xy_real=oi)
        if r1 is None:
            return None
        Q1, t_k, q_k, qd_k, tv1 = r1
        r2 = line_segs(q_k, qd_k, t_k, lead, top, 0.0, H, v_out, N_VIA_LIFT, off2[:N_VIA_LIFT])
        if r2 is None:
            return None
        Q2, t_l, q_l, qd_l, tv2 = r2
        if it == N_CORR:
            break
        Up = np.vstack([Uh, Q1, Q2[1:], np.repeat(Q2[-1:], len(hk), 0)])
        qr = np.stack([np.convolve(Up[:, j], hk)[:len(Up)] for j in range(6)], -1)[nh:nh + len(Q1) + len(Q2) - 1]
        tcp = tcp_batch(qr * SG); tt = np.arange(len(qr)) * DT
        for k, tk in enumerate(tv1[:-1]):                                     # 접촉 (마지막) 은 창이 정확히 맞춘다
            i = int(round(tk / DT)); off1[k] -= tcp[i, :2] - oi(tt[i])[:2]
        for k, tk in enumerate(tv2[:N_VIA_LIFT]):
            i = int(round(tk / DT)); off2[k] -= tcp[i, :2] - oi(tt[i])[:2]
    W = 0.0 if WIN_MODE == "opt" else MEM
    T1 = t_k - W; T2 = t_l - t_k; t_ws = t_k - W
    # 3 아크 → 릴리즈 정확점
    q_r, qd_r, p_r, v_r = release_state(tgt, r_rel, z_rel, phi)
    r3 = ptp_real(q_l, qd_l, np.vstack([Uh, Q1, Q2[1:]])[-len(hk) - 2:], q_r, qd_r) if WIN_MODE == "opt" else ptp_win(q_l, qd_l, q_r, qd_r)
    if r3 is None:
        return None
    T3, Q3, q_re, qd_re = r3; t_rws = t_l + T3; t_rel = t_rws + W
    # 4 복귀 → 다음 물체 위 (xy, H) 하강 속도 zd_h
    nend = lambda T_: (ik(np.r_[on(t_rel + T_)[:2], H]), qd_of(ik(np.r_[on(t_rel + T_)[:2], H]), np.r_[VB[:2], -abs(zd_h)]))
    r4 = seg_to(q_re, qd_re, nend)
    if r4 is None:
        return None
    T4, Q4, _, _ = r4; t_end = t_rel + T4
    U = np.vstack([Q1, Q2[1:], Q3[1:], Q4[1:]])
    seg.update(t_ws=t_ws, t_k=t_k, t_l=t_l, t_rws=t_rws, t_rel=t_rel, t_end=t_end, T=[T1, W, T2, T3, W, T4], q_rel=q_r, qd_rel=qd_r, p_rel=p_r, v_rel=v_r, q_h=q_h, qd_h=qd_h)
    return U, seg


def evaluate(U, seg, o_i, tgt, top=ZI):
    """실제 (필터) 궤적 검사. 이력: t<0 은 hover 상태를 등속으로 거꾸로 연장."""
    q_h, qd_h = seg["q_h"], seg["qd_h"]; nh = len(hk) + 5
    Uh = np.stack([q_h - qd_h * k * DT for k in range(nh, 0, -1)])
    Uf = np.vstack([Uh, U, np.repeat(U[-1:], len(hk), 0)])
    qr = np.stack([np.convolve(Uf[:, j], hk)[:len(Uf)] for j in range(6)], -1)[nh:nh + len(U)]
    tt = np.arange(len(U)) * DT; oi = C.moving(o_i)
    k_k = int(round(seg["t_k"] / DT)); k_rel = int(round(seg["t_rel"] / DT)); k_l = int(round(seg["t_l"] / DT))
    tcp = tcp_batch(qr * SG); tcp_c = tcp_batch(U * SG)
    des = np.stack([oi(t) for t in tt]); desl = np.stack([oi(t + LAG) for t in tt]); ip = tt <= seg["t_k"]
    exy = np.linalg.norm(tcp[ip, :2] - des[ip, :2], axis=1); exy_c = np.linalg.norm(tcp_c[ip, :2] - desl[ip, :2], axis=1)
    qdr = np.gradient(qr, DT, axis=0)
    v_k = np.array(F_JT(qr[k_k]) @ qdr[k_k]).ravel()
    e2, tau = F_LAND(qr[k_rel], qdr[k_rel], tgt); p_rel = np.array(F_P(qr[k_rel])).ravel(); v_rel = np.array(F_V(qr[k_rel], qdr[k_rel])).ravel()
    w_rel = np.array(F_W(qr[k_rel], qdr[k_rel])).ravel()
    arc = (tt >= seg["t_l"] + MEM) & (tt <= seg["t_rel"])                      # 실제가 h 를 넘은 뒤부터
    ipz = tt <= seg["t_k"] + 0.1
    return dict(contact_xy_mm=1e3 * np.linalg.norm(tcp[k_k, :2] - des[k_k, :2]), contact_z_mm=1e3 * (tcp[k_k, 2] - top), contact_zd=float(v_k[2]), contact_vxy_err=float(np.linalg.norm(v_k[:2] - VB[:2])),
                band_real_mm=1e3 * float(exy.max()), band_cmd_mm=1e3 * float(exy_c.max()), z_path_min_mm=1e3 * float(tcp[ipz, 2].min() - top),
                land_cm=100 * float(np.sqrt(float(e2))), tau=float(tau), v_rel=float(np.linalg.norm(v_rel)), v_rel_plan=float(np.linalg.norm(seg["v_rel"])),
                p_rel_err_mm=1e3 * float(np.linalg.norm(p_rel - seg["p_rel"])), w_rel=float(np.linalg.norm(w_rel)),
                x_min=float(tcp[:, 0].min()), z_arc_min=float(tcp[arc, 2].min()) if arc.any() else float(H), z_max=float(tcp[:, 2].max()),
                q_box_ok=bool(np.all(U >= Q_LO - 1e-9) and np.all(U <= Q_HI + 1e-9)),
                qd_cmd_max_ratio=float(np.max(np.abs(np.gradient(U, DT, axis=0)) / QD)), qdd_cmd_max_ratio=float(np.max(np.abs(np.gradient(np.gradient(U, DT, axis=0), DT, axis=0))[2:-2] / QDD)),
                qd_real_max_ratio=float(np.max(np.abs(qdr) / QD)))


def main():
    z = np.load(sys.argv[1], allow_pickle=False); K = z["key"]; idx = [int(v) for v in sys.argv[2:]] or [100, 500, 900]
    print("N_VIA", N_VIA, "N_CORR", N_CORR, "WIN", WIN_MODE, "MEM %.3f LAG %.3f VAR %.5f" % (MEM, LAG, VAR))
    for i in idx:
        xi, yi, xn, yn, b = K[i]; o_i = np.array([xi, yi, ZI]); o_n = np.array([xn, yn, ZI]); tgt = np.array(BINS[int(b)], float)
        best = None; t0 = time.time(); n_try = 0; fails = {}; rej = {}
        G = dict(zd=(0.3, 0.6), vo=(1.0, 1.5), r=(0.55, 0.65), z=(0.5, 0.65), phi=(1.1, 1.4)) if os.environ.get("V3_GRID", "small") == "small" else             dict(zd=(0.3, 0.6, 0.9), vo=(0.5, 1.0, 1.5), r=(0.45, 0.55, 0.65), z=(0.35, 0.5, 0.65), phi=(0.5, 0.8, 1.1, 1.4))
        for zd_h in G["zd"]:
            for v_out in G["vo"]:
                for r_rel in G["r"]:
                    for z_rel in G["z"]:
                      for phi in G["phi"]:
                        try:
                            r = plan_cycle(o_i, o_n, tgt, zd_h, v_out, r_rel, z_rel, phi)
                        except Exception as e:                                # noqa: BLE001  IK 밖 등
                            fails[str(e)[:30]] = fails.get(str(e)[:30], 0) + 1; continue
                        if r is None:
                            continue
                        n_try += 1; U, seg = r; ev = evaluate(U, seg, o_i, tgt)
                        ok = ev["q_box_ok"] and ev["x_min"] >= 0.24 and ev["z_arc_min"] >= H - 0.003 and ev["qd_cmd_max_ratio"] <= 1.001 and ev["z_path_min_mm"] >= -0.5
                        if not ok:
                            why = "box" if not ev["q_box_ok"] else ("x" if ev["x_min"] < 0.24 else ("zarc" if ev["z_arc_min"] < H - 0.003 else ("qd" if ev["qd_cmd_max_ratio"] > 1.001 else "zpath")))
                            rej[why] = rej.get(why, 0) + 1
                        if ok and (best is None or seg["t_end"] < best[0]):
                            best = (seg["t_end"], dict(zd_h=zd_h, v_out=v_out, r_rel=r_rel, z_rel=z_rel, phi=phi), seg, ev)
        dt = time.time() - t0
        print(f"칸 {i} {K[i].tolist()} DB t_c {z['theta'][i][0]:.3f} J {z['J'][i]:.1f} | 후보 {n_try} 계산 {dt:.1f}s 예외 {fails} 거부 {rej}")
        if best is None:
            print("   통과 후보 없음"); continue
        te, par, seg, ev = best
        print("   선택", par, "구간 [하강, 창, 상승, 아크, 창, 복귀] =", [round(v, 3) for v in seg["T"]], "사이클 %.3f s (접촉→다음 hover %.3f, 접촉→릴리즈 %.3f)" % (te, te - seg["t_k"], seg["t_rel"] - seg["t_k"]))
        print("   ", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ev.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
