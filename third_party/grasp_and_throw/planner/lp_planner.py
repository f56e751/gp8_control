"""simpipe 용 계획 함수: version3 4 점 LP + SLP (lp_cycle2 와 같은 문제 정의). plan(start, end, hist, ctx) — 계약은 simpipe.py 머리말.

궤적 종류 → 사건 (모두 실제 = 필터를 지난 궤적 조건)
  잡기(i) → 따라가기(n) : 접촉 i, 상승 끝 (i 위 h), 릴리즈, n 위 hover, [n 접촉 — 내다보기만, 실행은 hover 까지]
  잡기(i) → 정지(p)     : 접촉 i, 상승 끝, 릴리즈, p 에서 정지 (명령을 p 자세에 고정)
  이동    → 따라가기(n) : n 위 hover, [n 접촉 내다보기]
  이동    → 정지(p)     : p 에서 정지
높이 하한: 하강·상승 ≥ 물체 윗면, 그 외 ≥ h (lp_cycle2 의 선형화 + 재선형화). 관절 4 고정. 목적 = 끝 사건 시각 최소 (SLP).
출발 시각: (realtime) DB 최근접 칸 → 기본 시각; 잡기 부분·끝 부분 배율을 따로 늘려 처음 가능한 조합 → 마감까지 SLP. (cold) 기본 시각에서 수렴까지.
환경: LPP_DB_NEXT (v3db_next_final.npz), LPC_BAND_POS / LPC_BAND_VEL (4 점 허용 띠; 여기서는 기본 3 mrad·0.03 rad/s — 11-10 사용자 허용), LPC_SLP_TR."""
import os, sys, time
os.environ.setdefault("GP8_CYC_NCTRL", "52"); os.environ.setdefault("LPC_LS", "0")
os.environ.setdefault("LPC_BAND_POS", "0.003"); os.environ.setdefault("LPC_BAND_VEL", "0.03")   # 11-10 사용자: 4 점 허용 띠 위치 3 mrad·속도 0.03 rad/s 허용
os.environ["LPC_BS"] = os.environ.get("LPP_BS", "5"); os.environ["LPC_BS_DEG"] = os.environ.get("LPP_BS_DEG", "2")   # 11-10 사용자: 명령 = 2 차 B-spline, 매듭 10 ms (제약은 2 ms 표본 그대로). db_env.sh 의 LPC_BS=0 을 덮어쓴다 — 바꾸려면 LPP_BS
import numpy as np

TCP_MODE = os.environ.get("LPP_TCP", "0") == "1"                              # 11-10 사용자: 4 점을 TCP 제약 (실제 궤적) 으로, 관절 4 free
if TCP_MODE:
    os.environ["LPC_FIX4"] = "0"
REL_MODE = os.environ.get("LPP_RELFREE", "0")                                  # 0 = 빈별 고정 릴리즈 상태, 1 = 발사 속도 3 행 (농구식 발사각·속력; 위치·각속도 자유 — 11-10 사용자), 2 = 착지점 2 행만
REL_FREE = REL_MODE in ("1", "2")                           # 11-10 사용자: 릴리즈 지점도 고정하지 않음 — 조건 = 착지점이 빈 조준점 (실제 궤적 기준)
WIN_ON = os.environ.get("LPP_WIN", "0") == "1"                                 # 11-10 사용자: 기존 윈도우 항 (릴리즈 ±RT/2 의 착지 오차²·각속도²) 을 목적에 추가
WIN_W1 = float(os.environ.get("LPP_W1", "100")); WIN_WACC = float(os.environ.get("LPP_WACC", "1e6")); WIN_WOM = float(os.environ.get("LPP_WOM", "1.0"))
WIN_RT = float(os.environ.get("LPP_WIN_RT", "0.1")); WIN_N = int(os.environ.get("LPP_WIN_N", "13")); TQ_TIE = float(os.environ.get("LPP_TQ_TIE", "1e-4"))
XMIN = float(os.environ["LPP_XMIN"]) if os.environ.get("LPP_XMIN") else None     # 11-10 사용자 "x 하한을 넣자": 실제 TCP x ≥ XMIN (궤적 전체). 없으면 끔
OBJ_NEXT = os.environ.get("LPP_OBJ_NEXT", "1") == "1"                          # 11-11 사용자: 시간 항 = 다음 물체 접촉 순간 (기본). 0 = 예전 (다음 물체 위 도착 순간)
_HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(_HERE, "lp_cycle2.py"), encoding="utf-8").read().split("def ev_contact(k, t0, n):")[0])   # lp, z_cons_from, realized_q, _ik_safe, REL, RELP, BINS, SG, …

_DB = {}
S1 = (1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0)
S2 = (1.0, 1.15, 1.35, 1.6, 2.0, 2.6)


_DBU = {}
VIN_ON = os.environ.get("LPP_VIN", "0") == "1"                                 # 11-11 사용자: 사이클 경계 = 명령이 물체 위 수직선을 v_in 상한으로 내려오며 h 를 지나는 순간
_VIN = {}


def _vin(r):
    """반경 r 의 v_in 상한 [m/s] — 미리 계산해 둔 표 (반경 5 cm 칸, 칸 안에서는 내림한 값). 범위 밖 반경은 가장 가까운 칸."""
    if not _VIN:
        import json
        T = json.load(open(os.environ.get("LPP_VIN_TABLE", os.path.join(_HERE, "logs", "vin_table.json")))); _VIN.update(e=np.asarray(T["r_edges"], float), v=np.asarray(T["v"], float))
    return float(_VIN["v"][int(np.clip(np.searchsorted(_VIN["e"], r + 1e-9, side="right") - 1, 0, len(_VIN["v"]) - 1))])


def vin_line(p_xy, v, Hh, k):
    """표준 수직 하강 직선의 명령 자세: 명령이 h 를 지나는 순간 (k = 0) 의 물체 위치 p_xy 기준, k 표본 전. 수평은 벨트 추종 (앞섬 LAG), 높이 h + v*k*DT. 도달 불가면 None."""
    tau = -k * DT
    return _ik_safe(np.r_[p_xy[0] + VB[0] * (tau + LAG), p_xy[1] + VB[1] * (tau + LAG), Hh + v * (-tau)])


def _db(path):
    if path and os.path.exists(path) and path not in _DB:
        Z = np.load(path, allow_pickle=True); _DB[path] = (np.asarray(Z["keys"], float), np.asarray(Z["ns"], int)); _DBU[path] = np.asarray(Z["U"], np.float32) if "U" in Z.files else None
    return _DB.get(path)


def _bin_idx(tgt):
    return int(np.argmin([np.linalg.norm(np.asarray(tgt, float)[:2] - np.asarray(b, float)[:2]) for b in BINS]))


def _mover(o):
    p0 = np.asarray(o["p"], float); v = np.asarray(o["v"], float)
    return lambda n: p0 + v * (n * DT)


def plan(start, end, hist, ctx):
    t0 = time.time(); NLP[0] = 0; deadline = ctx.get("deadline"); Hh = float(ctx.get("h", H))
    _lp_raw = globals()["lp"]

    def lp(*a_, **k_):                                                         # 11-11: 실시간이면 LP 한 번의 시간 제한 = 마감까지 남은 시간
        if deadline is not None:
            os.environ["LPC_TLIM"] = f"{max(0.05, deadline - time.time()):.3f}"
        return _lp_raw(*a_, **k_)
    Sh = np.asarray(hist, float) * SG
    if len(Sh) < L + 4:
        Sh = np.vstack([np.repeat(Sh[:1], L + 4 - len(Sh), 0), Sh])
    q0 = Sh[-1]; qd0 = (Sh[-1] - Sh[-2]) / DT
    mk, gaps, FL, seed_a, seed_b = [], [], [], [], []
    pick = start["kind"] == "pick"
    rel_free = REL_FREE and not ctx.get("_rel_fixed")
    TRUST = [float(os.environ.get("LPP_TRUST_Q", "0.15")), float(os.environ.get("LPP_TRUST_V", "1.0"))]   # 릴리즈 선형화 신뢰 구간 (rad, rad/s)
    REF = {}                                                                   # TCP 사건의 선형화 기준 (사건 번호 → 실제 자세, 실제 관절 속도) — 앞 반복 해로 갱신

    def tcp_ev(i, n, o, z, vfree):
        p = np.r_[o(n)[:2], z]; v = np.r_[VB[:2], 0.0] if vfree else np.asarray(VB, float)
        ref = REF.get(i)
        if ref is None:                                                        # 첫 기준: 관절 4 = 0 인 IK 자세 (도달 불가면 None)
            q = _ik_safe(p)
            if q is None:
                return None
            ref = (q, qd_of(q, v))
        return dict(n=n, tcp=True, p=p, v=v, vd=(True if vfree else None), lo=None, hi=None, qref=ref[0], qdref=ref[1], pos=ref[0])
    H_mv = Hh                                                                  # 11-10 사용자: 홈 TCP 높이를 0.12 로 — 이동 궤적도 하한 h 그대로 (예외 없음)
    if pick:
        oi = _mover(start["obj"]); top_i = float(start["obj"]["p"][2]); b = _bin_idx(start["obj"]["bin"]); q_r, qd_r = REL[b][0], REL[b][1]

        def ev_c(n, o=oi, top=top_i):
            q = _ik_safe(np.r_[o(n)[:2], top]); return None if q is None else dict(n=n, pos=q, vb=qd_of(q, VB), vd=None)

        def ev_l(n, o=oi):
            q = _ik_safe(np.r_[o(n)[:2], Hh]); return None if q is None else dict(n=n, pos=q, vb=qd_of(q, np.r_[VB[:2], 0.0]), vd=VZ(q), lo=None, hi=None)
        if TCP_MODE:
            ev_c = lambda n, o=oi, top=top_i: tcp_ev(0, n, o, top, False)
            ev_l = lambda n, o=oi: tcp_ev(1, n, o, Hh, True)
        tgt_b = np.asarray(start["obj"]["bin"], float)

        def ev_r(n):
            if not rel_free:
                return dict(n=n, pos=q_r, vb=qd_r, vd=None)
            ref = REF.get(2) or (q_r, qd_r)                                    # 첫 기준 = 빈별 고정 릴리즈 상태
            return dict(n=n, tgt=tgt_b, qref=ref[0], qdref=ref[1], pos=ref[0], trust=tuple(TRUST), **({"relv": True} if REL_MODE == "1" else {"land": True}))
        mk += [ev_c, ev_l, ev_r]; gaps += [20, 16, 30]; FL += [top_i, top_i, Hh]; seed_a = [142, 202, 364]
    if end["kind"] == "track":
        on = _mover(end["obj"]); top_n = float(end["obj"]["p"][2])

        def ev_h2(n, o=on):
            q = _ik_safe(np.r_[o(n)[:2], Hh]); return None if q is None else dict(n=n, pos=q, vb=qd_of(q, np.r_[VB[:2], 0.0]), vd=VZ(q), lo=None, hi=None)

        def ev_c2(n, o=on, top=top_n):
            q = _ik_safe(np.r_[o(n)[:2], top]); return None if q is None else dict(n=n, pos=q, vb=qd_of(q, VB), vd=None)
        if TCP_MODE:
            _i0 = len(mk)
            ev_h2 = lambda n, o=on, i=_i0: tcp_ev(i, n, o, Hh, True)
            ev_c2 = lambda n, o=on, top=top_n, i=_i0 + 1: tcp_ev(i, n, o, top, False)
        if VIN_ON and pick and TCP_MODE:                                       # 11-11 사용자: 자르는 곳 = 명령이 다음 물체 위 수직선을 v_in 상한으로 내려오며 h 를 지나는 순간; 그 뒤 hover·접촉은 내다보기
            def ev_cmd(n, o=on):
                p_ = o(n); v_ = _vin(float(np.hypot(p_[0], p_[1]))); a_ = vin_line(p_, v_, Hh, 0); b_ = vin_line(p_, v_, Hh, 1)
                return None if (a_ is None or b_ is None) else dict(n=n, cmd=True, pos=a_, pos1=b_, vin=v_)
            _i0 = len(mk) + 1
            ev_h2 = lambda n, o=on, i=_i0: tcp_ev(i, n, o, Hh, True)
            ev_c2 = lambda n, o=on, top=top_n, i=_i0 + 1: tcp_ev(i, n, o, top, False)
            mk += [ev_cmd, ev_h2, ev_c2]; gaps += [30, 10, 16]; FL += [Hh, Hh, top_n]; seed_b = [int(x_) for x_ in os.environ.get("LPP_VIN_SEED", "110,200,290").split(",")]; OB = len(mk) - 3
        else:
            mk += [ev_h2, ev_c2]; gaps += [30 if pick else 20, 16]; FL += [Hh if pick else H_mv, top_n]; seed_b = [134, 268] if pick else [260, 394]; OB = len(mk) - 2
        OBJ = (len(mk) - 1) if OBJ_NEXT else OB                                # 시간 항의 사건: 다음 물체 접촉 순간 (11-11 사용자) — 궤적은 여전히 hover 순간 (OB) 에서 자른다
    else:
        q_p = ik(np.asarray(end["p"], float))
        mk += [lambda n: dict(n=n, pos=q_p, hold=True, vb=None, vd=None)]; gaps += [30 if pick else 20]; FL += [Hh if pick else H_mv]; seed_b = [200] if pick else [320]; OB = len(mk) - 1; OBJ = OB
    NE = len(mk); na = len(seed_a)
    # 높이 하한 선형화의 시작 기준 자세 = 궤적 시작 시각의 **실제** 자세 (필터를 지난 값). 마지막 명령 Sh[-1] 은 필터 앞섬 때문에 실제 팔과 멀 수 있다
    # (11-10: 명령 TCP z 0.75 vs 실제 0.12 → 기준이 틀려 가짜 불가능, "잡기 → 정지" 가 계속 버려짐).
    q_start = np.array([float(np.dot(hk, Sh[::-1][:len(hk), j])) for j in range(6)])

    def zc(ns, evs, qr=None):
        zz = z_cons_from(q_start, ns, [e["pos"] for e in evs], FL[:len(ns)], qr)
        if XMIN is not None:                                                   # x 하한: z 하한과 같은 표본·같은 기준 자세에서 선형화
            zz = list(zz) + [(c[0], c[1], XMIN, 0) for c in zz]
        return zz
    CUR = {"u": None}                                                          # 지금 해 (창 항 선형화 기준)
    WTR0 = (float(os.environ.get("LPP_WTRUST_Q", "0.05")), float(os.environ.get("LPP_WTRUST_V", "0.3"))); WTR = list(WTR0)   # 창 노드 선형화 신뢰 구간 (rad, rad/s) — 풀이 장치
    TOLP = float(os.environ.get("LPP_TOL_TCP", "0.004")); TOLL = float(os.environ.get("LPP_TOL_LAND", "0.02"))

    def wtr_scale(k):
        WTR[0] = min(WTR0[0], WTR[0] * k); WTR[1] = min(WTR0[1], WTR[1] * k)

    def asum(u_):
        """가속 크기 합 (한계 대비 비율의 표본 합) — LP 의 smooth 항과 같은 값."""
        a_ = np.abs(np.diff(np.vstack([Sh[-2:], u_]), 2, axis=0)) / (DT * DT)
        return float(sum(np.sum(a_[:, j]) / V.QDD[j] for j in range(6)))

    def resid(ns_, u_):
        """비선형 조건의 진짜 잔차: (TCP 위치 [m], 릴리즈 조건 [m 또는 m/s])."""
        qr_ = realized_q(q0, Sh, u_); rp = 0.0; rl = 0.0
        for i_, n_ in enumerate(ns_):
            e_ = mk[i_](n_)
            if e_ is None:
                continue
            if e_.get("tcp"):
                rp = max(rp, float(np.linalg.norm(np.array(F_TCP(qr_[n_ - 1])).ravel() - e_["p"])))
            if e_.get("land") or e_.get("relv"):
                rl = max(rl, float(np.linalg.norm(np.array(F_LVEC(qr_[n_ - 1], (qr_[n_] - qr_[n_ - 2]) / (2 * DT), e_["tgt"])).ravel())))
        return rp, rl
    WINP = WIN_ON and pick; DBG = bool(os.environ.get("LPP_DEBUG"))

    def win_nodes(ns_, nq):
        """창 노드 (표본 번호, 사다리꼴 가중): 릴리즈 ± RT/2 에 WIN_N 개 — 상승 끝과 다음 사건 사이로 자른다."""
        offs = np.round(np.linspace(-WIN_RT / 2, WIN_RT / 2, WIN_N) / DT).astype(int); wts = np.full(WIN_N, WIN_RT / (WIN_N - 1)); wts[[0, -1]] *= 0.5
        lo = ns_[1] + 2; hi = min(ns_[3] - 2, nq - 3)
        return [(ns_[2] + int(o_), float(w_)) for o_, w_ in zip(offs, wts) if lo <= ns_[2] + int(o_) <= hi]

    def mkwin(ns_, with_d=False):
        """릴리즈 창 항 (LP 용): 지금 해의 실제 궤적을 기준으로 노드마다 선형화. 해가 아직 없거나 창 항을 안 쓰면 None."""
        if not WINP or CUR["u"] is None:
            return None
        qr_ = realized_q(q0, Sh, CUR["u"]); st_ = lambda nn: (qr_[nn - 1], (qr_[nn] - qr_[nn - 2]) / (2 * DT)); nodes = []
        sh_ = (CUR["ns"][2] - ns_[2]) if CUR.get("ns") else 0                  # 시각이 바뀐 LP: 기준 = 앞 해에서 릴리즈 기준 같은 위치의 상태
        for n_, w_ in win_nodes(ns_, 10 ** 9):
            m_ = int(np.clip(n_ + sh_, 3, len(qr_) - 3)); qb, qdb = st_(m_); de = dw = None
            if with_d:
                (qp, qdp), (qm, qdm) = st_(m_ + 1), st_(m_ - 1)
                de = (np.array(F_LVEC(qp, qdp, tgt_b)).ravel() - np.array(F_LVEC(qm, qdm, tgt_b)).ravel()) / (2 * DT)
                dw = (np.array(F_W(qp, qdp)).ravel() - np.array(F_W(qm, qdm)).ravel()) / (2 * DT)
            nodes.append((n_, w_, qb.copy(), qdb.copy(), de, dw))
        return dict(nodes=nodes, tgt=tgt_b, wacc=WIN_WACC, wom=WIN_WOM, w1=WIN_W1, irel=2, trust=tuple(WTR)) if nodes else None

    def wcost(ns_, u_):
        """창 항의 진짜 값 (선형화 없이): Σ_m w_m (W_ACC·‖착지 오차‖² + W_OMEGA·‖ω‖²), 그리고 (착지 부분, 각속도 부분)."""
        if not WINP:
            return 0.0, 0.0
        qr_ = realized_q(q0, Sh, u_); ca = 0.0; co = 0.0
        for n_, w_ in win_nodes(ns_, len(qr_)):
            qb, qdb = qr_[n_ - 1], (qr_[n_] - qr_[n_ - 2]) / (2 * DT)
            e_ = np.array(F_LVEC(qb, qdb, tgt_b)).ravel(); o_ = np.array(F_W(qb, qdb)).ravel(); ca += w_ * WIN_WACC * float(e_ @ e_); co += w_ * WIN_WOM * float(o_ @ o_)
        return ca, co

    def Jtot(ns_, u_):
        return WIN_W1 * ns_[OBJ] * DT + sum(wcost(ns_, u_))

    def upd_ref(ns_, u_):
        """TCP 사건의 선형화 기준을 해 u_ 의 실제 자세·속도로."""
        if not (TCP_MODE or rel_free):
            return
        qr_ = realized_q(q0, Sh, u_)
        for i_, n_ in enumerate(ns_):
            e_ = mk[i_](n_)
            if e_ is not None and (e_.get("tcp") or e_.get("land") or e_.get("relv")) and n_ + 1 <= len(qr_):
                REF[i_] = (qr_[n_ - 1].copy(), (qr_[n_] - qr_[n_ - 2]) / (2 * DT))
    est = [1.0 if pick else 0.15]                                               # LP 1 번 예상 시간 (측정값으로 갱신) — 이동 궤적은 문제가 작다

    def time_ok(k=1.0):
        return deadline is None or (deadline - time.time()) > k * est[0] * 1.15

    def exact(ns, qr=None, sm=False):
        if ns[0] < gaps[0] or any(ns[i + 1] - ns[i] < gaps[i + 1] for i in range(NE - 1)):
            return None
        evs = [mk[j](ns[j]) for j in range(NE)]
        if any(e is None for e in evs):
            return None
        t1 = time.time(); r = lp(q0, qd0, Sh, evs, ns[-1], zcons=zc(ns, evs, qr), win=mkwin(ns), smooth=sm); est[0] = 0.6 * est[0] + 0.4 * (time.time() - t1); return r
    # ---- 출발 시각
    ns = None; sol = None; how = ""; _dbref = [False]
    cands = []
    _dcands = []
    if ctx.get("mode") == "realtime" and pick and end["kind"] == "track" and (start.get("state") == "track" or os.environ.get("LPP_RT_DIRECT", "0") == "1"):
        _dbp = os.environ.get("LPP_DB_NEXT", os.path.join(_HERE, "logs", "v3db_next_final.npz")); D = _db(_dbp)
        if D is not None:
            K, NSd = D; m = K[:, -1].astype(int) == b
            key = np.r_[start["obj"]["p"][:2], end["obj"]["p"][:2]]
            if m.any():
                d = np.sum((K[m][:, :4] - key) ** 2, axis=1); i = int(np.argmin(d)); base = [int(x) for x in NSd[m][i]]
                if start.get("state") == "track":
                    cands += [(f"db{np.sqrt(d[i]) * 100:.0f}cm x{f:.2f}", [int(round(n * f / 2) * 2) for n in base]) for f in (1.1, 1.2, 1.35)]
                if rel_free and os.environ.get("LPP_DB_REF", "1") == "1" and _DBU.get(_dbp) is not None:   # 11-11: DB 해의 실제 릴리즈 상태 = 릴리즈 선형화 첫 기준
                    _u0 = np.asarray(_DBU[_dbp][int(np.flatnonzero(m)[i])][:base[3]], float) * SG; _qr0 = realized_q(q0, Sh, _u0)
                    if 3 <= base[2] and base[2] + 1 <= len(_qr0):
                        REF[2] = (_qr0[base[2] - 1].copy(), (_qr0[base[2]] - _qr0[base[2] - 2]) / (2 * DT)); _dbref[0] = True
                        if os.environ.get("LPP_RT_DIRECT", "0") == "1":             # 11-11 사용자: DB 해를 거의 그대로, LP 한두 번
                            if WIN_ON:
                                CUR.update(u=_u0, ns=list(base))                   # 창 항 선형화 기준도 DB 해 (릴리즈 기준으로 맞춰 씀)
                            if start.get("cont"):
                                _hs = [max(gaps[0], int(start["cont"]))]           # 접촉 시각 = 앞 계획이 예정한 시각
                            elif start.get("state") == "track":
                                _hs = [base[0]]
                            else:
                                _hs = [int(round(seed_a[0] * f / 2) * 2) for f in (1.5, 1.75, 1.35, 2.0)]   # 대기점 출발: 예전 실시간 하강 배율 순서
                            _dcands = [(f"dbd{np.sqrt(d[i]) * 100:.0f}cm h{h} x{g:.2f}", [h] + [h + int(round((base[j] - base[0]) * g / 2) * 2) for j in range(1, 5)]) for h in _hs for g in (1.0, 1.1, 1.2, 1.35)]
    rt = ctx.get("mode") == "realtime"                                         # 실시간: 넉넉한 시각부터 (첫 가능해를 빨리 얻고, 남는 시간에 SLP 로 당긴다)
    A = []
    if pick and start.get("cont"):                                             # 따라가던 중 잡기: 앞 계획이 예정한 '도착 → 접촉' 시간에 맞춘 시각표를 먼저 (초기값만)
        for h in (int(start["cont"]), int(start["cont"]) + 2, int(start["cont"]) + 6):   # 예정 시각 그대로 (반올림 없음), 그리고 조금 늦춘 것
            h = max(gaps[0], h); A.append((f"cont{h}", [h, h + seed_a[1] - seed_a[0], h + seed_a[2] - seed_a[0]]))
    A += [(f"seed x{f1:.2f}", [int(round(n * f1 / 2) * 2) for n in seed_a]) for f1 in ((((1.5, 1.75, 1.35, 2.0, 2.5, 1.2, 3.0) if rt else S1)) if pick else (1.0,))]   # 잡기 부분 배율 × 끝 부분 배율
    for tg, a in A:
        for f2 in ((1.15, 1.35, 1.6, 2.0) if rt else S2):
            bb = []; last = a[-1] if a else 0
            for g in seed_b:
                last = last + int(round(g * f2 / 2) * 2) if not bb else bb[0] + int(round((g - seed_b[0]) * f2 / 2) * 2); bb.append(last)
            cands.append((f"{tg}/{f2:.2f}", a + bb))
    cands = _dcands + cands
    if ctx.get("ns0") is not None:                                             # 출발 시각표를 직접 받음 (다중 초기값) — 안 풀리면 조금씩 늘려 처음 풀리는 것
        cands = [(f"ns0 x{f:.2f}", [int(n) if f == 1.0 else int(round(n * f / 2) * 2) for n in ctx["ns0"]]) for f in (1.0, 1.05, 1.1, 1.2, 1.35)]
    if ctx.get("ns0") is not None and ctx.get("u0") is not None and pick and rel_free:   # 11-11: 출발 시각표를 준 해의 실제 릴리즈 상태를 릴리즈 선형화 첫 기준으로 (풀이 장치)
        _u0 = np.asarray(ctx["u0"], float) * SG; _qr0 = realized_q(q0, Sh, _u0); _n2 = int(ctx["ns0"][2])
        if 3 <= _n2 and _n2 + 1 <= len(_qr0):
            REF[2] = (_qr0[_n2 - 1].copy(), (_qr0[_n2] - _qr0[_n2 - 2]) / (2 * DT))
    n_try = 0
    for tag, c in cands:
        if n_try > 0 and not time_ok():                                        # 첫 후보는 무조건 시도
            break
        if _dbref[0] and not tag.startswith("db"):                           # DB 후보가 다 안 풀림 → 릴리즈 기준을 원래 (빈별 고정 상태) 로
            REF.pop(2, None); _dbref[0] = False; CUR["u"] = None; CUR.pop("ns", None)
        n_try += 1; _tc = time.time(); r = exact(c)
        if os.environ.get("LPP_DEBUG"):
            print(f"   [출발 후보] {tag} {c} → {'가능' if r is not None else '불가'} ({time.time() - _tc:.1f} s)", flush=True)
        if r is not None:
            ns, sol, how = list(c), r, tag; break
    if ns is None:
        return dict(ok=False, why=f"출발 가능해 없음 (시도 {n_try}/{len(cands)}, LP {NLP[0]}, {time.time() - t0:.1f}s)")
    _direct = how.startswith("dbd")                                            # DB 직접: 시각 탐색·다듬기 없이 (비선형 잔차 재선형화만)
    upd_ref(ns, sol[0]); CUR.update(u=sol[0], ns=list(ns)); Jc = [Jtot(ns, sol[0])]
    if WINP and not _direct:                                                                   # 출발해는 창 항 없이 얻었다 → 같은 시각에서 창 항을 넣어 다시 (진짜 목적이 줄 때만 채택)
        for _ in range(6):
            if not time_ok() or WTR[0] < 2e-3:
                break
            r = exact(ns, realized_q(q0, Sh, sol[0]))
            if r is None:
                wtr_scale(0.5); continue
            j2 = Jtot(ns, r[0])
            if DBG:
                print(f"   [win 출발] J {Jc[0]:.2f} → {j2:.2f} (착지 {wcost(ns, r[0])[0]:.2f}, 각속도 {wcost(ns, r[0])[1]:.2f}) 신뢰 {WTR[0]:.3f}", flush=True)
            if j2 >= Jc[0] - 1e-6:
                wtr_scale(0.5); continue
            wtr_scale(2.0)
            sol = r; Jc[0] = j2; upd_ref(ns, sol[0]); CUR.update(u=sol[0], ns=list(ns))
    # ---- SLP (마감까지; cold 는 수렴까지)
    TR = float(os.environ.get("LPC_SLP_TR", "0.02")); it = 0
    TQ_W = float(os.environ.get("LPP_TQ_W", "0"))                               # 11-10 사용자 "토크 항 가중치": 목적 = 끝 시각 [s] + TQ_W × (관절 가속 크기 합, 한계 대비 비율 × 시간 [s]). 0 = 시간 우선 (동점 처리만)
    RST = os.environ.get("LPC_SLP_RESTART", "0") == "1"; TR0 = TR; n_rs = 0
    _rsv = lambda: (Jc[0] if WINP else WIN_W1 * ns[OBJ] * DT); v_rs = _rsv(); rs_log = []
    while it < int(os.environ.get("LPC_SLP_IT", "40")) and time_ok(2.0) and not _direct:
        if TR < DT * 0.99:                                                     # 한 바퀴 끝 (이동 폭 < 2 ms)
            if not RST or v_rs - _rsv() <= WIN_W1 * DT + 1e-9:
                break
            n_rs += 1; rs_log.append([it, int(ns[OBJ]), round(float(_rsv()), 2)]); v_rs = _rsv(); TR = TR0
            if WINP:
                WTR[:] = list(WTR0)
            if DBG:
                print(f"   [slp 재시작 {n_rs}] 반복 {it}, 시각 {ns}, 목적 {v_rs:.2f}", flush=True)
        it += 1; u, sc = sol; qr = realized_q(q0, Sh, u); upd_ref(ns, u); CUR.update(u=u, ns=list(ns)); evs = [mk[j](ns[j]) for j in range(NE)]; zi = 0; bad = False
        for i, e in enumerate(evs):
            if e.get("hold"):
                continue
            if e.get("cmd"):                                                   # 명령 사건: 시각 이동 계수 = (지금 해의 명령 속도) - (목표의 시간 변화율)
                n = ns[i]; ep, em = mk[i](n + 1), mk[i](n - 1)
                if ep is None or em is None:
                    bad = True; break
                e["dpos"] = (u[n] - u[n - 2]) / (2 * DT) - (ep["pos"] - em["pos"]) / (2 * DT); e["dpos1"] = (u[n - 1] - u[n - 3]) / (2 * DT) - (ep["pos1"] - em["pos1"]) / (2 * DT)
                continue
            if e.get("land") or e.get("relv"):                                 # 릴리즈 자유: 앞 반복 실제 궤적에서 잔차의 시간 변화율
                n = ns[i]
                el = (lambda nn, tg=e["tgt"]: relv_res(qr[nn - 1], (qr[nn] - qr[nn - 2]) / (2 * DT), tg)) if e.get("relv") else                      (lambda nn, tg=e["tgt"]: np.array(F_LVEC(qr[nn - 1], (qr[nn] - qr[nn - 2]) / (2 * DT), tg)).ravel())
                e["tdland"] = (el(n + 1) - el(n - 1)) / (2 * DT)
                continue
            if e.get("tcp"):                                                   # TCP 사건: 앞 반복 실제 궤적의 작업공간 속도·가속으로 δ 계수
                n = ns[i]

                def _tw(nn):
                    qq = qr[nn - 1]; qqd = (qr[nn] - qr[nn - 2]) / (2 * DT)
                    return np.array(F_JT(qq)) @ qqd, np.array(F_AX(qq)).ravel()[:2]
                tw0, _a0 = _tw(n); twm, am = _tw(n - 1); twp, ap = _tw(n + 1)
                e["tdpos"] = np.r_[tw0[:3] - np.r_[VB[:2], 0.0], (ap - am) / (2 * DT)]; e["tdvel"] = (twp - twm) / (2 * DT)
                if e.get("vd") is not None:
                    zi += 1
                continue
            ep, em = mk[i](ns[i] + 1), mk[i](ns[i] - 1)
            if ep is None or em is None:
                bad = True; break
            n = ns[i]; qd_ref = (qr[n] - qr[n - 2]) / (2 * DT); qdd_ref = (qr[n] - 2 * qr[n - 1] + qr[n - 2]) / DT ** 2
            kd = (ep["pos"] - em["pos"]) / (2 * DT); vt = (ep["vb"] - em["vb"]) / (2 * DT)
            if e.get("vd") is not None:
                vt = vt + float(sc[zi]) * (ep["vd"] - em["vd"]) / (2 * DT); zi += 1
            e["dpos"] = qd_ref - kd; e["dvel"] = qdd_ref - vt
        if bad:
            break
        t1 = time.time(); r = lp(q0, qd0, Sh, evs, ns[-1], zcons=zc(ns, evs, qr), slp=dict(tr=TR, gaps=gaps, ns=ns, obj=OBJ), smooth=(TQ_W * DT * (WIN_W1 if WINP else 1.0) if TQ_W > 0 else False), win=mkwin(ns, True)); est[0] = 0.6 * est[0] + 0.4 * (time.time() - t1)
        if r is None:
            TR /= 2; continue
        sh = [int(round(x / DT)) for x in r[1][-NE:]]
        if all(x == 0 for x in sh):
            if TR <= DT * 1.01:
                TR = 0.0; continue
            TR /= 2; continue
        ns_new = [a_ + b_ for a_, b_ in zip(ns, sh)]
        if ((not WINP) and ns_new[OBJ] > ns[OBJ]) or not time_ok():
            TR /= 2; continue
        s2 = exact(ns_new)
        if s2 is None:
            TR /= 2; continue
        if WINP:                                                               # 창 항이 있으면 채택 기준 = 진짜 목적 J (시간 + 창 항) 감소
            j2 = Jtot(ns_new, s2[0])
            if DBG:
                print(f"   [slp {it}] 시각 {ns_new} J {Jc[0]:.2f} → {j2:.2f} (착지 {wcost(ns_new, s2[0])[0]:.2f}, 각속도 {wcost(ns_new, s2[0])[1]:.2f}) TR {TR * 1e3:.0f} ms", flush=True)
            if j2 >= Jc[0] - 1e-6:
                TR /= 2; wtr_scale(0.5); continue
            Jc[0] = j2; wtr_scale(2.0)
        ns, sol = ns_new, s2; upd_ref(ns, sol[0]); CUR.update(u=sol[0], ns=list(ns))
    n_ref = 0
    if WINP and not _direct:                                                                   # 시각 고정 다듬기: 순차 선형화, 진짜 목적 (창 항 + 가속 합 동점 처리) 이 줄 때만 채택 — 아니면 창 신뢰 구간 반으로
        merit = lambda u_: sum(wcost(ns, u_)) + TQ_TIE * asum(u_)
        WTR[:] = list(WTR0); m_c = merit(sol[0]); rp_c, rl_c = resid(ns, sol[0])
        for _ in range(int(os.environ.get("LPP_WIN_IT", "14"))):
            if not time_ok() or WTR[0] < 2e-3:
                break
            upd_ref(ns, sol[0]); CUR.update(u=sol[0], ns=list(ns)); r = exact(ns, realized_q(q0, Sh, sol[0]), sm=TQ_TIE); ok_ = False; m2 = rp2 = rl2 = float("nan")
            if r is not None:
                m2 = merit(r[0]); rp2, rl2 = resid(ns, r[0]); f_c = rp_c <= TOLP and rl_c <= TOLL; f_2 = rp2 <= TOLP and rl2 <= TOLL
                ok_ = (f_2 and (m2 < m_c - 1e-3 or not f_c)) or ((not f_c) and (not f_2) and max(rp2 / TOLP, rl2 / TOLL) < 0.9 * max(rp_c / TOLP, rl_c / TOLL))
            if DBG:
                print(f"   [다듬기] 목적 {m_c:.2f} → {m2:.2f}, TCP {rp2 * 1e3:.2f} mm, 릴리즈 {rl2 * 100:.2f}, 신뢰 {WTR[0]:.3f}, 채택 {ok_}", flush=True)
            if not ok_:
                wtr_scale(0.5); continue
            gain = m_c - m2; sol = r; m_c, rp_c, rl_c = m2, rp2, rl2; n_ref += 1; wtr_scale(1.5)
            if 0 <= gain < max(0.02, 0.003 * m_c):
                break
    for _ in range(0 if WINP else int(os.environ.get("LPC_ZITER", "2")) + (2 if TCP_MODE else 0)):   # 높이 하한 (그리고 TCP 사건) 재선형화
        if not time_ok():
            break
        upd_ref(ns, sol[0]); CUR.update(u=sol[0], ns=list(ns)); qr = realized_q(q0, Sh, sol[0]); r = exact(ns, qr)
        if r is None:
            break
        sol = r
        if WINP and DBG:
            print(f"   [재선형화] J {Jtot(ns, sol[0]):.2f} (착지 {wcost(ns, sol[0])[0]:.2f}, 각속도 {wcost(ns, sol[0])[1]:.2f})", flush=True)
    n_tq = 0
    if os.environ.get("LPP_TORQUE", "1") == "1" and not WINP:                               # 11-10 사용자: 시간 최소 뒤, 같은 시각에서 관절 가속 크기 합 (토크 대용) 최소 — 사이클 시간 불변
        for _ in range(2):
            if not time_ok():
                break
            upd_ref(ns, sol[0]); CUR.update(u=sol[0], ns=list(ns)); qr = realized_q(q0, Sh, sol[0]); evs = [mk[j](ns[j]) for j in range(NE)]
            if any(e is None for e in evs):
                break
            t1 = time.time(); r = lp(q0, qd0, Sh, evs, ns[-1], smooth=(TQ_TIE if WINP else True), zcons=zc(ns, evs, qr), win=mkwin(ns)); t_tq = time.time() - t1
            if r is None:                                                       # 불가·시간 제한 → 1 단계 해 그대로
                break
            sol = r; n_tq += 1
    n_nl = 0
    nl_ok = True
    if TCP_MODE or rel_free:                                                   # 비선형 조건 (TCP·착지) 의 진짜 잔차가 허용값 안에 들 때까지 같은 시각에서 재선형화 반복
        def _resid(u_):
            qr_ = realized_q(q0, Sh, u_); rp = 0.0; rl = 0.0
            for i_, n_ in enumerate(ns):
                e_ = mk[i_](n_)
                if e_ is None:
                    continue
                if e_.get("tcp"):
                    rp = max(rp, float(np.linalg.norm(np.array(F_TCP(qr_[n_ - 1])).ravel() - e_["p"])))
                if e_.get("land") or e_.get("relv"):
                    rl = max(rl, float(np.linalg.norm(np.array(F_LVEC(qr_[n_ - 1], (qr_[n_] - qr_[n_ - 2]) / (2 * DT), e_["tgt"])).ravel())))
            return rp, rl
        TOL_P = float(os.environ.get("LPP_TOL_TCP", "0.004")); TOL_L = float(os.environ.get("LPP_TOL_LAND", "0.02")); rl_prev = 1e9
        for _ in range(int(os.environ.get("LPP_NL_IT", "16"))):
            rp, rl = _resid(sol[0])
            if rl > 0.7 * rl_prev:                                              # 잔차가 충분히 안 줄면 신뢰 구간을 반으로 (진동 억제)
                TRUST[0] *= 0.5; TRUST[1] *= 0.5
            rl_prev = rl
            if os.environ.get("LPP_DEBUG"):
                print(f"   [nl {n_nl}] TCP 잔차 {rp * 1e3:.2f} mm, 착지 잔차 {rl * 100:.2f} cm, 시각 {ns}", flush=True)
            if (rp <= TOL_P and rl <= TOL_L) or not time_ok():
                break
            upd_ref(ns, sol[0]); CUR.update(u=sol[0], ns=list(ns)); qr = realized_q(q0, Sh, sol[0]); evs = [mk[j](ns[j]) for j in range(NE)]
            if any(e is None for e in evs):
                break
            r = lp(q0, qd0, Sh, evs, ns[-1], smooth=(TQ_TIE if WINP else (n_tq > 0)), zcons=zc(ns, evs, qr), win=mkwin(ns))
            if r is None:
                break
            sol = r; n_nl += 1
        rp, rl = _resid(sol[0]); nl_ok = bool(rp <= TOL_P and rl <= TOL_L)
        if os.environ.get("LPP_DEBUG"):
            print(f"   [nl 끝] TCP 잔차 {rp * 1e3:.2f} mm, 착지 잔차 {rl * 100:.2f} cm, 반복 {n_nl}, 수렴 {nl_ok}", flush=True)
    if not nl_ok:                                                              # 수렴 못 한 해는 내보내지 않는다 (11-10: 28 cm 빗나감). 릴리즈 자유였으면 고정 릴리즈로 다시
        if rel_free and not ctx.get("_rel_fixed"):
            c2 = dict(ctx); c2["_rel_fixed"] = True; out = plan(start, end, hist, c2)
            if out.get("ok"):
                out["info"]["rel_fallback"] = 1
            return out
        return dict(ok=False, why=f"비선형 조건 미수렴 (TCP {rp * 1e3:.1f} mm, 착지 {rl * 100:.1f} cm)")
    u = sol[0]; n_end = ns[OB]; U = (u[:n_end] * SG)
    suction = [(20, True), (ns[2], False)] if pick else []
    # 판정 (계획 모델): 끝 상태·z 최소
    Sj = np.vstack([Sh, u[:ns[-1] + 2]]); Qr = np.stack([np.convolve(Sj[:, j], hk)[:len(Sj)] for j in range(6)], -1); P = tcp_batch(Qr * SG); n0 = len(Sh) - 1
    info = dict(ns=[int(x) for x in ns], t_end=round(n_end * DT, 3), t_obj=round(ns[OBJ] * DT, 3), obj_next=int(OBJ != OB), how=how, slp_it=it, n_lp=int(NLP[0]), s=round(time.time() - t0, 1), bs=f"{os.environ['LPC_BS']}/{os.environ['LPC_BS_DEG']}", tq=n_tq, nl_it=n_nl, tq_w=TQ_W,
                z_min=round(float(P[n0 + (ns[1] if pick else 0):n0 + n_end + 1, 2].min()), 4), x_min=round(float(P[n0:n0 + n_end + 1, 0].min()), 3))
    if pick:                                                                   # 릴리즈 상태 (실제): 발사점 위치·속력, 착지 오차
        qR, qdR = Qr[n0 + ns[2]], (Qr[n0 + ns[2] + 1] - Qr[n0 + ns[2] - 1]) / (2 * DT)
        info.update(rel_free=int(rel_free), land_cm=round(float(np.linalg.norm(np.array(F_LVEC(qR, qdR, np.asarray(start["obj"]["bin"], float))).ravel())) * 100, 2),
                    rel_tcp=[round(float(v), 3) for v in np.array(F_TCP(qR)).ravel()], rel_speed=round(float(np.linalg.norm(np.array(F_JT(qR))[:3] @ qdR)), 2))
    if pick:                                                                   # 릴리즈 창 (±RT/2) 의 실제 착지 오차·각속도 (창 항을 안 써도 보고)
        tg_ = np.asarray(start["obj"]["bin"], float); nr_ = n0 + ns[2]; k_ = int(round(WIN_RT / 2 / DT)); lo_ = max(n0 + ns[1] + 2, nr_ - k_); hi_ = min(n0 + ns[3] - 2, nr_ + k_)
        sd_ = lambda i_: (Qr[i_], (Qr[i_ + 1] - Qr[i_ - 1]) / (2 * DT))
        ew = [float(np.linalg.norm(np.array(F_LVEC(*sd_(i_), tg_)).ravel())) for i_ in range(lo_, hi_ + 1, 4)]
        ow = [float(np.linalg.norm(np.array(F_W(*sd_(i_))).ravel())) for i_ in range(lo_, hi_ + 1, 4)]
        ca_, co_ = wcost(ns, u)
        info.update(win=int(WIN_ON), win_span_ms=[int((lo_ - nr_) * DT * 1e3), int((hi_ - nr_) * DT * 1e3)], win_land_cm_mean=round(float(np.mean(ew)) * 100, 1), win_land_cm_max=round(float(np.max(ew)) * 100, 1),
                    win_omega_mean=round(float(np.mean(ow)), 2), win_omega_max=round(float(np.max(ow)), 2), rel_omega=round(float(np.linalg.norm(np.array(F_W(*sd_(nr_))).ravel())), 2),
                    J_time=round(WIN_W1 * ns[OBJ] * DT, 2), J_land=round(ca_, 2), J_omega=round(co_, 2), win_it=n_ref)
    if TCP_MODE:                                                               # 비선형 조건의 실제 잔차 (선형화가 아니라 진짜 FK 로)
        Qs = Qr[n0:]; res_p = 0.0; res_a = 0.0
        for i_, n_ in enumerate(ns):
            e_ = mk[i_](n_)
            if e_ is not None and e_.get("tcp"):
                res_p = max(res_p, float(np.linalg.norm(np.array(F_TCP(Qs[n_])).ravel() - e_["p"]))); res_a = max(res_a, float(np.linalg.norm(np.array(F_AX(Qs[n_])).ravel()[:2])))
        info.update(tcp=1, tcp_pos_err_mm=round(res_p * 1e3, 2), tcp_tilt_mrad=round(res_a * 1e3, 2),
                    j4_deg=[round(float(np.degrees(Qs[:n_end + 1, 3].min())), 1), round(float(np.degrees(Qs[:n_end + 1, 3].max())), 1)])
    info.update(slp_rs=n_rs, slp_rs_log=rs_log)
    return dict(ok=True, U=U, suction=suction, info=info)


def plan_ms(start, end, hist, ctx):
    """다중 초기값 계획 (cold): 잡기 → 따라가기 궤적은 기준 해 (고정 시각표) 를 먼저 풀고, 초기값 3 개 —
    self (그 기준 해의 시각), nb1·nb2 (DB LPP_DB_MS 에서 같은 빈의 가장 가까운 2 칸의 시각) — 를 따로 (프로세스 3 개로 동시에) 더 풀어
    목적값 (시간 + 창 착지 + 창 각속도) 이 가장 작은 해를 돌려준다. 다른 종류의 궤적은 plan 그대로."""
    t0 = time.time(); r0 = plan(start, end, hist, ctx)
    if not (start.get("kind") == "pick" and end.get("kind") == "track" and ctx.get("mode") == "cold"):
        return r0
    import pickle
    J_of = lambda r: float(r["info"]["J_time"] + r["info"]["J_land"] + r["info"]["J_omega"])
    st = {}
    if r0.get("ok"):
        st["self"] = [int(x) for x in r0["info"]["ns"]]
    D = _db(os.environ.get("LPP_DB_MS", os.path.join(_HERE, "logs", "v3db_win_x25.npz")))
    if D is not None:
        K, NSd = D; b = _bin_idx(start["obj"]["bin"]); m = K[:, -1].astype(int) == b; key = np.r_[start["obj"]["p"][:2], end["obj"]["p"][:2]]
        if m.any():
            d = np.sum((K[m][:, :4] - key) ** 2, axis=1); o = np.argsort(d, kind="stable")
            for i in range(min(2, len(o))):
                st[f"nb{i + 1}"] = [int(x) for x in NSd[m][o[i]]]
    procs = {}
    for name, ns0 in st.items():                                               # 초기값마다 자식 프로세스 (os.fork — 계획 워커가 daemon 이라 multiprocessing 은 못 씀)
        rd, wr = os.pipe(); pid = os.fork()
        if pid == 0:
            try:
                os.close(rd); c2 = dict(ctx); c2["ns0"] = ns0
                try:
                    r = plan(start, end, hist, c2)
                except Exception as e:                                         # noqa: BLE001
                    r = dict(ok=False, why=f"{type(e).__name__}: {str(e)[:200]}")
                with os.fdopen(wr, "wb") as fh:
                    pickle.dump(r, fh)
            finally:
                os._exit(0)
        os.close(wr); procs[name] = (pid, rd)
    best = r0 if r0.get("ok") else None; win = "seed" if best is not None else None
    tried = {"seed": ([round(r0["info"]["t_end"], 3), round(J_of(r0), 2), r0["info"]["s"]] if r0.get("ok") else None)}
    for name, (pid, rd) in procs.items():
        with os.fdopen(rd, "rb") as fh:
            data = fh.read()
        os.waitpid(pid, 0)
        try:
            r = pickle.loads(data) if data else dict(ok=False, why="자식 결과 없음")
        except Exception:                                                      # noqa: BLE001
            r = dict(ok=False, why="자식 결과 읽기 실패")
        if not r.get("ok"):
            tried[name] = None; continue
        tried[name] = [round(r["info"]["t_end"], 3), round(J_of(r), 2), r["info"]["s"]]
        if best is None or J_of(r) < J_of(best) - 1e-6:
            best, win = r, name
    if best is None:
        return r0
    best["info"].update(ms=tried, start=win, s_total=round(time.time() - t0, 1))
    return best
