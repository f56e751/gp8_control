"""11-06 시험: cyc2 (명령 공간 NLP, 사이클 A) 의 실제 궤적 등식을 관절 식으로 바꾼다. 실제 관절 = 명령 제어점의 선형 함수라 관절 식은 선형 제약이 된다.
수직 툴 구조 (계획 규약, kin_chk.py 로 1e-16 확인): J4 = 0, J5 = π/2 − J2 − J3, J1 = atan2(y, x), ω_z = 0 ⇔ J6 − J6_0 = J1 − J1_0.
  [0, t_s] 노드: J1 = atan2(o_i) (띠 BAND_XY/ρ), J1 속도 (띠 BAND_V/ρ), J4·J5 식·J6 식 (위치 띠 BJ, 속도 띠 BAND_W) — 선형
                 반경 ρ(q) = |o_i xy| (띠 BAND_XY), 반경 속도 (띠 BAND_V), z ≥ z_i — 비선형 (J2·J3 평면 2 링크)
  t_k: q = IK(o_i + 윗면) (J6 규칙), q̇ = 자코비안⁻¹ (벨트, ω 0) — 선형 등식 12 개
  t_c: q = IK(o_{i+1} xy, h) (J6 = j6), q̇ = q̇_벨트 + λ q̇_z (λ = ż ∈ [−v̄, −1e-3]) — 선형 등식 12 개
  [t_s, t_c]: z ≥ h (비선형, 그대로)
사용: python jcyc1.py case out.json   (이력 = real1.hist_exact, real_ref.py 와 같은 과제)"""
import os, sys, json, time
if os.environ.get("CYC4_EXT", "0") == "1":                              # A안 (11-07 사용자 "지연까지 궤적 생성"): 연장 구간까지 스플라인 — 제어점 52 (cyc6 와 같게)
    os.environ.setdefault("GP8_CYC_NCTRL", "52")
import numpy as np
import cyc2 as C
import real1 as RL
from cyc2 import CY, T, SG, DT, D, H, NP_PATH, NP_HI, BAND_XY, BAND_V, BAND_W
from gp8_control.grasping.workspace_gate import tcp_batch

BJ = float(os.environ.get("CYC4_BJ", "0.003"))                        # 자세 관절 식 위치 띠 (rad)
ELIM = os.environ.get("CYC4_ELIM", "0") == "1"                         # 1: 접촉·끝 선형 등식 24 개를 대입 (관절마다 제어점 4 개 = 나머지 + λ 의 선형식)
CLOSED = os.environ.get("CYC4_CLOSED", "1") == "1"                     # 1: 창 비행시간 τ 를 닫힌 식으로 (항력 0) — τ 변수·z 등식 없음 / 0: τ 변수 + RK4
ZLIN = os.environ.get("CYC4_ZLIN", "1") == "1"                         # 1: [0, t_s] 의 z ≥ z_i 를 관절 선형식 ℓ·(J2, J3) ≥ ℓ·IK(o_i, z_i) 로 (ℓ = 반경 고정 곡선의 z 증가 방향)
ZH = os.environ.get("CYC4_ZH", "lin")                                  # [t_s, t_c] 의 z ≥ h — 0: 뺌 / 1: FK (비선형) / lin: 손목 높이 ≥ h + 툴 길이 의 (J2, J3) 접평면 (선형)
RLIN = os.environ.get("CYC4_RLIN", "1") == "1"                        # 1: [0, t_s] 반경 ρ·ρ̇ 띠를 ∇ρ(q*) 선형식으로 (q* = IK(물체 xy, 직전 해 높이), 11-06 사용자 결정) / 0: FK
LANDEQ = os.environ.get("CYC4_LANDEQ", "0") == "1"                    # 1: 창 노드마다 착지 xy 오차 = 0 등식 (2 개씩), 목적 = W1·t_c 만 (11-06 사용자 결정)
LANDLIN = os.environ.get("CYC4_LANDLIN", "0") == "1"                  # 1: 위 등식을 기준 해 (직전 해) 에서 1 차 근사한 선형 등식으로, 풀 때마다 다시 선형화
NREF = int(os.environ.get("CYC4_NREF", "6"))                           # 다시 선형화 최대 횟수 (반경·착지)
LAZY = os.environ.get("CYC4_LAZY", "1") == "1"                        # 1: θ 단계 (웜) 풀이는 재선형화 없이 한 번 — 끝에서 실제 반경 띠 확인, 넘으면 그때 다시 풂
WPROX = float(os.environ.get("CYC4_WPROX", "1.0"))                     # LANDLIN 2 단계 ‖P − P_ref‖² 가중치 (rad²)
OMEGA = os.environ.get("CYC4_OMEGA", "0") == "1"                      # 1: 창 W_OMEGA·|ω|² 벌점 (j0~j5 는 이 상태) / 0: 뺌 (11-06 사용자 결정, 기본)
PEXACT = os.environ.get("CYC4_PEXACT", "0") == "1"                    # 1: [0, t_s] 반경·반경 속도·z ≥ z_i 를 (J2, J3, J5) 닫힌 식으로 (J4 = 0 에서 FK 와 같음, p_chk.py) — 재선형화 없음 (11-07)
RLIN = RLIN and not PEXACT
TCP = os.environ.get("CYC4_TCP", "0") == "1"                         # 1: 비교용 (11-07 사용자 "비교해봐") — 집기·접촉·이동 높이·끝 조건을 모두 TCP (FK) 식으로 (cyc2 와 같은 꼴), 목적·변수·명령 한계는 같음
PEXACT = PEXACT and not TCP; RLIN = RLIN and not TCP
ZQ2 = os.environ.get("CYC4_ZQ2", "0") == "1"                         # 1 (PEXACT 와): z ≥ z_i 를 J2 상한 J2 ≤ J2_i(ρ_m) 로 (반경 고정이면 z 는 J2 에 단조, 11-07)
VB = C.VB
EXT_ON = os.environ.get("CYC4_EXT", "0") == "1"                          # 1: 명령을 t_c 뒤 EXT 만큼 더 계획 (보내지 않음). 실제 (t_c, t_c+EXT] 에 다음 물체 i+1 하강 조건
EXT = float(os.environ.get("CYC4_EXT_S", str(round(C.MEM - D + 0.12, 3)))) if EXT_ON else 0.0   #   (관절식, 띠 EXT_BF 배) + 끝 t_c+EXT 접촉 (관절 등식) — 다음 사이클이 그 명령에서 풀린다는 보증
EXT_BF = float(os.environ.get("CYC4_EXT_BAND", "0.5")); NP_EXT = 24
EXT_REST = EXT_ON and os.environ.get("CYC4_EXT_REST", "1") == "1"       # 1: 연장 끝 명령 = 정지 (마지막 제어점 3 개 = 접촉 자세 IK(다음 물체 xy@t_c+EXT, z_n)) — 연장 명령을 그대로 보내고
                                                                       #    마지막 명령을 유지해도 실제가 물체 위에서 감속해 선다 (앱 자리표시 안전, 11-07 게이트 거부 x 0.166·z −0.10)


def Tsp(t_c):
    """스플라인 구간 길이 (실제 t_c + 연장)."""
    return float(t_c) + EXT


def herm_(s, T_, p0, v0, p1, v1):
    u = s / T_; h00 = 2*u**3 - 3*u**2 + 1; h10 = u**3 - 2*u**2 + u; h01 = -2*u**3 + 3*u**2; h11 = u**3 - u**2
    return h00 * p0 + h10 * T_ * v0 + h01 * p1 + h11 * T_ * v1
_CH = T._CHAIN; A1, D1, A2, A3, D4, LT = _CH[1][0][0], _CH[1][0][2], _CH[2][0][2], _CH[3][0][2], _CH[4][0][0], _CH[5][0][0]


def pick_rz(q, qd):
    """J4 = 0 의 TCP 반경·반경 속도·높이 (J2, J3, J5 만): φ = J2 + J3, ψ = φ + J5.
    ρ = a1 + a2 sin J2 + a3 sin φ + d4 cos φ + L cos ψ,  z = d1 + a2 cos J2 + a3 cos φ − d4 sin φ − L sin ψ."""
    import casadi as cs
    ph = q[1] + q[2]; ps = ph + q[4]; phd = qd[1] + qd[2]; psd = phd + qd[4]
    rho = A1 + A2 * cs.sin(q[1]) + A3 * cs.sin(ph) + D4 * cs.cos(ph) + LT * cs.cos(ps)
    rhod = A2 * cs.cos(q[1]) * qd[1] + (A3 * cs.cos(ph) - D4 * cs.sin(ph)) * phd - LT * cs.sin(ps) * psd
    z = D1 + A2 * cs.cos(q[1]) + A3 * cs.cos(ph) - D4 * cs.sin(ph) - LT * cs.sin(ps)
    return rho, rhod, z


def build():
    import casadi as cs
    c = CY._consts(); NW = T.N_WIN
    HW_LO = np.maximum(T.GP8_Q_MIN, T.Q_LO_REAL) + CY.Q_MARGIN; HW_HI = np.minimum(T.GP8_Q_MAX, T.Q_HI_REAL) - CY.Q_MARGIN
    o = cs.Opti(); lam_z = o.variable()
    par = {k: o.parameter(*((n,) if isinstance(n, int) else n)) for k, n in dict(
        q0=6, v0=6, t_c=1, wpen=1, wprox=1, Pref=(6, N - 1), tgt=3, j6=1, zi=1, zh=1, vbar=1, d16=1,
        Aw=(N, NW), Adw=(N, NW), hw=(6, NW), hdw=(6, NW),
        Ap=(N, NP_PATH), Apd=(N, NP_PATH), hp=(6, NP_PATH), hpd=(6, NP_PATH),
        J1s=NP_PATH, J1d=NP_PATH, RHO=NP_PATH, RHOd=NP_PATH, Lr=(6, NP_PATH), cr=NP_PATH, Lz=(2, NP_PATH), cz=NP_PATH, Lh=(2, NP_HI), ch=NP_HI,
        Ah=(N, NP_HI), hh=(6, NP_HI), Gq=(6, 2 * NW), Gd=(6, 2 * NW), cl=2 * NW,
        Ak=N, Akd=N, hk_=6, hkd=6, qk=6, qdk=6,
        Ae=N, Aed=N, he=6, hed=6, qe=6, qdb=6, qdz=6, Op=(2, NP_PATH), ok=3, oe=2,
        tcr=1, zn=1, d16n=1, Ax=(N, NP_EXT), Axd=(N, NP_EXT), hx=(6, NP_EXT), hxd=(6, NP_EXT), J1x=NP_EXT, J1dx=NP_EXT, RHOx=NP_EXT, RHOdx=NP_EXT,
        qkx=6, qdkx=6, pw=NP_PATH, qrest=6).items()}
    ex = {}
    if ELIM:                                         # Pf = [Y, Dᵀ] sel,  D = S⁻¹ (Cc + e4 λ q̇_zᵀ − RF Yᵀ) — 행 = (접촉 q, q̇, 끝 q, q̇), 관절마다 같은 행렬
        Y = o.variable(6, N - 5)
        ex = dict(Y=Y, sel=o.parameter(N - 1, N - 1), Sinv=o.parameter(4, 4), RF=o.parameter(4, N - 5), Cc=o.parameter(4, 6))
        Dd = cs.mtimes(ex["Sinv"], ex["Cc"] + cs.mtimes(cs.DM([0.0, 0.0, 0.0, 1.0]), lam_z * par["qdz"].T) - cs.mtimes(ex["RF"], Y.T))
        Pf = cs.mtimes(cs.horzcat(Y, Dd.T), ex["sel"])
    else:
        Pf = o.variable(6, N - 1)
    t_c = par["t_c"]; P = cs.horzcat(par["q0"], Pf)
    for k in range(1, N):
        o.subject_to(o.bounded(HW_LO, P[:, k], HW_HI))
    for k in range(N - 1):
        span = c["KN"][k + CY.DEG + 1] - c["KN"][k + 1]
        o.subject_to(o.bounded(-T.QD_LIM, CY.DEG * (P[:, k + 1] - P[:, k]) / span / t_c, T.QD_LIM))
    qd0 = CY.DEG * (P[:, 1] - P[:, 0]) / (c["U_HEAD"] * t_c)
    o.subject_to(o.bounded(-T.QDD_LIM * 0.002, qd0 - par["v0"], T.QDD_LIM * 0.002))
    for i in range(c["DDB"].shape[0]):
        o.subject_to(o.bounded(-T.QDD_LIM, cs.mtimes(P, c["DDB"][i]) / t_c ** 2, T.QDD_LIM))
    F_p, F_a, F_pd, _ = T.end_kinematics_functions()
    C.GROUPS.clear(); C.GROUPS.append(("cmd", 0, o.ng)); mark = [o.ng]

    def grp(name):
        C.GROUPS.append((name, mark[0], o.ng)); mark[0] = o.ng
    if TCP:                                                            # TCP 식: [0, t_s] xy = o_i, xy 속도 = 벨트, ω = 0 (띠), z ≥ z_i / 접촉 위치·속도 / 이동 z ≥ h / 끝 위치·툴축·J6·ω·속도
        vbc = cs.DM(VB)
        for m in range(NP_PATH):
            q = cs.mtimes(P, par["Ap"][:, m]) + par["hp"][:, m]; qd = cs.mtimes(P, par["Apd"][:, m]) + par["hpd"][:, m]
            p = F_p(q); v = F_pd(q, qd); w = T.fk_omega_sym(q, qd)
            o.subject_to(p[2] - par["zi"] >= 0)
            o.subject_to(o.bounded(-BAND_XY, p[0:2] - par["Op"][:, m], BAND_XY)); o.subject_to(o.bounded(-BAND_V, v[0:2] - vbc[0:2], BAND_V))
            o.subject_to(o.bounded(-BAND_W, w, BAND_W))
        grp("path_tcp")
        q = cs.mtimes(P, par["Ak"]) + par["hk_"]; qd = cs.mtimes(P, par["Akd"]) + par["hkd"]
        o.subject_to(F_p(q) - par["ok"] == 0); o.subject_to(F_pd(q, qd) - vbc == 0); grp("k")
        for m in range(NP_HI):
            o.subject_to(F_p(cs.mtimes(P, par["Ah"][:, m]) + par["hh"][:, m])[2] - par["zh"] >= 0)
        grp("zh")
        q = cs.mtimes(P, par["Ae"]) + par["he"]; qd = cs.mtimes(P, par["Aed"]) + par["hed"]
        p = F_p(q); v = F_pd(q, qd); a = F_a(q)
        o.subject_to(p[0:2] - par["oe"] == 0); o.subject_to(p[2] - par["zh"] == 0)
        o.subject_to(a[0] == 0); o.subject_to(a[1] == 0); o.subject_to(a[2] <= -0.5); o.subject_to(q[5] - par["j6"] == 0)
        o.subject_to(T.fk_omega_sym(q, qd) == 0); o.subject_to(v[0:2] - vbc[0:2] == 0); o.subject_to(v[2] - lam_z == 0)
    for m in range(0 if TCP else NP_PATH):
        q = cs.mtimes(P, par["Ap"][:, m]) + par["hp"][:, m]; qd = cs.mtimes(P, par["Apd"][:, m]) + par["hpd"][:, m]
        pw = par["pw"][m]                                              # 노드 띠 배율 (기본 1, 앱 첫 사이클 완화 = 앞 r_free 구간 크게)
        b1 = pw * BAND_XY / par["RHO"][m]; b1d = pw * BAND_V / par["RHO"][m]
        o.subject_to(o.bounded(-b1, q[0] - par["J1s"][m], b1)); o.subject_to(o.bounded(-b1d, qd[0] - par["J1d"][m], b1d))
        for e, ed in ((q[3], qd[3]), (q[4] + q[1] + q[2] - np.pi / 2, qd[4] + qd[1] + qd[2]), (q[5] - q[0] - par["d16"], qd[5] - qd[0])):
            o.subject_to(o.bounded(-pw * BJ, e, pw * BJ)); o.subject_to(o.bounded(-pw * BAND_W, ed, pw * BAND_W))
    grp("path_lin")
    for m in range(0 if TCP else NP_PATH):
        q = cs.mtimes(P, par["Ap"][:, m]) + par["hp"][:, m]; qd = cs.mtimes(P, par["Apd"][:, m]) + par["hpd"][:, m]
        if PEXACT:
            rho, rhod, z = pick_rz(q, qd)
            pw = par["pw"][m]
            o.subject_to(o.bounded(-pw * BAND_XY, rho - par["RHO"][m], pw * BAND_XY)); o.subject_to(o.bounded(-pw * BAND_V, rhod - par["RHOd"][m], pw * BAND_V))
            o.subject_to(((par["cz"][m] - q[1]) if ZQ2 else (z - par["zi"])) >= 0); continue
        if RLIN:                                                       # ρ ≈ ρ* + g·(q − q*), ρ̇ = g·q̇ (g = ∇ρ(q*), q* = IK(물체 xy, 기준 높이))
            o.subject_to(o.bounded(-BAND_XY, cs.dot(par["Lr"][:, m], q) - par["cr"][m], BAND_XY))
            o.subject_to(o.bounded(-BAND_V, cs.dot(par["Lr"][:, m], qd) - par["RHOd"][m], BAND_V))
        else:
            p = F_p(q); v = F_pd(q, qd); rho = cs.sqrt(p[0] ** 2 + p[1] ** 2)
            o.subject_to(o.bounded(-BAND_XY, rho - par["RHO"][m], BAND_XY))
            o.subject_to(o.bounded(-BAND_V, (p[0] * v[0] + p[1] * v[1]) / rho - par["RHOd"][m], BAND_V))
        if ZLIN:
            o.subject_to(par["Lz"][0, m] * q[1] + par["Lz"][1, m] * q[2] - par["cz"][m] >= 0)
        else:
            o.subject_to(p[2] - par["zi"] >= 0)
    grp("path_nl")
    # 접촉 t_k — 관절 등식
    if not ELIM and not TCP:
        o.subject_to(cs.mtimes(P, par["Ak"]) + par["hk_"] - par["qk"] == 0); o.subject_to(cs.mtimes(P, par["Akd"]) + par["hkd"] - par["qdk"] == 0)
    grp("k")
    # [t_s, t_c] 노드: z ≥ h
    for m in range(NP_HI if (ZH != "0" and not TCP) else 0):
        qh = cs.mtimes(P, par["Ah"][:, m]) + par["hh"][:, m]
        if ZH == "lin":
            o.subject_to(par["Lh"][0, m] * qh[1] + par["Lh"][1, m] * qh[2] - par["ch"][m] >= 0)
        else:
            o.subject_to(F_p(qh)[2] - par["zh"] >= 0)
    grp("zh")
    # 끝 t_c — 관절 등식 (ż = λ)
    if not ELIM and not TCP:
        o.subject_to(cs.mtimes(P, par["Ae"]) + par["he"] - par["qe"] == 0)
        o.subject_to(cs.mtimes(P, par["Aed"]) + par["hed"] - par["qdb"] - lam_z * par["qdz"] == 0)
    o.subject_to(o.bounded(-par["vbar"], lam_z, -1e-3)); grp("end")
    if EXT_ON:                                                         # 연장 (t_c, t_c+EXT]: 다음 물체 위 하강 (관절식, 띠 EXT_BF 배) + 끝 접촉 (관절 등식)
        for m in range(NP_EXT):
            q = cs.mtimes(P, par["Ax"][:, m]) + par["hx"][:, m]; qd = cs.mtimes(P, par["Axd"][:, m]) + par["hxd"][:, m]
            b1 = EXT_BF * BAND_XY / par["RHOx"][m]; b1d = EXT_BF * BAND_V / par["RHOx"][m]
            o.subject_to(o.bounded(-b1, q[0] - par["J1x"][m], b1)); o.subject_to(o.bounded(-b1d, qd[0] - par["J1dx"][m], b1d))
            for e, ed in ((q[3], qd[3]), (q[4] + q[1] + q[2] - np.pi / 2, qd[4] + qd[1] + qd[2]), (q[5] - q[0] - par["d16n"], qd[5] - qd[0])):
                o.subject_to(o.bounded(-EXT_BF * BJ, e, EXT_BF * BJ)); o.subject_to(o.bounded(-EXT_BF * BAND_W, ed, EXT_BF * BAND_W))
            rho, rhod, z = pick_rz(q, qd)
            o.subject_to(o.bounded(-EXT_BF * BAND_XY, rho - par["RHOx"][m], EXT_BF * BAND_XY))
            o.subject_to(o.bounded(-EXT_BF * BAND_V, rhod - par["RHOdx"][m], EXT_BF * BAND_V)); o.subject_to(z - par["zn"] >= 0)
        o.subject_to(cs.mtimes(P, par["Ax"][:, NP_EXT - 1]) + par["hx"][:, NP_EXT - 1] - par["qkx"] == 0)
        o.subject_to(cs.mtimes(P, par["Axd"][:, NP_EXT - 1]) + par["hxd"][:, NP_EXT - 1] - par["qdkx"] == 0)
        if EXT_REST:                                                   # 끝 명령 u(T) = q_rest, u̇(T) = ü(T) = 0 (끝 매듭 4 중 → 마지막 제어점 3 개 같음), 선형
            for k_ in (1, 2, 3):
                o.subject_to(P[:, N - k_] - par["qrest"] == 0)
        grp("ext")
    mark[0] = o.ng
    F_err, F_z, _ = T.make_objective_functions_sym(); dragc = cs.DM(np.zeros(3))
    w_tr = np.full(NW, c["RT"] / (NW - 1)); w_tr[[0, -1]] *= 0.5
    J_win = 0; taus = []; F_c = landing_closed(); F_r = landing_vec()
    for m in range(NW):
        q = cs.mtimes(P, par["Aw"][:, m]) + par["hw"][:, m]; qd = cs.mtimes(P, par["Adw"][:, m]) + par["hdw"][:, m]
        if LANDLIN:                                                    # Gqᵀ q + Gdᵀ q̇ = cl  (= ∂r/∂q q_r + ∂r/∂q̇ q̇_r − r_r)
            sl = slice(2 * m, 2 * m + 2)
            o.subject_to(cs.mtimes(par["Gq"][:, sl].T, q) + cs.mtimes(par["Gd"][:, sl].T, qd) - par["cl"][sl] == 0)   # 1 단계: 행 0 (0 = 0) + 벌점
        elif LANDEQ:
            o.subject_to(F_r(q, qd, par["tgt"]) == 0); continue
        if CLOSED:
            e_land = F_c(q, qd, par["tgt"])[0]
        else:
            tau = o.variable(); taus.append(tau)
            o.subject_to(o.bounded(0.05, tau, 1.5)); o.subject_to(F_z(q, qd, tau, par["tgt"], dragc) == 0)
            e_land = F_err(q, qd, tau, par["tgt"], dragc)
        J_win = J_win + w_tr[m] * (T.W_ACC * e_land + (T.W_OMEGA_LADDER[0] * cs.sumsqr(T.fk_omega_sym(q, qd)) if OMEGA else 0))
    J = T.W1 * (par["tcr"] if EXT_ON else t_c) + T.W2 * par["wpen"] * J_win                       # wpen: 착지 벌점 켜기 (LANDLIN 2 단계 = 0)
    o.minimize(J + par["wprox"] * cs.sumsqr(Pf - par["Pref"]))         # LANDLIN 2 단계: 기준 해에서 최소 변화 (선형화가 맞는 범위에 머물게, J 에는 안 넣음)
    ip = {"max_iter": 3000, "print_level": 0, "sb": "yes", "tol": 1e-6, "acceptable_tol": 1e-4, "acceptable_iter": 10,
          "mu_strategy": "adaptive", "warm_start_init_point": "yes", "warm_start_bound_push": 1e-9,
          "warm_start_mult_bound_push": 1e-9, "warm_start_slack_bound_push": 1e-9}
    so = {"print_time": False, "expand": True}; cb = None
    if os.environ.get("GP8_JC_DEADLINE", "1") == "1":                       # 11-08 실시간: 앱 마감 (throw_nlp.SOLVE_DEADLINE) 이 지나면 그 반복에서 멈춤 (DB 빌드: 마감 None)
        cb = T._DeadlineStop(o.nx, o.ng); so["iteration_callback"] = cb     # 참조를 돌려주는 dict 에 둔다 — 없으면 GC 돼 casadi 가 매 반복 '멈춤' 으로 봄 (11-08)
    o.solver("ipopt", so, ip)
    Lf = cs.Function("L", [o.x, o.p, o.lam_g], [o.f + cs.dot(o.lam_g, o.g)])
    return dict(opti=o, Pf=Pf, taus=taus, lam_z=lam_z, par=par, J=J, J_win=J_win, L=Lf, cb=cb, **ex)


def landing_closed():
    """항력 0 포물선 닫힌 식 (throw_nlp 의 발사점·RK4 와 같은 모델): τ = (v_z + √(v_z² + 2g(p_z − z_t)))/g, 착지 xy = p_xy + v_xy τ.
    리턴 Function (q, q̇, tgt) → (‖xy − tgt_xy‖², τ)."""
    if "Fc" not in _K:
        import casadi as cs
        q = cs.MX.sym("q", 6); qd = cs.MX.sym("qd", 6); tgt = cs.MX.sym("tgt", 3)
        pts = T.fk_col_points_sym(q); wrist, tcp = pts[0], pts[3]
        p = tcp + (T.GRIP_OFF / T._CHAIN[5][0][0]) * (tcp - wrist); v = cs.jtimes(p, q, qd)
        tau = (v[2] + cs.sqrt(cs.fmax(v[2] ** 2 + 2 * T.G * (p[2] - tgt[2]), 1e-9))) / T.G
        _K["Fc"] = cs.Function("Fc", [q, qd, tgt], [cs.sumsqr(p[0:2] + v[0:2] * tau - tgt[0:2]), tau])
    return _K["Fc"]


def landing_vec():
    """landing_closed 와 같은 식의 착지 오차 벡터 (q, q̇, tgt) → p_xy + v_xy τ − tgt_xy (2) — 등식 제약용 (제곱합 = 0 은 해에서 기울기 0)."""
    if "Fr" not in _K:
        import casadi as cs
        q = cs.MX.sym("q", 6); qd = cs.MX.sym("qd", 6); tgt = cs.MX.sym("tgt", 3)
        pts = T.fk_col_points_sym(q); wrist, tcp = pts[0], pts[3]
        p = tcp + (T.GRIP_OFF / T._CHAIN[5][0][0]) * (tcp - wrist); v = cs.jtimes(p, q, qd)
        tau = (v[2] + cs.sqrt(cs.fmax(v[2] ** 2 + 2 * T.G * (p[2] - tgt[2]), 1e-9))) / T.G
        _K["Fr"] = cs.Function("Fr", [q, qd, tgt], [p[0:2] + v[0:2] * tau - tgt[0:2]])
    return _K["Fr"]


def closed_check(res, task, hist):
    """해의 창 노드에서 닫힌 식 τ 를 기존 RK4 식에 넣어: z 잔차 (0 이어야), 착지 오차 차이 (0 이어야), τ 범위."""
    c = CY._consts(); t_c, t_st = (float(v) for v in res["theta"][:2])
    Aw, Adw, hw, hdw = CY.weights(Tsp(t_c), t_st + c["OFFS"] + D, hist); P = np.asarray(res["P"])
    F_err, F_z, _ = T.make_objective_functions_sym(); z3 = np.zeros(3); F_c = landing_closed(); zr = []; de = []; ts = []
    for m in range(Aw.shape[0]):
        q = P @ Aw[m] + hw[:, m]; qd = P @ Adw[m] + hdw[:, m]
        e, tau = (float(v) for v in F_c(q, qd, task["tgt"]))
        zr.append(abs(float(F_z(q, qd, tau, task["tgt"], z3)))); de.append(abs(float(F_err(q, qd, tau, task["tgt"], z3)) - e)); ts.append(tau)
    return dict(closed_z_res_max=max(zr), closed_err_diff_max=max(de), tau_min=min(ts), tau_max=max(ts))


def zlin_rows(xy, zi, dz=0.005):
    """노드 xy 마다 ℓ = IK(z_i + dz) − IK(z_i) 의 (J2, J3) 방향 (단위), c = ℓ·IK(z_i)[J2, J3] — z ≥ z_i ⇔ ℓ·(J2, J3) ≥ c (반경 고정 곡선 위)."""
    L = np.zeros((2, len(xy))); c = np.zeros(len(xy))
    for m, (x_, y_) in enumerate(xy):
        Ja = ik_pl([x_, y_, zi])[1:3]; Jb = ik_pl([x_, y_, zi + dz])[1:3]
        l = (Jb - Ja) / np.linalg.norm(Jb - Ja); L[:, m] = l; c[m] = l @ Ja
    return L, c


def rlin_rows(xy, zr):
    """노드마다 q* = IK(물체 xy, 기준 높이 zr) (수직 툴), g = ∇ρ_TCP(q*) (관절 6 개), c = g·q* — |ρ − ρ*| ≤ b ⇔ |g·q − c| ≤ b (1 차)."""
    if "Frho" not in _K:
        import casadi as cs
        q = cs.MX.sym("q", 6); p = T.end_kinematics_functions()[0](q); rho = cs.sqrt(p[0] ** 2 + p[1] ** 2)
        _K["Frho"] = cs.Function("Frho", [q], [cs.gradient(rho, q)])
    L = np.zeros((6, len(xy))); c = np.zeros(len(xy))
    for m, ((x_, y_), z_) in enumerate(zip(xy, zr)):
        qs = ik_pl([x_, y_, z_]); g = np.array(_K["Frho"](qs)).ravel(); L[:, m] = g; c[m] = g @ qs
    return L, c


def land_rows(Qw, Qdw, tgt, nw):
    """창 노드마다 착지 오차 r(q, q̇) 의 기준 (Qw, Qdw: 6×NW) 1 차 근사 → (Gq, Gd (6, 2NW), cl (2NW), 기준의 실제 |r| 최대).
    기준 없음 (Qw None) 이면 0 행 (제약 0 = 0)."""
    Gq = np.zeros((6, 2 * nw)); Gd = np.zeros((6, 2 * nw)); cl = np.zeros(2 * nw)
    if Qw is None:
        return Gq, Gd, cl, np.inf
    if "Frj" not in _K:
        import casadi as cs
        q = cs.MX.sym("q", 6); qd = cs.MX.sym("qd", 6); tg = cs.MX.sym("tgt", 3); r = landing_vec()(q, qd, tg)
        _K["Frj"] = cs.Function("Frj", [q, qd, tg], [r, cs.jacobian(r, q), cs.jacobian(r, qd)])
    res = 0.0
    for m in range(nw):
        r, Jq, Jd = (np.array(v) for v in _K["Frj"](Qw[:, m], Qdw[:, m], tgt)); r = r.ravel()
        sl = slice(2 * m, 2 * m + 2); Gq[:, sl] = Jq.T; Gd[:, sl] = Jd.T; cl[sl] = Jq @ Qw[:, m] + Jd @ Qdw[:, m] - r
        res = max(res, float(np.abs(r).max()))
    return Gq, Gd, cl, res


def rho_check(P, th, task, hist):
    """경로 노드 (연속 모형) 에서 실제 FK 반경·반경 속도의 띠 초과량 최대 (m, m/s) — 0 이면 비선형 원식 만족."""
    t_c, _, t_s, _ = (float(v) for v in th); rp = np.linspace(0.0, t_s, NP_PATH + 1)[1:]
    Ap, Apd, hp, hpd = CY.weights(Tsp(t_c), rp + D, hist); P = np.asarray(P)
    xy = np.stack([task["oi"](r)[:2] for r in rp]); rs = np.hypot(xy[:, 0], xy[:, 1]); rsd = (xy[:, 0] * VB[0] + xy[:, 1] * VB[1]) / rs
    ep = 0.0; ev = 0.0
    for m in range(NP_PATH):
        q = P @ Ap[m] + hp[:, m]; qd = P @ Apd[m] + hpd[:, m]
        p = tcp_batch((q * SG)[None])[0]; v = vel_jac(q)[:3] @ qd; rho = np.hypot(p[0], p[1])
        ep = max(ep, abs(rho - rs[m]) - BAND_XY); ev = max(ev, abs((p[0] * v[0] + p[1] * v[1]) / rho - rsd[m]) - BAND_V)
    return max(ep, 0.0), max(ev, 0.0)


def zlin_check(task, th, n=8):
    """관절 선형 높이식 확인 (노드 n 개): ① 반경 고정 곡선에서 g(z) = ℓ·J(z) − c 가 z_i..h+0.1 에서 단조 증가인가
    ② 반경이 ±1 mm 벗어났을 때 g = 0 이 되는 높이와 z_i 의 차이 (= 높이식 오차)."""
    t_s = float(th[2]); rp = np.linspace(0.0, t_s, NP_PATH + 1)[1:][::NP_PATH // n]
    xy = np.stack([task["oi"](r)[:2] for r in rp]); zi = task["zi"]; L, c = zlin_rows(xy, zi)
    g = lambda m, xy_, z: float(L[:, m] @ ik_pl([xy_[0], xy_[1], z])[1:3] - c[m])
    dmin = np.inf; err = 0.0
    for m in range(len(xy)):
        gz = [g(m, xy[m], z) for z in np.arange(zi, H + 0.1, 0.005)]; dmin = min(dmin, float(np.diff(gz).min()))
        for d in (-1e-3, 1e-3):
            xy_ = xy[m] * (1 + d / np.linalg.norm(xy[m])); lo, hi = zi - 0.02, zi + 0.02
            for _ in range(40):
                mid = 0.5 * (lo + hi); lo, hi = (lo, mid) if g(m, xy_, mid) > 0 else (mid, hi)
            err = max(err, abs(0.5 * (lo + hi) - zi))
    return dict(zlin_monotone_min_step=dmin, zlin_height_err_mm_at_1mm_radial=1e3 * err)


def zh_check(res, Uh, task):
    """2 ms 실제 궤적 [t_s, t_c] 에서: 선형 손목식 여유 (m 단위, ≥ 0 이면 만족) 와 실제 TCP z − h, 둘의 최저·시각."""
    t_c, t_st, t_s, t_k = res["theta"]
    U = CY.sample(res["P"], Tsp(t_c))[:int(round(t_c / DT)) + 1] * SG
    Uf = np.vstack([Uh[:-1], U, np.repeat(U[-1:], len(C.hk) + C.ND + 5, 0)])
    qr = np.stack([np.convolve(Uf[:, j], C.hk)[:len(Uf)] for j in range(6)], -1)
    k0 = len(Uh) - 1; rr = np.arange(t_s, t_c + 1e-9, DT); Q = qr[[k0 + int(round((r + D) / DT)) for r in rr]] * SG
    L, c, gn = zh_rows(task, t_s, t_c, rr); lin = gn * (np.einsum("jm,mj->m", L, Q[:, 1:3]) - c)
    zt = tcp_batch(Q * SG)[:, 2] - task["zh"]; kl = int(np.argmin(lin)); kz = int(np.argmin(zt))
    return dict(zh_lin_margin_min=float(lin[kl]), zh_lin_tmin=float(rr[kl]), zh_true_min=float(zt[kz]), zh_true_tmin=float(rr[kz]),
                zh_lin_ok_but_true_below=float(min(0.0, zt[lin >= 0].min())) if (lin >= 0).any() else 0.0)


def z_profile(res, Uh):
    """2 ms 이산 실제 TCP z: [t_s, t_c] 최저 시각·값, 0.05 s 간격 높이 (r, z)."""
    t_c, t_st, t_s, t_k = res["theta"]
    U = CY.sample(res["P"], Tsp(t_c))[:int(round(t_c / DT)) + 1] * SG
    Uf = np.vstack([Uh[:-1], U, np.repeat(U[-1:], len(C.hk) + C.ND + 5, 0)])
    qr = np.stack([np.convolve(Uf[:, j], C.hk)[:len(Uf)] for j in range(6)], -1)
    k0 = len(Uh) - 1; rr = np.arange(0.0, t_c + 1e-9, DT); z = tcp_batch(qr[[k0 + int(round((r + D) / DT)) for r in rr]])[:, 2]
    hi = rr >= t_s; km = int(np.argmin(np.where(hi, z, np.inf)))
    return dict(z_hi_min=float(z[km]), z_hi_tmin=float(rr[km]), z_prof=[[round(float(r), 2), round(float(zz), 3)] for r, zz in zip(rr[::25], z[::25])])


def wrist_z():
    """(q) → (손목 중심 z, ∂z/∂q) — 손목 z 는 J2, J3 만의 함수."""
    if "Fwz" not in _K:
        import casadi as cs
        q = cs.MX.sym("q", 6); zw = T.fk_col_points_sym(q)[0][2]
        _K["Fwz"] = cs.Function("Fwz", [q], [zw, cs.gradient(zw, q)])
    return _K["Fwz"]


def zh_rows(task, t_s, t_c, rr):
    """[t_s, t_c] 노드마다 ℓ·(J2, J3) ≥ c: 손목 z ≥ h + L (L = 손목→TCP, 툴 수직이면 TCP z ≥ h 와 같음) 의 접평면.
    기준점 = 수직 툴, TCP 높이 h, 반경 = ρ(o_i(t_s)) → ρ(o_{i+1}(t_c)) 시간 선형 보간. 리턴 (ℓ (2, m), c (m,), |∇| (m,))."""
    ra = float(np.hypot(*task["oi"](t_s)[:2])); rb = float(np.hypot(*task["on"](t_c)[:2])); Fw = wrist_z()
    L = np.zeros((2, len(rr))); c = np.zeros(len(rr)); gn = np.zeros(len(rr))
    for m, r in enumerate(rr):
        w = (r - t_s) / max(t_c - t_s, 1e-9); qs = ik_pl([(1 - w) * ra + w * rb, 0.0, task["zh"]])
        g = np.array(Fw(qs)[1]).ravel()[1:3]; gn[m] = np.linalg.norm(g); L[:, m] = g / gn[m]; c[m] = L[:, m] @ qs[1:3]
    return L, c, gn


def pivots(Rf):
    """대입할 제어점 4 개: 접촉 행 (q, q̇) 과 끝 행 (q, q̇) 에서 각각 2×2 행렬식이 가장 큰 쌍. 한 Run 동안 고정 (변수 뜻이 θ 에 따라 바뀌지 않게)."""
    def pair(M, excl):
        n = M.shape[1]
        return list(max((abs(np.linalg.det(M[:, [a, b]])), a, b) for a in range(n) for b in range(a + 1, n) if a not in excl and b not in excl)[1:])
    piv = pair(Rf[:2], ()); piv += pair(Rf[2:], piv)
    if np.linalg.cond(Rf[:, piv]) > 1e10:
        raise ValueError(f"대입 행렬 특이 {piv}")
    return piv


N = C.N
_K = {}


def vel_jac(q):
    """[TCP 속도; ω] = Jm(q) q̇ (계획 규약)."""
    if "Jm" not in _K:
        import casadi as cs
        qs = cs.MX.sym("q", 6); qds = cs.MX.sym("qd", 6); _, _, F_pd, _ = T.end_kinematics_functions()
        _K["Jm"] = cs.Function("Jm", [qs, qds], [cs.jacobian(cs.vertcat(F_pd(qs, qds), T.fk_omega_sym(qs, qds)), qds)])
    return np.array(_K["Jm"](q, np.zeros(6)))


def ik_pl(xyz):
    q = C.prim()._pick_ik(np.asarray(xyz, float))
    if q is None:
        raise ValueError(f"IK 실패 {xyz}")
    return q * SG


class Run4(C.Run):
    def __init__(self, task, hist):
        super().__init__(task, hist)
        s = C.S(); s["opti"].set_value(s["par"]["d16"], task["qr0"][5] - task["qr0"][0]); self.piv = None

    def set_theta(self, th):
        t_c, t_st, t_s, t_k = (float(v) for v in th); c = CY._consts(); s = C.S(); o, p = s["opti"], s["par"]
        W = lambda r: CY.weights(Tsp(t_c), np.asarray(r, float) + D, self.hist)
        rp = np.linspace(0.0, t_s, NP_PATH + 1)[1:]; rh = np.linspace(t_s, t_c, NP_HI)
        Aw, Adw, hw, hdw = W(t_st + c["OFFS"]); Ap, Apd, hp, hpd = W(rp); Ah, _, hh, _ = W(rh)
        oi = self.t["oi"]
        xy = np.stack([oi(r)[:2] for r in rp]); rho = np.hypot(xy[:, 0], xy[:, 1])
        R, c4, tg = eq_rows(self.t, self.hist, th)
        vals = dict(pw=np.where(rp < float(self.t.get("r_free", 0.0)), 1e3, 1.0), t_c=Tsp(t_c), tcr=t_c, zn=float(self.t.get("zn", self.t["zi"])), Aw=Aw.T, Adw=Adw.T, hw=hw, hdw=hdw, Ap=Ap.T, Apd=Apd.T, hp=hp, hpd=hpd,
                    J1s=np.arctan2(xy[:, 1], xy[:, 0]), J1d=(xy[:, 0] * VB[1] - xy[:, 1] * VB[0]) / rho ** 2,
                    RHO=rho, RHOd=(xy[:, 0] * VB[0] + xy[:, 1] * VB[1]) / rho, Ah=Ah.T, hh=hh, **tg)
        Pr = getattr(self, "P_ref", None)
        zr = np.full(NP_PATH, self.t["zi"]) if Pr is None else tcp_batch(((Pr @ Ap.T + hp).T) * SG)[:, 2]
        self.zr_used = np.maximum(zr, self.t["zi"])
        vals["Lr"], vals["cr"] = rlin_rows(xy, self.zr_used)
        ph1 = LANDLIN and getattr(self, "phase1", False)                # LANDLIN 1 단계: 착지 벌점 (선형 행 0) / 2 단계: 선형 등식, 벌점 끔
        vals["Gq"], vals["Gd"], vals["cl"], self.land_res = land_rows(None if (Pr is None or ph1) else (Pr @ Aw.T + hw),
                                                                      None if (Pr is None or ph1) else (Pr @ Adw.T + hdw), self.t["tgt"], Aw.shape[0])
        vals["wpen"] = 0.0 if (LANDLIN and not ph1) else 1.0
        vals["wprox"] = WPROX if (LANDLIN and not ph1 and Pr is not None) else 0.0
        vals["Pref"] = np.zeros((6, N - 1)) if Pr is None else Pr[:, 1:]
        if ZLIN:
            vals["Lz"], vals["cz"] = zlin_rows(xy, self.t["zi"])
        if EXT_ON:                                                      # 연장 노드: 다음 물체 o_{i+1} (반경·J1 목표), 자세 J6 − J1 = 끝 자세 값, 끝 접촉 IK
            rx = np.linspace(t_c, t_c + EXT, NP_EXT + 1)[1:]; Ax, Axd, hx, hxd = W(rx); on = self.t["on"]
            xn = np.stack([on(r)[:2] for r in rx]); rn = np.hypot(xn[:, 0], xn[:, 1]); d16n = float(tg["qe"][5] - tg["qe"][0])
            vbn = np.asarray(self.t.get("vbn", VB), float)                  # 다음 물체 속도 (준비점 사이클 = 정지점: 0)
            zn = float(self.t.get("zn", self.t["zi"])); qkx = ik_pl(np.r_[on(t_c + EXT)[:2], zn]); qkx[5] = qkx[0] + d16n
            qrest = qkx.copy()                                              # 정지 자세 = 접촉 자세 (h 위 정지는 접촉 (z_n, ż 0) 과 0.28 s 평균으로 모순 → ext 위반 0.002~0.02, 11-07 B 실패·DB 칸 실패)
            vals.update(Ax=Ax.T, Axd=Axd.T, hx=hx, hxd=hxd, J1x=np.arctan2(xn[:, 1], xn[:, 0]), J1dx=(xn[:, 0] * vbn[1] - xn[:, 1] * vbn[0]) / rn ** 2,
                        RHOx=rn, RHOdx=(xn[:, 0] * vbn[0] + xn[:, 1] * vbn[1]) / rn, d16n=d16n, qkx=qkx, qrest=qrest,
                        qdkx=np.linalg.solve(vel_jac(qkx), np.r_[vbn, 0.0, 0.0, 0.0]))
        if TCP:
            vals.update(Op=xy.T, ok=np.r_[oi(t_k)[:2], self.t["zi"]], oe=self.t["on"](t_c)[:2])
        if PEXACT and ZQ2:                                              # J2_i(ρ_m) = IK(물체 xy, z_i) 의 J2
            vals["cz"] = np.array([ik_pl([x_, y_, self.t["zi"]])[1] for x_, y_ in xy])
        if ZH == "lin":
            vals["Lh"], vals["ch"] = zh_rows(self.t, t_s, t_c, rh)[:2]
        for k, v in vals.items():
            o.set_value(p[k], v)
        if ELIM:
            Rf = R[:, 1:]
            if self.piv is None:
                self.piv = pivots(Rf)
            F = [i for i in range(N - 1) if i not in self.piv]
            sel = np.zeros((N - 1, N - 1)); sel[np.arange(N - 1), F + self.piv] = 1.0
            for k, v in dict(Sinv=np.linalg.inv(Rf[:, self.piv]), RF=Rf[:, F], Cc=c4 - np.outer(R[:, 0], self.t["q0"]), sel=sel).items():
                o.set_value(s[k], v)

    def solve(self, th, init):
        """선형식 기준 (반경 높이, 착지 1 차 근사) = 초기값 P (콜드) / 직전 해. 풀고 난 해에서 다시 선형화해 기준 높이가 2 mm 넘게 바뀌거나
        실제 착지 오차가 0.1 mm 넘으면 웜으로 다시 (최대 NREF 회)."""
        if "P" in init:
            self.P_ref = np.asarray(init["P"], float)
        self.phase1 = LANDLIN and "P" in init                           # 콜드: 착지 벌점으로 먼저 풀어 기준을 만든다
        r = self._solve1(th, init); r["nref"] = 0; self.phase1 = False
        lazy = LAZY and "x" in init and not LANDLIN                     # 웜 (θ 단계): 직전 해 기준으로 한 번만 — 기준은 다음 풀이로 넘김
        for k in range(0 if lazy else (NREF if (RLIN or LANDLIN) else 0)):
            if not r["ok"]:
                break
            self.P_ref = np.hstack([np.asarray(self.t["q0"], float)[:, None], r["Pf"]]); zu = self.zr_used.copy()
            self.set_theta(th)
            if np.abs(self.zr_used - zu).max() < 0.002 and (not LANDLIN or self.land_res < 1e-4):
                break
            r2 = self._solve1(th, dict(x=r["x"], lam=r["lam"])); r2["it"] += r["it"]; r2["nref"] = k + 1; r = r2
        if r["ok"]:
            self.P_ref = np.hstack([np.asarray(self.t["q0"], float)[:, None], r["Pf"]])
            if LANDLIN:                                                 # 반환 해의 실제 착지 오차 — 1 mm 넘으면 수렴 실패로 (바깥 루프가 받지 않게)
                self.set_theta(th); r["land_res"] = self.land_res
                if self.land_res > 1e-3:
                    r["ok"] = False; r["status"] = f"재선형화 미수렴 (착지 {1e3 * self.land_res:.2f} mm, {r['nref']} 회)"
        return r

    def _solve1(self, th, init):
        if not ELIM or "x" in init:
            return super().solve(th, init)
        self.piv = None; self.set_theta(th); s = C.S(); o = s["opti"]          # 초기값마다 대입 제어점을 새로 고른다
        F = [i for i in range(N - 1) if i not in self.piv]
        o.set_initial(s["Y"], np.asarray(init["P"], float)[:, 1:][:, F]); o.set_initial(s["lam_z"], -0.5 * self.t["vbar"])
        for tau in s["taus"]:
            o.set_initial(tau, 0.45)
        return super().solve(th, dict(x=np.array(o.value(o.x, o.initial())).ravel(), lam=np.zeros(o.ng)))


def path_init_ext(task, hist_U, vin0, cert=None):
    """A안 초기값 (cyc6.path_init 과 같은 꼴): 실제 경로 (잡기 i → 던지기 → 다음 물체 위 hover → 연장 하강) 를 L−D 앞당긴 명령, 코드축 [0, t_c+EXT].
    cert (앞 사이클의 보증 명령, 로봇 규약, 코드축 [0, EXT]) 가 있으면 t_k = EXT 초기값을 먼저 만들고 그 구간 명령을 cert 로."""
    from scipy.interpolate import BSpline
    P_ = C.prim(); oi, on = task["oi"], task["on"]; zi = task["zi"]; zn = float(task.get("zn", zi)); out = []
    lead = C.LAG - D; q_seed = hist_U[-1]; W12 = C.MEM - D
    tks = ([EXT] if cert is not None else []) + [W12 + 0.02, W12 + 0.08]
    nvar = int(os.environ.get("CYC4_NVAR", "3"))                            # 던지기 모양 초기값 수 (DB: 1 — 해는 t_k 초기값으로만 갈렸다, 11-07 사용자 "줄여")
    for kv, (v, t_k) in enumerate([(v, tk) for tk in tks for v in [v for v in C.INIT_VARIANTS if v is not None][:nvar]]):
        t_s = t_k + W12 + 0.02; tf = float(np.clip(v["T0"], 0.7, 1.0)); t_c = t_s + tf
        try:
            rr = np.arange(0.0, t_s + 1e-9, DT)
            pts = np.array([np.r_[oi(r)[:2], C.zprof(r, t_k, t_s, zi, vin0)] for r in rr])
            Qp = C.TS._ik_path(P_.robot, pts, q_seed, P_.j6); Qp = C.yaw_fix(Qp, Qp[0, 0])
            q_h = P_._pick_ik(np.r_[on(t_c)[:2], H], q_init=Qp[-1])
            if q_h is None:
                PI_ERRS.append(f"{kv}: hover IK {np.round(on(t_c)[:2], 3).tolist()}"); continue
            Pt = T.init_ctrl_points(Qp[-1] * SG, np.asarray(v["dq_swing"], float), float(v["chi0"]), p_f=q_h * SG)
            sp = BSpline(T.KNOTS, Pt.T, T.DEGREE); n1 = int(round(tf / DT)); Qt = sp(np.arange(1, n1 + 1) / n1) * SG
            ve = 0.5 * task["vbar"]; sx = np.arange(1, int(round((EXT + lead) / DT)) + 3) * DT
            zx = np.maximum(zn + 0.005, [herm_(min(x, EXT), EXT, H, -ve, zn + 0.03, 0.0) for x in sx])
            Qe = C.TS._ik_path(P_.robot, np.array([np.r_[on(t_c + x)[:2], z] for x, z in zip(sx, zx)]), q_h, P_.j6); Qe = C.yaw_fix(Qe, q_h[0])
            Qr = np.vstack([Qp, Qt, Qe]); nl = int(round(lead / DT)); n_c = int(round((t_c + EXT) / DT))
            Qc = Qr[nl:nl + n_c + 1]; Qc[0] = hist_U[-1]
            use_cert = cert is not None and t_k == EXT
            if use_cert:
                nC = min(len(cert), len(Qc)); Qc[:nC] = cert[:nC]
            tag = ("cert" if use_cert else "cold") + str(kv)
            out.append((tag, dict(P=CY.fit(Qc * SG, t_c + EXT, hist_U[-1] * SG), theta=(t_c, t_s + float(v["chi0"]) * tf, t_s, t_k))))
        except ValueError as e_:
            PI_ERRS.append(f"{kv}: {str(e_)[:120]}")
            continue
    return out


PI_ERRS = []                                                                  # path_init_ext 실패 이유 (앱 진단용)


def eval_ext(res, Uh, task):
    """2 ms 실제 (로봇 이산 필터): 보낸 명령 [0, t_c] 만 → 집기·이동 높이·작업공간·창 착지·끝 명령 TCP 속도, 명령만 두고 멈추면 (다음 사이클 없음)
    다음 물체 xy 벗어남. 연장 명령까지 보냈다면 (다음 사이클이 보증 명령을 그대로 쓰면) → 연장 구간 다음 물체 xy·z·끝 접촉 위치·속도.
    리턴 (dict, U = 보낸 명령, cert = 연장 명령 (로봇 규약, 코드축 [0, EXT]))."""
    t_c, t_st, t_s, t_k = (float(v) for v in res["theta"]); n_c = int(round(t_c / DT))
    Ux = CY.sample(res["P"], Tsp(t_c)) * SG; U = Ux[:n_c + 1]; pad = len(C.hk) + C.ND + int(round(EXT / DT)) + 10; k0 = len(Uh) - 1

    def real(Us):
        Uf = np.vstack([Uh[:-1], Us, np.repeat(Us[-1:], pad, 0)])
        return np.stack([np.convolve(Uf[:, j], C.hk)[:len(Uf)] for j in range(6)], -1)
    qr, qg = real(U), real(Ux); kr = lambda r: k0 + int(round((r + D) / DT)); oi, on = task["oi"], task["on"]
    rr = np.arange(0.0, t_c + 1e-9, DT); pr = tcp_batch(qr[[kr(r) for r in rr]]); ip = rr < t_s
    out = dict(e_xy_path=float(np.max(np.linalg.norm(pr[ip, :2] - np.stack([oi(r)[:2] for r in rr[ip]]), axis=1))),
               z_path_min=float(pr[ip, 2].min() - task["zi"]), z_hi_min=float(pr[~ip, 2].min() - float(task.get("zh", H))), x_min=float(pr[:, 0].min()))
    F_c = landing_closed(); c = CY._consts(); land = []
    for r in t_st + np.linspace(-c["RT"] / 2, c["RT"] / 2, 41):
        k = kr(r); land.append(float(F_c(qr[k] * SG, (qr[k + 1] - qr[k - 1]) / (2 * DT) * SG, task["tgt"])[0]))
    out["land_max"] = float(np.sqrt(max(land)))
    pc = tcp_batch(U[-int(round((C.MEM - D) / DT)):]); out["tail_cmd_speed_max"] = float(np.max(np.linalg.norm(np.diff(pc, axis=0), axis=1)) / DT)
    rx = np.arange(t_c, t_c + EXT - DT / 2, DT); ph = tcp_batch(qr[[kr(r) for r in rx]]); pg = tcp_batch(qg[[kr(r) for r in rx]])
    onx = np.stack([on(r)[:2] for r in rx]); zn = float(task.get("zn", task["zi"]))
    out.update(hold_exy=float(np.max(np.linalg.norm(ph[:, :2] - onx, axis=1))), ext_exy=float(np.max(np.linalg.norm(pg[:, :2] - onx, axis=1))),
               ext_zmin=float(pg[:, 2].min() - zn))
    ke = kr(t_c + EXT); pk = tcp_batch(qg[[ke - 1, ke, ke + 1]])
    out.update(ext_contact_p=float(np.linalg.norm(pk[1] - np.r_[on(t_c + EXT)[:2], zn])),
               ext_contact_v=float(np.linalg.norm((pk[2] - pk[0]) / (2 * DT) - np.asarray(task.get("vbn", VB), float))))
    rH = np.arange(t_c + EXT, t_c + EXT + 0.30 + 1e-9, DT); pH = tcp_batch(qg[[kr(r) for r in rH]])   # 연장 명령 뒤 마지막 명령 유지 (앱 자리표시) 의 실제
    out.update(ext_x_min=float(min(pg[:, 0].min(), pH[:, 0].min())), hold_z_min=float(pH[:, 2].min() - zn), hold_end_z=float(pH[-1, 2]),
               hold_end_xy_err=float(np.linalg.norm(pH[-1, :2] - on(t_c + EXT)[:2])), cmd_end_z=float(tcp_batch(Ux[-1:])[0, 2]))
    return out, U, Ux[n_c:]


def eq_rows(task, hist, th):
    """접촉·끝 선형 등식: 관절 j 마다 R @ P[j] = c4[:, j] (+ λ q̇_z[j] 끝 속도 행). R 행 = (접촉 q, 접촉 q̇, 끝 q, 끝 q̇) 의 연속시간 가중치."""
    t_c, _, _, t_k = (float(v) for v in th)
    W = lambda r: CY.weights(Tsp(t_c), np.asarray(r, float) + D, hist)
    Ak, Akd, hk_, hkd = W([t_k]); Ae, Aed, he, hed = W([t_c])
    oi, on, qr0 = task["oi"], task["on"], task["qr0"]
    qk = ik_pl(np.r_[oi(t_k)[:2], task["zi"]]); qk[5] = qr0[5] + qk[0] - qr0[0]
    qdk = np.linalg.solve(vel_jac(qk), np.r_[VB, 0.0, 0.0, 0.0])
    qe = ik_pl(np.r_[on(t_c)[:2], task["zh"]]); qe[5] = task["j6"]; Je = vel_jac(qe); vbn = np.asarray(task.get("vbn", VB), float)
    qdb = np.linalg.solve(Je, np.r_[vbn[:2], 0.0, 0.0, 0.0, 0.0])
    qdz = np.zeros(6) if task.get("ready") else np.linalg.solve(Je, np.r_[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])   # 준비점 사이클: 끝 실제 정지 (λ 항 없음)
    R = np.vstack([Ak[0], Akd[0], Ae[0], Aed[0]])
    c4 = np.vstack([qk - hk_[:, 0], qdk - hkd[:, 0], qe - he[:, 0], qdb - hed[:, 0]])
    tg = dict(Ak=Ak[0], Akd=Akd[0], hk_=hk_[:, 0], hkd=hkd[:, 0], qk=qk, qdk=qdk, Ae=Ae[0], Aed=Aed[0], he=he[:, 0], hed=hed[:, 0],
              qe=qe, qdb=qdb, qdz=qdz)
    return R, c4, tg


def eq_residual(res, task, hist):
    """해에서 모형 (연속시간) 등식 잔차: 접촉 q·q̇, 끝 q, 끝 q̇ (λ 는 최소제곱)."""
    R, c4, tg = eq_rows(task, hist, res["theta"]); E = R @ np.asarray(res["P"]).T - c4      # (4, 6)
    lam = float(tg["qdz"] @ E[3] / (tg["qdz"] @ tg["qdz"]))
    return dict(eq_res_contact=float(np.abs(E[:2]).max()), eq_res_end_q=float(np.abs(E[2]).max()),
                eq_res_end_qd=float(np.abs(E[3] - lam * tg["qdz"]).max()), end_zdot_model=lam)


def joint_check(res, Uh, task):
    """2 ms 이산 실제 궤적에서 관절 식·툴축 기울기 (경로 [0, t_s])."""
    t_c, t_st, t_s, t_k = res["theta"]
    U = CY.sample(res["P"], Tsp(t_c))[:int(round(t_c / DT)) + 1] * SG
    Uf = np.vstack([Uh[:-1], U, np.repeat(U[-1:], len(C.hk) + C.ND + 5, 0)])
    qr = np.stack([np.convolve(Uf[:, j], C.hk)[:len(Uf)] for j in range(6)], -1) * SG
    k0 = len(Uh) - 1; ks = [k0 + int(round((r + D) / DT)) for r in np.arange(0, t_s, DT)]
    Q = qr[ks]; qr0 = task["qr0"]
    _, F_a, _, _ = T.end_kinematics_functions()
    tilt = max(float(np.degrees(np.arccos(min(1.0, -float(np.array(F_a(q)).ravel()[2]))))) for q in Q)
    return dict(path_J4_max=float(np.abs(Q[:, 3]).max()), path_J5rel_max=float(np.abs(Q[:, 4] + Q[:, 1] + Q[:, 2] - np.pi / 2).max()),
                path_J6rel_max=float(np.abs(Q[:, 5] - Q[:, 0] - (qr0[5] - qr0[0])).max()), path_tilt_deg_max=tilt)


if __name__ == "__main__":
    ci = int(sys.argv[1]); out = sys.argv[2]
    (xi, yi), (xn, yn), b, _, _ = C.CASES[ci]
    oi = C.moving([xi, yi, C.ZI]); on = C.moving([xn, yn, C.ZI]); vin0 = 0.6 * C.vbar_of((xi, yi), C.ZI)
    Uh = RL.hist_exact(oi, vin0)
    task, hd = C.make_task(Uh, oi, on, b, C.ZI, C.ZI)
    task["qr0"] = RL.hist_realized(Uh)[2][-1] * SG                          # 실제 r = 0 (계획 규약)
    C._S["S"] = build(); C.Run = Run4
    inits = C.path_init(task, Uh, vin0); n_init = len(inits)
    if os.environ.get("CYC4_INIT"):                                     # 콜드 초기값 하나만 (k 번째) — 6 개 비교용 (11-07)
        inits = [inits[int(os.environ["CYC4_INIT"])]]
    t0 = time.time(); r = C.optimize(task, hd, inits); fix = {}
    if r["ok"] and RLIN and LAZY:                                       # 끝 확인: 실제 반경 띠 초과 (0.05 mm, 0.5 mm/s 허용) → 최종 해 기준으로 다시 풂
        ep, ev = rho_check(r["P"], r["theta"], task, hd); fix = dict(rho_viol_mm=1e3 * ep, rhod_viol_mm_s=1e3 * ev, final_fix=False)
        if ep > 5e-5 or ev > 5e-4:
            R2 = Run4(task, hd); r2 = R2.solve(r["theta"], dict(P=r["P"]))
            if r2["ok"]:
                r.update(P=np.hstack([np.asarray(task["q0"], float)[:, None], r2["Pf"]]), J=r2["J"]); r["it"] += r2["it"]; r["n_solve"] += R2.n_solve
                ep, ev = rho_check(r["P"], r["theta"], task, hd); fix.update(final_fix=True, rho_viol_mm_after=1e3 * ep, rhod_viol_mm_s_after=1e3 * ev)
    if r["ok"] and PEXACT:
        ep, ev = rho_check(r["P"], r["theta"], task, hd); fix = dict(rho_viol_mm=1e3 * ep, rhod_viol_mm_s=1e3 * ev)
    log = dict(case=ci, n_init=n_init, init_sel=os.environ.get("CYC4_INIT"), BJ=BJ, ELIM=ELIM, CLOSED=CLOSED, OMEGA=OMEGA, TCP=TCP, PEXACT=PEXACT, ZQ2=ZQ2, RLIN=RLIN, LANDEQ=LANDEQ, LANDLIN=LANDLIN, LAZY=LAZY, s=time.time() - t0, ok=r["ok"], **fix)
    if r["ok"]:
        e = C.evaluate(r, Uh, task); e.pop("U"); log.update(e); log.update(joint_check(r, Uh, task)); log["n_solve"] = r["n_solve"]
        log.update(eq_residual(r, task, hd)); log.update(closed_check(r, task, hd)); log["init"] = r["tag"]
        log.update(ZLIN=ZLIN, ZH=ZH); log.update(zh_check(r, Uh, task)); log.update(zlin_check(task, r["theta"]) if ZLIN else {})
        log.update(z_profile(r, Uh)); np.savez(out.replace(".json", ".npz"), P=r["P"], theta=r["theta"])
    else:
        log["why"] = r["why"]
    json.dump(log, open(out, "w"), ensure_ascii=False, indent=1); print(json.dumps(log, ensure_ascii=False))
