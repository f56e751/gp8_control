"""점에서 점으로 가는 실제 궤적 (접근·복귀용): 시작 상태 (자세, 속도, 가속) → 끝 상태 (자세, 속도[, 가속]).
변수 = 실제 관절 궤적 (칸마다 가가속 일정) 과 걸리는 시간. 한계 = rnlp.py 와 같은 값. 조건: 실제 TCP z >= zmin, x >= 0.25.  목적 = 시간 + Wa × 가속 합."""
import numpy as np
import casadi as ca


def solve(M, lim, q0, v0, a0, q1, v1, a1=None, T0=0.8, N=40, zmin=0.02, xmin=0.25, Wa=0.05, Tmin=0.05):
    VEL, ACC, JRK, QLO, QHI = lim; n1 = N + 1
    T = ca.MX.sym("T"); Q = ca.MX.sym("Q", 6, n1); Vv = ca.MX.sym("V", 6, n1); A = ca.MX.sym("A", 6, n1); J = ca.MX.sym("J", 6, N); S = ca.MX.sym("S", 6, n1); h = T / N
    g = []; lb = []; ub = []

    def eq(e, n):
        g.append(e); lb.append(np.zeros(n)); ub.append(np.zeros(n))
    for m in range(N):
        eq(Q[:, m + 1] - (Q[:, m] + Vv[:, m] * h + A[:, m] * h ** 2 / 2 + J[:, m] * h ** 3 / 6), 6); eq(Vv[:, m + 1] - (Vv[:, m] + A[:, m] * h + J[:, m] * h ** 2 / 2), 6); eq(A[:, m + 1] - (A[:, m] + J[:, m] * h), 6)
    eq(Q[:, 0] - ca.DM(q0), 6); eq(Vv[:, 0] - ca.DM(v0), 6); eq(A[:, 0] - ca.DM(a0), 6); eq(Q[:, N] - ca.DM(q1), 6); eq(Vv[:, N] - ca.DM(v1), 6)
    if a1 is not None:
        eq(A[:, N] - ca.DM(a1), 6)
    for m in range(1, N):
        p = M.F_TCP(Q[:, m]); g += [p[2], p[0]]; lb += [np.array([zmin]), np.array([xmin])]; ub += [np.array([np.inf]), np.array([np.inf])]
    sc = ca.repmat(ca.DM(ACC), 1, n1); g += [ca.vec(S - A / sc), ca.vec(S + A / sc)]; lb += [np.zeros(6 * n1)] * 2; ub += [np.full(6 * n1, np.inf)] * 2
    Jacc = h * (ca.sum1(ca.sum2(S)) - 0.5 * ca.sum1(S[:, 0]) - 0.5 * ca.sum1(S[:, N])); w = ca.vertcat(T, ca.vec(Q), ca.vec(Vv), ca.vec(A), ca.vec(J), ca.vec(S))
    rep = lambda a_, n: np.tile(np.asarray(a_, float), n)
    lbw = np.r_[Tmin, rep(QLO, n1), rep(-VEL, n1), rep(-ACC, n1), rep(-JRK, N), np.zeros(6 * n1)]; ubw = np.r_[3.0, rep(QHI, n1), rep(VEL, n1), rep(ACC, n1), rep(JRK, N), np.full(6 * n1, 1.0 + 1e-9)]
    s = np.linspace(0, 1, n1)[:, None]; h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2      # 초기값: 3 차 다항식
    Q0 = h00 * q0 + h10 * T0 * v0 + h01 * q1 + h11 * T0 * v1; V0 = np.gradient(Q0, T0 / N, axis=0); A0 = np.gradient(V0, T0 / N, axis=0); J0 = np.diff(A0, axis=0) / (T0 / N)
    w0 = np.r_[T0, Q0.ravel(), np.clip(V0, -VEL, VEL).ravel(), np.clip(A0, -ACC, ACC).ravel(), np.clip(J0, -JRK, JRK).ravel(), (np.abs(np.clip(A0, -ACC, ACC)) / ACC + 1e-3).ravel()]
    G_ = ca.vertcat(*g); LB = np.concatenate([np.atleast_1d(x).ravel() for x in lb]); UB = np.concatenate([np.atleast_1d(x).ravel() for x in ub])
    Sol = ca.nlpsol("p2p", "ipopt", dict(x=w, f=T + Wa * Jacc, g=G_), {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 1500, "ipopt.tol": 1e-6, "ipopt.mu_strategy": "adaptive"})
    R = Sol(x0=w0, lbx=lbw, ubx=ubw, lbg=LB, ubg=UB); x = np.array(R["x"]).ravel(); gv = np.array(R["g"]).ravel(); viol = float(max(np.max(LB - gv), np.max(gv - UB), 0.0)); o = 1
    Qs = x[o:o + 6 * n1].reshape(n1, 6); o += 6 * n1; Vs = x[o:o + 6 * n1].reshape(n1, 6); o += 6 * n1; As = x[o:o + 6 * n1].reshape(n1, 6); o += 6 * n1; Js = x[o:o + 6 * N].reshape(N, 6)
    return dict(ok=bool(viol <= 1e-5), status=Sol.stats()["return_status"], viol=viol, T=float(x[0]), tt=np.linspace(0, float(x[0]), n1), Q=Qs, V=Vs, A=As, J=Js)
