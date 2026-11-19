"""ordsim: picking-order learning for the realized-trajectory cycles — flow simulator with the runtime rules of rt_cont, cycle-time model,
candidates, rollout expert and features (the existing method of gp8_control/learning: roll-K expert, rollout-gain regression, DAgger; K = 8).
 Time: object y(t) = y0 + VY t.  A cycle starts DWC after the contact with A (ts) and ends at the contact with B, T(A, B) later (cycle-time model).
 Candidate B: its contact point is inside the reach ring (r_lo <= r <= r_hi, i.e. |y| <= yr(x) in the lanes x >= r_lo) — the whole ring, no other limit.  No candidate -> throw, wait point, new start (T_END, T_RESTART).
 The same functions are used by the runtime (rt_cont) so the simulator and the runtime decide in the same way."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SPEED = float(os.environ.get("RGE_BELT_V", "0.12")); VY = -SPEED; TDW = float(os.environ.get("RT_DWC", "60")) * 0.002
R_HI = json.load(open(os.path.join(HERE, "logs", "reach_r.json")))["r_hi"]; AHEAD = float(os.environ.get("RT_AHEAD", "3.0"))
ZTOP = 0.02                  # default object top (rt_cont top, rn2 D.ZTOP)
K = 8; M_CTX = 12; N_VIS = 16; N_FEAT = 24 if os.environ.get("ORD_Z") else 22; ADV_LAMBDA = 0.2; ADV_MARGIN = 0.0        # margin 0 (user 11-12): the network's best candidate is always taken
T_END = 0.50                 # start of the last cycle -> at rest at the wait point (rt_cont logs: 0.49-0.50 s)
T_RESTART = 2.56             # at rest at the wait point -> earliest first contact of the next start (rt_cont: 0.36 + T_SEG 2.2)
T_START = 3.4                # very first start from home (rt_cont: LEAD0 1.2 + T_SEG 2.2)
yr = lambda x: np.sqrt(np.maximum(R_HI ** 2 - np.asarray(x, float) ** 2, 0.0))          # half width of the reach ring in lane x: inside <=> -yr(x) <= y <= yr(x)


class MLP:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False); n = int(z["n_layers"]); self.W = [np.asarray(z[f"W{k}"], float) for k in range(n)]; self.b = [np.asarray(z[f"b{k}"], float) for k in range(n)]
        self.m = np.asarray(z["x_mean"], float); self.s = np.where(np.asarray(z["x_std"], float) < 1e-3, 1.0, np.asarray(z["x_std"], float)); self.bin_code = np.asarray(z["bin_code"], float) if "bin_code" in z.files else None

    def scores(self, F):
        h = (np.asarray(F, float) - self.m) / self.s
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if k < len(self.W) - 1:
                h = np.tanh(h)
        return h[..., 0]


class TM(MLP):                                                                 # cycle time [s] from (xa, ya, xb, yb) at the cycle start and A's bin
    def __call__(self, xa, ya, xb, yb, b, za=None, zb=None):                   # 11-18: with a 9-input model also the tops of A and B (default top when not given)
        n = len(self.m); xa, ya, xb, yb, za, zb = np.broadcast_arrays(*[np.asarray(ZTOP if v is None else v, float) for v in (xa, ya, xb, yb, za, zb)])
        X = np.zeros(xa.shape + (n,)); X[..., 0] = xa; X[..., 1] = ya; X[..., 2] = xb; X[..., 3] = yb; X[..., 4 + int(b)] = 1.0
        if n >= 9:
            X[..., 7] = za; X[..., 8] = zb
        return self.scores(X)


class Stream:
    def __init__(self, x, y0, b, ids=None, code=(1.0, 0.0, -1.0), z=None):     # z: object tops (default top when not given)
        self.z = np.full(len(x), ZTOP) if z is None else np.asarray(z, float); self.code = np.asarray(code, float); self.x = np.asarray(x, float); self.y0 = np.asarray(y0, float); self.b = np.asarray(b, int); self.n = len(self.x); self.ids = list(range(self.n)) if ids is None else list(ids)


def cands(S, tm, a, ts, idx):
    """Objects idx (visible, remaining) for the cycle that starts at ts holding a. Returns ([(j, T)] that can be touched, downstream first; [j] already past)."""
    if len(idx) == 0:
        return [], []
    idx = np.asarray(idx, int); yb = S.y0[idx] + VY * ts; T = tm(S.x[a], S.y0[a] + VY * ts, S.x[idx], yb, S.b[a], S.z[a], S.z[idx]); yc = yb + VY * T; w = yr(S.x[idx]); okm = (yc >= -w) & (yc <= w); o = np.argsort(yb)
    return [(int(idx[i]), float(T[i])) for i in o if okm[i]], [int(idx[i]) for i in o if yc[i] < -w[i]]


def first(S, t_min, idx, tm):
    """First object of a start and its contact time (>= t_min): furthest downstream inside the reach at t_min; if nobody is inside, the first to enter,
    late enough that the next two are inside at their turn. Their turns are predicted with the cycle-time model (no constant cycle time)."""
    if len(idx) == 0:
        return None
    idx = np.asarray(idx, int); y = S.y0[idx] + VY * t_min; top = yr(S.x[idx]); ins = (y >= -top) & (y <= top)
    if ins.any():
        return int(idx[ins][np.argmin(y[ins])]), float(t_min)
    up = y > top
    if not up.any():
        return None
    te = (top[up] - S.y0[idx][up]) / VY; o = np.argsort(te)[:3]; te = te[o]; c = idx[up][o]
    return int(c[0]), float(start_time([float(S.x[j]) for j in c], [float(S.y0[j]) for j in c], [int(S.b[j]) for j in c], [float(v) for v in te], tm, [float(S.z[j]) for j in c]) + 0.05)


def start_time(xs, y0s, bs, te, tm, zs=None):
    """Earliest contact time with the first object (>= its entry te[0]) such that the following objects are contacted after their own entries te[n]:
    contact n+1 = contact n + hold + cycle time predicted by the model from the positions at that cycle start."""
    tc1 = te[0]
    for _ in range(12):
        t = tc1; d = 0.0
        for n in range(1, len(xs)):
            ts = t + TDW; t = ts + float(tm(xs[n - 1], y0s[n - 1] + VY * ts, xs[n], y0s[n] + VY * ts, bs[n - 1], None if zs is None else zs[n - 1], None if zs is None else zs[n])); d = max(d, te[n] - t)
        if d <= 1e-3:
            break
        tc1 += d
    return tc1


def features(S, a, ts, vis, ok, passed, nf=None):
    """(F (K, N_FEAT), mask (K,)) for the candidates ok[:K] (downstream first). Positions at the cycle start ts."""
    F = np.zeros((K, nf or N_FEAT), np.float32); mask = np.zeros(K, bool); ps = set(passed); pool = sorted([j for j in vis if j not in ps], key=lambda j: S.y0[j]); ctx = pool[:M_CTX]
    ax, ay = float(S.x[a]), float(S.y0[a] + VY * ts); abin = float(S.code[S.b[a]]); c = ok[:K]; T1s = [T + TDW for _j, T in c]; Tmin = min(T1s); ymin = float(S.y0[c[0][0]] + VY * ts)
    for i, (j, T) in enumerate(c):
        px, py = float(S.x[j]), float(S.y0[j] + VY * ts); T1 = T1s[i]; sl = (py + float(yr(px))) / SPEED; oth = [q for q in ctx if q != j]
        sa = [((S.y0[q] + VY * ts - SPEED * T1) + float(yr(S.x[q]))) / SPEED for q in oth]; near = min((float(np.hypot(S.x[q] - px, S.y0[q] + VY * ts - py)) for q in oth), default=1.0)
        F[i] = ([1.0, T1, px, py, i / (K - 1), py - ymin, sl, px - ax, py - ay, 1.0 if S.b[j] == S.b[a] else 0.0, float(S.code[S.b[j]]), min(len(pool), N_VIS) / 10.0, min(sa, default=10.0), sum(s < 1.0 for s in sa),
                sum(s < 2.5 for s in sa), near, T1 - Tmin, sl - T1, ax, ay, abin, len(c) / K, float(S.z[j]), float(S.z[a])])[:F.shape[1]]      # 11-18: the last two columns (tops) only for networks trained with them
        mask[i] = True
    return F, mask


def roll_values(S, tm, a, ts, vis, ok):
    """For each candidate ok[:K]: take it first, then first-in-first-out over the objects visible now, to the end. Returns [(number picked, end time)]."""
    out = []
    for j, T in ok[:K]:
        rest = [q for q in vis if q != j]; n = 1; cur = j; t = ts + T + TDW; hold = True
        while rest:
            rest = [q for q in rest if S.y0[q] + VY * t >= -yr(S.x[q])]
            if not rest:
                break
            if hold:
                c, _p = cands(S, tm, cur, t, rest)
                if c:
                    cur, T2 = c[0]; rest.remove(cur); n += 1; t = t + T2 + TDW
                else:
                    hold = False; t = t + T_END
            else:
                r = first(S, t + T_RESTART, [q for q in rest if S.y0[q] + VY * (t + T_RESTART) >= -yr(S.x[q])], tm)
                if r is None:
                    break
                cur, tc = r; rest.remove(cur); n += 1; t = tc + TDW; hold = True
        out.append((n, t))
    return out


def advantage(vals):
    adv = np.full(K, np.nan, np.float32)
    for i, (n, t) in enumerate(vals):
        adv[i] = (n - vals[0][0]) - ADV_LAMBDA * (t - vals[0][1])
    return adv


def choose_fifo(S, tm, a, ts, vis, ok, passed):
    return 0


def choose_roll(S, tm, a, ts, vis, ok, passed):
    v = roll_values(S, tm, a, ts, vis, ok); return int(min(range(len(v)), key=lambda i: (-v[i][0], v[i][1])))


def make_nn(model, margin=ADV_MARGIN):
    def choose(S, tm, a, ts, vis, ok, passed):
        F, mask = features(S, a, ts, vis, ok, passed, nf=len(model.m)); s = model.scores(F); s[~mask] = -np.inf; i = int(np.argmax(s))
        return i if (i == 0 or s[i] - s[0] >= margin) else 0                    # margin 0: always the best-scored candidate
    return choose


def run(S, tm, choose, y_vis, on_decision=None):
    """One stream to the end. choose(...) -> index into ok[:K]. Returns the picked objects in order, the end time, how often the arm went to the wait point."""
    rem = set(range(S.n)); done = []; mode = "rest"; t = 0.0; a = None; n_rest = 0; started = False
    for _ in range(2000):
        if not rem:
            break
        if mode == "rest":
            t_min = t + (T_RESTART if started else T_START); rem = {j for j in rem if S.y0[j] + VY * t_min >= -yr(S.x[j])}; vis = [j for j in rem if S.y0[j] + VY * (t_min - AHEAD) <= y_vis]; r = first(S, t_min, vis, tm)
            if r is None:
                t += 0.1; continue
            a, tc = r; rem.discard(a); done.append(a); t = tc + TDW; mode = "hold"; started = True
        else:
            rem = {j for j in rem if S.y0[j] + VY * t >= -yr(S.x[j])}; vis = [j for j in rem if S.y0[j] + VY * (t - AHEAD) <= y_vis]; ok, passed = cands(S, tm, a, t, vis)
            if not ok:
                mode = "rest"; t = t + T_END; n_rest += 1; continue
            if on_decision is not None:
                on_decision(S, a, t, vis, ok, passed)
            i = choose(S, tm, a, t, vis, ok, passed); a2, T = ok[i]; rem.discard(a2); done.append(a2); a = a2; t = t + T + TDW
    return dict(done=done, n_done=len(done), t=float(t), n_rest=n_rest)
