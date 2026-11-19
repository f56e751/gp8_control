"""11-11 사용자: "v_in 을 항상 상한으로 … 이전 명령은 v_in 수직 하향이었다고 가정하고 DB 를 새 출발 설정에서 추가로 만들어봐".
칸 = (잡는 물체 x, y, 다음 물체 x, y, 빈) — db_win.py 와 같은 492 칸. 다른 점:
 - 출발: 명령이 물체 i 위 수직선을 v_in 상한 (logs/vin_table.json, 반경 5 cm 칸) 으로 내려오며 높이 h 를 지나는 순간. 그 전 명령 = 같은 속력의 수직 직선 (가정).
 - 자르는 곳: 명령이 다음 물체 위 수직선을 v_in 상한으로 내려오며 h 를 지나는 순간 (명령 조건). 그 뒤 실제 hover·접촉은 내다보기.
 - 접촉·상승 끝·릴리즈·높이/x 하한·릴리즈 창·목적 (다음 물체 접촉 시각) 은 그대로.
사건 시각 ns = [접촉, 상승 끝, 릴리즈, 명령 경계 (자르는 곳), 다음 물체 위 실제 도착, 다음 물체 접촉].
사용: python db_vin.py out.npz 워커   (DB_SPLIT="a-b/n", DB_REVERSE=1, DB_KEYS="x,y,x,y,b;…" 로 몇 칸만, DB_NS0=json (칸 → 이웃 해의 시각표·명령) 은 db_vin_ns0.py)"""
import os, sys, time, json
os.environ["LPP_VIN"] = "1"
import numpy as np
import multiprocessing as mp
import db_win as D                                                              # 설정 (LPP_TCP·RELFREE·WIN), 칸 선별, 계획기

M = D.M; DT = D.DT; H = float(D.H); VB = D.VB; SG = D.SG; NH = len(M.hk) + 4
kt = lambda k: tuple([round(float(v), 4) for v in k[:4]] + [int(k[4])])


def start_hist(xi, yi):
    """출발 이력: 명령이 (xi, yi) 물체 위 수직선을 v_in 상한으로 내려와 지금 (t = 0) 높이 h. 필터 기억 + 4 표본."""
    v0 = M._vin(float(np.hypot(xi, yi))); Q = [M.vin_line((xi, yi), v0, H, k) for k in range(NH - 1, -1, -1)]
    return v0, (None if any(q is None for q in Q) else np.array(Q) * SG)


def run_cell(key, ns0=None, u0=None):
    xi, yi, xn, yn, b = key; b = int(b); vy = float(VB[1]); t0 = time.time()
    try:
        v0, hist = start_hist(xi, yi)
        if hist is None:
            return dict(key=key, ok=False, why="출발 직선 도달 불가", s=time.time() - t0)
        oi = dict(id=1, cls="x", p=[xi, yi, D.ZTOP], v=[0, vy, 0], bin=list(D.BINS[b])); on = dict(id=2, cls="x", p=[xn, yn, D.ZTOP], v=[0, vy, 0], bin=list(D.BINS[0]))
        ctx = dict(mode="cold", deadline=None, dt=DT, h=H)
        if ns0 is not None:
            ctx["ns0"] = [int(x) for x in ns0]
        if u0 is not None:
            ctx["u0"] = u0
        r = D.plan(dict(kind="pick", state="vin", obj=oi), dict(kind="track", obj=on), hist, ctx)
    except Exception as e:                                                    # noqa: BLE001
        return dict(key=key, ok=False, why=f"{type(e).__name__}: {e}"[:200], s=time.time() - t0)
    if not r.get("ok"):
        return dict(key=key, ok=False, why=str(r.get("why"))[:200], s=time.time() - t0)
    info = r["info"]; ns = [int(x) for x in info["ns"]]
    rn = lambda n: float(np.hypot(xn, yn + vy * n * DT))
    info["vin0"] = v0; info["vin1"] = M._vin(rn(ns[3])); info["next_in_reach"] = int(D.in_reach(xn, yn + vy * ns[3] * DT) and D.in_reach(xn, yn + vy * ns[5] * DT)); info["slp_cap"] = D.CAP
    return dict(key=key, ok=True, ns=ns, tc=ns[3] * DT, u=np.asarray(r["U"], np.float32), info=info, s=time.time() - t0)


def save(out, res, n_all):
    ok = [r for r in res if r["ok"]]; bad = [r for r in res if not r["ok"]]; NU = max([len(r["u"]) for r in ok] + [1])
    U = np.zeros((len(ok), NU, 6), np.float32)
    for i, r in enumerate(ok):
        U[i, :len(r["u"])] = r["u"]; U[i, len(r["u"]):] = r["u"][-1]
    g = lambda k, d=np.nan: np.array([r["info"].get(k, d) for r in ok], float)
    tmp = out + ".tmp.npz"
    np.savez(tmp, mode="vin", keys=np.array([r["key"] for r in ok], float).reshape(-1, 5), ns=np.array([r["ns"] for r in ok], int).reshape(-1, 6), tc=np.array([r["tc"] for r in ok]), U=U,
             secs=np.array([r["s"] for r in ok]), n_lp=g("n_lp"), slp_it=g("slp_it"), rel_omega=g("rel_omega"), win_omega_max=g("win_omega_max"), win_land_cm_mean=g("win_land_cm_mean"),
             win_land_cm_max=g("win_land_cm_max"), land_cm=g("land_cm"), J_time=g("J_time"), J_land=g("J_land"), J_omega=g("J_omega"), tcp_pos_err_mm=g("tcp_pos_err_mm"), z_min=g("z_min"), x_min=g("x_min"),
             rel_fallback=g("rel_fallback", 0), next_in_reach=g("next_in_reach"), vin0=g("vin0"), vin1=g("vin1"), how=np.array([str(r["info"].get("how")) for r in ok]), info=np.array([json.dumps(r["info"]) for r in ok]),
             fail_keys=np.array([r["key"] for r in bad], float).reshape(-1, 5), fail_why=np.array([r["why"] for r in bad]), fail_secs=np.array([r["s"] for r in bad]),
             XS=D.XS, YI=D.YI, DY=D.DY, bins=np.array([0, 1, 2]), reach=json.dumps(D.RCH), settings=json.dumps(dict(D.SET, VIN=1, vin_table=json.load(open(os.path.join(D.HERE, "logs", "vin_table.json"))))),
             t_cyc=D.T_CYC, t_pick=D.T_PICK, n_cells=n_all)
    os.replace(tmp, out)


U0 = {}


def job(a):
    """a = 칸 (튜플) 또는 dict(key, ns0, u0 (DB_U0 파일의 이름))."""
    if isinstance(a, dict):
        u0 = None
        if a.get("u0") and os.environ.get("DB_U0"):
            if not U0:
                U0.update(dict(np.load(os.environ["DB_U0"])))
            u0 = U0[a["u0"]]
        r = run_cell(kt(a["key"]), a.get("ns0"), u0)
        if r.get("ok"):
            r["info"]["ns0"] = a.get("ns0"); r["info"]["ns0_src"] = a.get("src")
        return r
    return run_cell(a)


def main():
    out, W = sys.argv[1], int(sys.argv[2])
    allk = [(xi, yi, xn, round(yi + dy, 4), b) for b in (0, 1, 2) for yi in D.YI for xi in D.XS for dy in D.DY for xn in D.XS]
    keys = [k for k in allk if D.cell_ok(k)]; n_all = len(keys)
    print(f"격자 {len(allk)} 칸 → 범위 안 {n_all} 칸 | v_in 표 {M._vin(0.25)} … {M._vin(0.64)}", flush=True); print("설정", dict(D.SET, VIN=1), flush=True)
    rng = np.random.default_rng(0); keys = [keys[i] for i in rng.permutation(len(keys))]
    if os.environ.get("DB_KEYS"):
        want = {kt([float(v) for v in s.split(",")]) for s in os.environ["DB_KEYS"].split(";")}; keys = [k for k in keys if kt(k) in want]
    if os.environ.get("DB_SPLIT"):
        ab, nn = os.environ["DB_SPLIT"].split("/"); a_, b_ = (int(x) for x in ab.split("-")); nn = int(nn)
        keys = [k for i, k in enumerate(keys) if a_ <= i % nn < b_]; print(f"나눠 갖기 {os.environ['DB_SPLIT']}: {len(keys)} 칸", flush=True)
    if os.environ.get("DB_REVERSE", "0") == "1":
        keys = keys[::-1]
    jobs = list(keys)
    if os.environ.get("DB_LIST"):                                             # 칸마다 출발 시각표 (이웃 해) 를 준 목록
        L = json.load(open(os.environ["DB_LIST"])); keep = {kt(k) for k in keys}; jobs = [a for a in L if kt(a["key"]) in keep]
    res = []
    if os.path.exists(out):
        Z = np.load(out, allow_pickle=True)
        for i, k in enumerate(Z["keys"]):
            res.append(dict(key=kt(k), ok=True, ns=Z["ns"][i].tolist(), tc=float(Z["tc"][i]), u=Z["U"][i][:int(Z["ns"][i][3])], info=json.loads(str(Z["info"][i])), s=float(Z["secs"][i])))
        for i, k in enumerate(Z["fail_keys"]):
            res.append(dict(key=kt(k), ok=False, why=str(Z["fail_why"][i]), s=float(Z["fail_secs"][i])))
    for fs in [x for x in os.environ.get("DB_SKIP", "").split(",") if x and os.path.exists(x)]:
        Z = np.load(fs, allow_pickle=True); res_k = {kt(k) for k in Z["keys"]} | {kt(k) for k in Z["fail_keys"]}
        jobs = [a for a in jobs if kt(a["key"] if isinstance(a, dict) else a) not in res_k]
    done = {r["key"] for r in res}; jobs = [a for a in jobs if kt(a["key"] if isinstance(a, dict) else a) not in done]
    print(f"이미 끝난 칸 {len(done)}, 풀 칸 {len(jobs)}, 워커 {W}", flush=True); t0 = time.time(); n = 0
    with mp.get_context("fork").Pool(W, maxtasksperchild=6) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r); n += 1
            if r["ok"]:
                i_ = r["info"]; ns = r["ns"]
                print(f"[{n}/{len(jobs)}] {r['key']} ns {ns} (경계→경계 {ns[3] * DT:.3f} s, 접촉→다음 접촉 {(ns[5] - ns[0]) * DT:.3f} s) v_in {i_.get('vin0')}→{i_.get('vin1')}, 릴리즈 ω {i_.get('rel_omega')}, 창 착지 {i_.get('win_land_cm_mean')} cm, "
                      f"반복 {i_.get('slp_it')}, {i_.get('how')}, {r['s']:.0f} s" + (" [고정 릴리즈 대체]" if i_.get("rel_fallback") else ""), flush=True)
            else:
                print(f"[{n}/{len(jobs)}] {r['key']} 실패: {r['why']} ({r['s']:.0f} s)", flush=True)
            if n % 8 == 0 or n == len(jobs):
                save(out, res, n_all); print(f"   저장 {len(res)} (경과 {(time.time() - t0) / 60:.1f} 분)", flush=True)
    save(out, res, n_all); print("DBWINDONE", flush=True)


if __name__ == "__main__":
    main()
