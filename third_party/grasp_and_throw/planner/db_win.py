"""11-10 사용자: "지금 딱 이 설정으로 DB 를 robot6, 9 에서 최대한 빠르게. 접근 가능 범위는 반경 r 로 보수적으로. 초기값은 1 개."
설정 = lp_planner.plan 그대로 (cold): TCP 제약 (실제 궤적)·관절 4 free·릴리즈 속도 3 행 (위치·각속도 자유)·릴리즈 창 ±0.05 s 항 (W1 100, 착지 1e6, 각속도 1)·가속 합 1e-4.
칸 = (잡는 물체 x, y, 다음 물체 x, y, 빈) — 계획 시작 순간 (물체 i 위 hover 추종 중) 의 위치. 초기값 1 개 = 고정 기본 시각표 (안 풀리면 배율, 처음 풀리는 것).
접근 가능 범위 = logs/reach_r.json (r_lo ≤ √(x²+y²) ≤ r_hi — 그 밖의 조건 없음). 칸 선별은 파이프라인 판정과 같게: 물체 i 가 0·t_pick 에, 다음 물체가 t_cyc·t_cyc+t_pick 에 범위 안.
사용: python db_win.py out.npz 워커   (DB_SPLIT="a-b/n" 로 서버 나눠 갖기, DB_SKIP=다른 서버 결과 파일)"""
import os, sys, time, json
os.environ["LPP_TCP"] = "1"; os.environ["LPP_RELFREE"] = "1"; os.environ["LPP_WIN"] = "1"
os.environ.setdefault("LPC_TLIM", "60")                                        # 워커가 많아 LP 가 느려져도 '시간 제한 = 불가' 로 잘못 판정하지 않게 (문제는 그대로)
import numpy as np
import multiprocessing as mp
import lp_planner as M
from lp_planner import plan, SG, DT, H, VB, LAG, ik, BINS

HERE = os.path.dirname(os.path.abspath(__file__))
RCH = json.load(open(os.path.join(HERE, "logs", "reach_r.json")))
XS = [0.25, 0.35, 0.45, 0.55]; YI = [float(v) for v in os.environ.get("DB_YI", "0.15,0.25,0.35,0.45,0.55,0.65").split(",")]; DY = [0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72]
T_CYC = float(os.environ.get("DB_T_CYC", "1.05")); T_PICK = float(os.environ.get("DB_T_PICK", "0.45")); ZTOP = 0.02
os.environ.setdefault("LPC_SLP_IT", "60")                                      # 11-10 사용자: 탐색 상한 60 회
SET = {k: os.environ.get(k) for k in ("LPP_TCP", "LPP_RELFREE", "LPP_WIN", "LPP_WIN_RT", "LPP_W1", "LPP_WACC", "LPP_WOM", "LPP_TQ_TIE", "LPP_TQ_W", "LPP_BS", "LPC_SLP_IT", "LPC_SLP_TR", "LPC_BAND_POS", "LPC_BAND_VEL",
                                     "LPC_TLIM", "LPP_XMIN", "LPP_OBJ_NEXT", "GP8_BELT_SPEED", "GP8_LIM_SCALE", "GP8_THR_FILTER")}
SET.update(OBJ_NEXT=M.OBJ_NEXT, XMIN=M.XMIN, WIN_RT=M.WIN_RT, WIN_N=M.WIN_N, W1=M.WIN_W1, WACC=M.WIN_WACC, WOM=M.WIN_WOM, TQ_TIE=M.TQ_TIE, TCP=M.TCP_MODE, REL=M.REL_MODE, WIN=M.WIN_ON)


CAP = int(os.environ.get("LPC_SLP_IT", "40"))                                  # 시각 탐색 반복 상한 (11-10 사용자: 60)


def in_reach(x, y):
    return bool(RCH["r_lo"] <= np.hypot(x, y) <= RCH["r_hi"])                    # 11-10 사용자: 고리만 (하류 한계 등 다른 조건 없음)


def cell_ok(k):
    xi, yi, xn, yn = k[:4]; vy = float(VB[1])
    return in_reach(xi, yi) and in_reach(xi, yi + vy * T_PICK) and in_reach(xn, yn + vy * T_CYC) and in_reach(xn, yn + vy * (T_CYC + T_PICK))


def run_cell(key, ns0=None):
    xi, yi, xn, yn, b = key; b = int(b); vy = float(VB[1]); t0 = time.time()
    try:
        Sp = np.array([ik(np.array([xi, yi + vy * ((j - 300) * DT + LAG), float(H)])) for j in range(301)]); hist = Sp * SG
        oi = dict(id=1, cls="x", p=[xi, yi, ZTOP], v=[0, vy, 0], bin=list(BINS[b])); on = dict(id=2, cls="x", p=[xn, yn, ZTOP], v=[0, vy, 0], bin=list(BINS[0]))
        r = plan(dict(kind="pick", state="track", obj=oi), dict(kind="track", obj=on), hist, dict(mode="cold", deadline=None, dt=DT, h=float(H), **({"ns0": [int(x) for x in ns0]} if ns0 is not None else {})))
    except Exception as e:                                                    # noqa: BLE001
        return dict(key=key, ok=False, why=f"{type(e).__name__}: {e}"[:200], s=time.time() - t0)
    if not r.get("ok"):
        return dict(key=key, ok=False, why=str(r.get("why"))[:200], s=time.time() - t0)
    info = r["info"]; ns = [int(x) for x in info["ns"]]
    y_h = yn + vy * ns[3] * DT; y_c = yn + vy * ns[4] * DT                                # 풀린 해에서 다음 물체 hover·접촉 순간의 위치가 범위 안인지 (기록만)
    info["next_in_reach"] = int(in_reach(xn, y_h) and in_reach(xn, y_c)); info["slp_cap"] = CAP
    return dict(key=key, ok=True, ns=ns, tc=ns[3] * DT, u=np.asarray(r["U"], np.float32), info=info, s=time.time() - t0)


def save(out, res, n_all):
    ok = [r for r in res if r["ok"]]; bad = [r for r in res if not r["ok"]]; NU = max([len(r["u"]) for r in ok] + [1])
    U = np.zeros((len(ok), NU, 6), np.float32)
    for i, r in enumerate(ok):
        U[i, :len(r["u"])] = r["u"]; U[i, len(r["u"]):] = r["u"][-1]
    g = lambda k, d=np.nan: np.array([r["info"].get(k, d) for r in ok], float)
    tmp = out + ".tmp.npz"
    np.savez(tmp, mode="next_win", keys=np.array([r["key"] for r in ok], float).reshape(-1, 5), ns=np.array([r["ns"] for r in ok], int).reshape(-1, 5), tc=np.array([r["tc"] for r in ok]), U=U,
             secs=np.array([r["s"] for r in ok]), n_lp=g("n_lp"), slp_it=g("slp_it"), rel_omega=g("rel_omega"), win_omega_max=g("win_omega_max"), win_land_cm_mean=g("win_land_cm_mean"),
             win_land_cm_max=g("win_land_cm_max"), land_cm=g("land_cm"), J_time=g("J_time"), J_land=g("J_land"), J_omega=g("J_omega"), tcp_pos_err_mm=g("tcp_pos_err_mm"), z_min=g("z_min"), x_min=g("x_min"),
             rel_fallback=g("rel_fallback", 0), next_in_reach=g("next_in_reach"), how=np.array([str(r["info"].get("how")) for r in ok]), info=np.array([json.dumps(r["info"]) for r in ok]),
             fail_keys=np.array([r["key"] for r in bad], float).reshape(-1, 5), fail_why=np.array([r["why"] for r in bad]), fail_secs=np.array([r["s"] for r in bad]),
             XS=XS, YI=YI, DY=DY, bins=np.array([0, 1, 2]), reach=json.dumps(RCH), settings=json.dumps(SET), t_cyc=T_CYC, t_pick=T_PICK, n_cells=n_all)
    os.replace(tmp, out)


def main():
    out, W = sys.argv[1], int(sys.argv[2])
    allk = [(xi, yi, xn, round(yi + dy, 4), b) for b in (0, 1, 2) for yi in YI for xi in XS for dy in DY for xn in XS]
    keys = [k for k in allk if cell_ok(k)]; n_all = len(keys)
    if os.environ.get("DB_BASE") and M.XMIN is not None:                      # 기준 DB 에서 x 최소 < 하한인 칸만
        Zb = np.load(os.environ["DB_BASE"], allow_pickle=True); kb = lambda k: tuple([round(float(v), 4) for v in k[:4]] + [int(k[4])])
        low = {kb(k) for k, xm in zip(Zb["keys"], Zb["x_min"]) if float(xm) < M.XMIN}
        keys = [k for k in keys if kb(k) in low]; print(f"기준 DB {os.environ['DB_BASE']} 에서 x 최소 < {M.XMIN}: {len(keys)} 칸만 다시 풉니다", flush=True)
    print(f"접근 가능 범위 {RCH} / 격자 {len(allk)} 칸 → 범위 안 {n_all} 칸", flush=True); print("설정", SET, flush=True)
    rng = np.random.default_rng(0); keys = [keys[i] for i in rng.permutation(len(keys))]             # 고르게 섞기 (서버 나눠 갖기·중간 결과가 공간을 고르게 덮도록)
    if os.environ.get("DB_ORDER"):                                            # 11-11 사용자: 기준 DB 의 사이클 시간 (접촉→다음 접촉) 이 긴 칸부터
        Zo = np.load(os.environ["DB_ORDER"], allow_pickle=True); ko = lambda k: tuple([round(float(v), 4) for v in k[:4]] + [int(k[4])])
        cyc = {ko(k): int(n[4]) - int(n[0]) for k, n in zip(Zo["keys"], Zo["ns"])}
        keys.sort(key=lambda k: -cyc.get(ko(k), 10 ** 9)); print(f"순서: {os.environ['DB_ORDER']} 의 사이클 시간 긴 칸부터 (가장 긴 칸 {keys[0]} {cyc.get(ko(keys[0]), 0) * 0.002:.3f} s)", flush=True)
    if os.environ.get("DB_SPLIT"):                                            # "a-b/n": 번호 % n 이 [a, b) 인 칸만
        ab, nn = os.environ["DB_SPLIT"].split("/"); a_, b_ = (int(x) for x in ab.split("-")); nn = int(nn)
        keys = [k for i, k in enumerate(keys) if a_ <= i % nn < b_]; print(f"나눠 갖기 {os.environ['DB_SPLIT']}: {len(keys)} 칸", flush=True)
    if os.environ.get("DB_REVERSE", "0") == "1":                              # 같은 몫을 다른 서버가 반대편 끝에서부터 (느린 서버 돕기)
        keys = keys[::-1]; print("칸 순서 뒤집음 (DB_REVERSE)", flush=True)
    res = []
    kt = lambda k: tuple([round(float(v), 4) for v in k[:4]] + [int(k[4])])
    for fs in [out] + [x for x in os.environ.get("DB_SKIP", "").split(",") if x]:   # 이어하기 / 다른 서버가 이미 푼 칸
        if not os.path.exists(fs):
            continue
        Z = np.load(fs, allow_pickle=True); infos = [json.loads(str(x)) for x in Z["info"]]
        redo = {kt(k) for k, i_ in zip(Z["keys"], infos) if int(i_.get("slp_cap", 40)) < CAP and int(i_.get("slp_it", 0)) >= int(i_.get("slp_cap", 40))}   # 예전 상한에 걸린 칸 → 다시
        done = ({kt(k) for k in Z["keys"]} | {kt(k) for k in Z["fail_keys"]}) - redo
        print(f"예전 상한에 걸려 다시 풀 칸 ({fs}): {len(redo)}", flush=True)
        if fs == out:
            for i, k in enumerate(Z["keys"]):
                if kt(k) in redo:
                    continue
                res.append(dict(key=kt(k), ok=True, ns=Z["ns"][i].tolist(), tc=float(Z["tc"][i]), u=Z["U"][i][:int(Z["ns"][i][3])], info=json.loads(str(Z["info"][i])), s=float(Z["secs"][i])))
            for i, k in enumerate(Z["fail_keys"]):
                res.append(dict(key=kt(k), ok=False, why=str(Z["fail_why"][i]), s=float(Z["fail_secs"][i])))
        n0 = len(keys); keys = [k for k in keys if kt(k) not in done]; print(f"이미 끝난 칸 ({fs}): {n0 - len(keys)}", flush=True)
    print(f"풀 칸 {len(keys)}, 워커 {W}", flush=True); t0 = time.time(); n = 0
    with mp.get_context("fork").Pool(W, maxtasksperchild=6) as pool:
        for r in pool.imap_unordered(run_cell, keys):
            res.append(r); n += 1
            if r["ok"]:
                i_ = r["info"]
                print(f"[{n}/{len(keys)}] {r['key']} t_c {r['tc']:.3f} s, 릴리즈 ω {i_.get('rel_omega')}, 창 착지 {i_.get('win_land_cm_mean')} cm, 반복 {i_.get('slp_it')}, LP {i_.get('n_lp')}, {r['s']:.0f} s"
                      + (" [고정 릴리즈 대체]" if i_.get("rel_fallback") else "") + ("" if i_.get("next_in_reach") else " [다음 물체 hover 가 범위 밖]"), flush=True)
            else:
                print(f"[{n}/{len(keys)}] {r['key']} 실패: {r['why']} ({r['s']:.0f} s)", flush=True)
            if n % 8 == 0 or n == len(keys):
                save(out, res, n_all); print(f"   저장 {len(res)} (경과 {(time.time() - t0) / 60:.1f} 분)", flush=True)
    save(out, res, n_all); print("DBWINDONE", flush=True)


if __name__ == "__main__":
    main()
