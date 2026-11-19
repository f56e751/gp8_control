"""nnws_pack (2026-11-13, user: "두 물체의 위치와 통 번호를 입력으로 하고 단계 시간 4개와 궤적 노드 값(Q, V, A, J)을 출력으로 하는 신경망을 만들 수 있을까"):
packs the solved cells of a library folder into one file for training.  usage: python nnws_pack.py FOLDER OUT.npz [workers]
Per cell: key (xa, ya, xb, yb, bin), the four free durations (lift, to the window, to the next hover, descent), Q V A (73 x 6), J (72 x 6), cycle time,
iterations / solve time / start distance of the solve that produced it."""
import os, sys, glob, json
os.environ["LPP_VIN"] = "1"
import numpy as np
from multiprocessing import Pool


def rd(f):
    try:
        Z = np.load(f, allow_pickle=True); I = json.loads(str(Z["info"]))
        if not (I["status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and I["viol"] <= 1e-5) or Z["Q"].shape != (73, 6):
            return None
        k = I["key"]; T = I["T"]
        return (list(k[:5]) + [float(I.get("zA", 0.02)), float(I.get("zB", 0.02))], [T[0], T[1], T[4], T[5]], Z["Q"], Z["V"], Z["A"], Z["J"], I["t_cycle"], I["iters"], I["solve_s"], I.get("start_dist", 0.0))
    except Exception as e:
        return None


if __name__ == "__main__":
    fs = sorted(f for f in glob.glob(os.path.join(sys.argv[1], "c_*.npz")) if len(os.path.basename(f)[2:-4].split("_")) in (5, 7))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 12) as p:
        R = [r for r in p.map(rd, fs, chunksize=200) if r is not None]
    import db_vin as V
    VU = np.asarray(V.M.V.QD, float) / 0.9
    np.savez(sys.argv[2], key=np.array([r[0] for r in R], float), T=np.array([r[1] for r in R], float), Q=np.array([r[2] for r in R], np.float32), V=np.array([r[3] for r in R], np.float32),
             A=np.array([r[4] for r in R], np.float32), J=np.array([r[5] for r in R], np.float32), tc=np.array([r[6] for r in R], float), iters=np.array([r[7] for r in R]),
             solve_s=np.array([r[8] for r in R], float), dist=np.array([r[9] for r in R], float), VU=VU)
    print("PACKED", sys.argv[1], len(fs), "files ->", len(R), "cells; rated speeds [rad/s]", VU.round(3).tolist())
