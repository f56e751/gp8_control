"""db_pool: solve a list of cycle cells with N persistent solver processes (rn2.py through srv.py; each start = nearest solved cell of RN_DB, re-listed on every call).
usage: python db_pool.py KEYS.txt OUT_DIR N [wave]   — KEYS lines "xa,ya,xb,yb,b [level]"; with 'wave' the levels are done one after another (a level starts from the ones before)."""
import os, sys, json, time, threading, queue
import numpy as np
import rt_core as C
keys = [l.split() for l in open(sys.argv[1]) if l.strip()]; out = sys.argv[2]; N = int(sys.argv[3]); wave = len(sys.argv) > 4; os.makedirs(out, exist_ok=True)
env = {"RN_AT_CONTACT": "1", "RN_LIM": os.environ.get("DB_LIM", "0.8,8,160"), "RN_A0": "0,0,0,0,0,0", "RN_DB": os.environ.get("RN_DB", out), "RN_MAXT": "60", "LPP_XMIN": "0.25"}
if os.environ.get("DB_REFRESH", "1") != "0":                                 # re-list the start library on every call (needed when it is the folder being filled); DB_REFRESH=0 for a fixed start library
    env["RN_DB_REFRESH"] = "1"
srvs = [C.Srv(sys.executable) for _ in range(N)]; res = []; lock = threading.Lock(); t0 = time.time()


def work(s, q):
    while True:
        try:
            k = q.get_nowait()
        except queue.Empty:
            return
        f = os.path.join(out, "c_" + k.replace(",", "_") + ".npz"); r = s.call("rn2.py", [k, f], env)
        try:
            I = json.loads(str(np.load(f, allow_pickle=True)["info"])); row = (k, I["status"], I["iters"], I["solve_s"], I["viol"], I["t_cycle"], I["start_dist"])
        except Exception as e:  # noqa: BLE001
            row = (k, "ERROR " + str(r.get("err"))[:80], 0, 0.0, 9.0, 0.0, 0.0)
        with lock:
            res.append(row)


levels = sorted({k[1] for k in keys if len(k) > 1}, reverse=True) if wave else [None]
for lv in levels:
    q = queue.Queue(); [q.put(k[0]) for k in keys if lv is None or k[1] == lv]; n0 = len(res); th = [threading.Thread(target=work, args=(s, q)) for s in srvs]; [t.start() for t in th]; [t.join() for t in th]
    good = [r for r in res[n0:] if r[1] in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and r[4] <= 1e-5]
    print(f"level {lv}: {len(res) - n0} cells, solved {len(good)}, iterations median {np.median([r[2] for r in res[n0:]]):.0f}, solve median {np.median([r[3] for r in res[n0:]]):.2f} s [{time.time() - t0:.0f} s]", flush=True)
json.dump(res, open(os.path.join(out, "_pool_%d.json" % int(t0)), "w")); [s.close() for s in srvs]
bad = [r for r in res if not (r[1] in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and r[4] <= 1e-5)]; print("total", len(res), "not solved", len(bad), [(r[0], r[1], "%.1e" % r[4]) for r in bad][:10], "[%.0f s]" % (time.time() - t0))
