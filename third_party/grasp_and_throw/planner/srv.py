"""srv: keeps the heavy imports loaded and runs a script per request line (JSON: {"script", "argv", "env"}). Prints one JSON line per request."""
import os, sys, json, time, runpy, io, contextlib
os.environ["LPP_VIN"] = "1"
import numpy as np, casadi  # noqa
import db_vin, p2p  # noqa
HERE = os.path.dirname(os.path.abspath(__file__)); out = sys.stdout; print(json.dumps(dict(ready=1)), file=out, flush=True)
for line in sys.stdin:
    r = json.loads(line); t0 = time.time(); os.environ.update({k: str(v) for k, v in r.get("env", {}).items()})
    for k in r.get("unset", []):
        os.environ.pop(k, None)
    sys.argv = [r["script"]] + [str(a) for a in r["argv"]]; buf = io.StringIO(); ok = True; err = ""
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(HERE, r["script"]), run_name="__main__")
    except SystemExit:
        pass
    except BaseException as e:  # noqa: BLE001
        ok = False; err = f"{type(e).__name__}: {e}"[:300]
    print(json.dumps(dict(ok=ok, err=err, s=round(time.time() - t0, 3), id=r.get("id"), tail=buf.getvalue()[-400:])), file=out, flush=True)
