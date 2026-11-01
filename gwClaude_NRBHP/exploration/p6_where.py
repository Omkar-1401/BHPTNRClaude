"""Band decomposition (p3_where) of the pass-6 M2 leader and its per-case floor (p5b) on T96.
Writes exploration/p6/p6_where.json."""
import json, sys
from pathlib import Path
import multiprocessing as mp
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p3_where as W, p6_m2_global as P6, p5b_floor as F

TH = json.loads((HERE / "p6" / "p6_m2.json").read_text())["stage2_T96"]["theta"]
FL = {(r["q"], r["chi1"]): r["M2"]["p"] for r in json.loads((HERE / "p5b" / "p5b_floor.json").read_text())}

def one(k):
    a, b = P6.ab(np.asarray(TH), k, "full")
    case = P6.get(k)["case"]
    C = F.Case(k); af, bf = C.ab(np.asarray(FL[k]), "M2", "full")
    return dict(q=k[0], chi1=k[1], M2=W.decompose(a, b, case), floor=W.decompose(af, bf, case))

if __name__ == "__main__":
    with mp.get_context("fork").Pool(96) as pool:
        rows = pool.map(one, P6.T96)
    (HERE / "p6" / "p6_where.json").write_text(json.dumps(rows, indent=1))
    for m in ("M2", "floor"):
        E = np.array([r[m]["E"] for r in rows]); P = np.array([[b["E_part"] for b in r[m]["bands"]] for r in rows])
        print(f"{m}: mean {E.mean():.3e}  band contributions to the mean (absolute):")
        print("   " + "  ".join(f"[{lo:.0f},{hi:.0f}] {P[:, j].mean():.2e}" for j, (lo, hi) in enumerate(W.BANDS)))
        for ch in sorted(set(r["chi1"] for r in rows)):
            ii = [i for i, r in enumerate(rows) if r["chi1"] == ch]
            print(f"   chi1 {ch:+.1f}: mean {E[ii].mean():.2e} | " + " ".join(f"{v:.1e}" for v in P[ii].mean(0)))
