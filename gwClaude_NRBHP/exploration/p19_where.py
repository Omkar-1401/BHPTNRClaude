"""EXPLORATION pass 19 -- where does the 27c leader (pass 18, V6r + 1 beta / 1 alpha shape) lose, against the per-case floor?

Band decomposition (p3_where.decompose, NR-time bands from the NR peak) of
  (a) the leader, p18/p18_p15_11.json, and
  (b) the pass-17 per-case spline floor (V6r x exp(per-case splines)),
on T96.  Printed by chi1 column and by q row (edges vs centre).  Writes exploration/p19/p19_where.json.
"""
import json, sys
from pathlib import Path
import multiprocessing as mp
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "V6", "p15", "1", "1"]
import p18_shapes2 as L                        # noqa: E402  (P6.ab = the 27c leader's ab)
import p17_spline_v6r as F17                   # noqa: E402  (V6r base + per-case splines)
import p3_where as W                           # noqa: E402
P6 = L.P6
OUT = HERE / "p19"; OUT.mkdir(exist_ok=True)
TH = np.array(json.loads((HERE / "p18" / "p18_p15_11.json").read_text())["theta"])
Z = {(r["q"], r["chi1"]): np.array(r["z"]) for r in json.loads((HERE / "p17" / "p17_p15.json").read_text())["rows"]}


def one(k):
    case = P6.get(k)["case"]; t = case["t_bhpt"]; s = t - P6.get(k)["tpk"]; n = len(F17.KN)
    a, b = L.ab(TH, k, "full")
    a0, b0 = F17.P16.ab(F17.TH, k, "full"); z = Z[k]
    af, bf = a0 * np.exp(F17.spl(z[n:], s)), b0 * np.exp(F17.spl(z[:n], s))
    return dict(q=k[0], chi1=k[1], lead=W.decompose(a, b, case), floor=W.decompose(af, bf, case))


if __name__ == "__main__":
    with mp.get_context("fork").Pool(96) as pool:
        rows = pool.map(one, P6.T96)
    (OUT / "p19_where.json").write_text(json.dumps(rows, indent=1))
    print("bands (NR time from the NR peak, M): " + " ".join(f"[{lo:.0f},{hi:.0f}]" for lo, hi in W.BANDS))
    for m in ("lead", "floor"):
        E = np.array([r[m]["E"] for r in rows]); P = np.array([[b["E_part"] for b in r[m]["bands"]] for r in rows])
        print(f"{m}: mean {E.mean():.3e}; band contributions to the mean: " + " ".join(f"{v:.2e}" for v in P.mean(0)))
        for lab, key, vals in (("chi1", "chi1", sorted(set(r["chi1"] for r in rows))),
                               ("q", "q", sorted(set(r["q"] for r in rows)))):
            for v in vals:
                ii = [i for i, r in enumerate(rows) if r[key] == v]
                print(f"   {lab} {v:+6.2f}: mean {E[ii].mean():.2e} | " + " ".join(f"{x:.1e}" for x in P[ii].mean(0)))
