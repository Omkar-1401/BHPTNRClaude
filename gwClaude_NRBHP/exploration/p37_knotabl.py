"""EXPLORATION pass 37 -- which knots of the pass-35 per-case splines carry the chi1 = +0.8 / -0.6 gain (no refit).
Each mask keeps only part of each case's spline correction z (pass 35, on the 18c pass-33 A model) and rescores
(full resolution, t0 refit only).  Writes exploration/p37/p37_knotabl.json.
"""
import json, sys
from pathlib import Path
import multiprocessing as mp
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p35_floor18 as F                          # noqa: E402  (spl, KN, TH, M25.ab, P6)
P6 = F.P6; KN = F.KN; n = len(KN)
from scipy.optimize import minimize_scalar      # noqa: E402
OUT = HERE / "p37"; OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "p35" / "p35_floor.json").read_text())
SB = KN <= -50; MB = (KN > -50) & (KN <= 0); LB = KN > 0
MASKS = {"none": (0 * KN, 0 * KN), "all": (1 + 0 * KN, 1 + 0 * KN), "B": (1 + 0 * KN, 0 * KN), "A": (0 * KN, 1 + 0 * KN),
         "B_early(<=-50)": (SB, 0 * KN), "B_merger(-30..0)": (MB, 0 * KN), "B_rd(>0)": (LB, 0 * KN),
         "A_early": (0 * KN, SB), "A_merger": (0 * KN, MB), "A_rd": (0 * KN, LB)}


def run(row):
    k = (row["q"], row["chi1"]); c = P6.get(k); case = c["case"]
    a0, b0 = F.M25.ab(F.TH, k, "full"); z = np.array(row["z"])
    t, h, tn, hn = case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"]; s = t - c["tpk"]
    out = {}
    for name, (mb, ma) in MASKS.items():
        a = a0 * np.exp(F.spl(z[n:] * ma, s)); b = b0 * np.exp(F.spl(z[:n] * mb, s))
        bc = F.creative.cumulative_trapezoid(b, t); ts = bc - np.interp(F.S.T_ANCHOR, t, bc)
        f = lambda x0: F.S._err_at_t0(x0, a, ts, t, h, tn, hn)
        g = np.linspace(-300, 200, 126); c0 = g[int(np.argmin([f(x) for x in g]))]
        out[name] = float(minimize_scalar(f, bounds=(c0 - 6, c0 + 6), method="bounded", options={"xatol": 1e-3}).fun)
    return dict(q=row["q"], chi1=row["chi1"], E=out)


if __name__ == "__main__":
    rows = [r for r in D["rows"] if abs(r["chi1"] - 0.8) < 1e-6 or abs(r["chi1"] + 0.6) < 1e-6 or abs(r["chi1"]) < 1e-6]
    with mp.get_context("fork").Pool(48) as pool:
        res = pool.map(run, rows)
    (OUT / "p37_knotabl.json").write_text(json.dumps(res, indent=1))
    for ch in (-0.6, 0.0, 0.8):
        rr = [r for r in res if abs(r["chi1"] - ch) < 1e-6]
        print(f"chi1 {ch:+.1f} medians (1e-4): " + "  ".join(f"{m} {1e4 * np.median([r['E'][m] for r in rr]):.2f}" for m in MASKS), flush=True)
