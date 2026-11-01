"""EXPLORATION pass 20 -- per-case INSPIRAL corrections on the 27c leader (pass 18), to design the inspiral regression.

Pass 19: the +0.8 column and the q = 3, 8 rows lose in the inspiral bands ([-5000, -150] M), where the per-case floor is
~1e-6 -- a regression problem.  Per T96 case, on top of the leader, before its blend (factor 1 - W, the leader's window):
    ln beta  += (d0 + d1 x^(3/2) + d2 x^2) (1 - W),     ln alpha += (d3 + d4 x) (1 - W)
x = the inspiral x of the flux branch (x_scale(q) min(x_pp, XCLIP)).  5 parameters per case, Powell on the fit-stride
error.  The d's against (nu, chi1) then say which inspiral terms the global form lacks.
Writes exploration/p20/p20_insp_floor.json.
"""
import json, sys
from pathlib import Path
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize, minimize_scalar
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "V6", "p15", "1", "1"]
import p18_shapes2 as L                        # noqa: E402
import p3_rd as P3                             # noqa: E402
import fit_pn_anchored_spin as S               # noqa: E402
import fit_scaling_PN_opt_creative as creative  # noqa: E402
P6 = L.P6; A = L.A
OUT = HERE / "p20"; OUT.mkdir(exist_ok=True)
TH = np.array(json.loads((HERE / "p18" / "p18_p15_11.json").read_text())["theta"])


def run(k):
    q, chi1 = k; c = P6.get(k); case = c["case"]
    d = dict(zip(A.NAMES, TH[:len(A.NAMES)]))
    s1 = d["t1a"] + d["t1b"] * chi1; s2 = d["t2a"] + d["t2b"] * chi1
    pre = {"fit": "fit_", "full": ""}
    arr = {}
    for key in ("fit", "full"):
        a, b = L.ab(TH, k, key)
        t = case[pre[key] + "t_bhpt"]
        W = P3.smooth5((t - c["tpk"] - s1) / (s2 - s1))
        x = S.PC.x_scale(q) * np.minimum(case[pre[key] + "x"], S.XCLIP)
        arr[key] = (a, b, 1 - W, x)
    st = {"t0": None}

    def err(p, key="fit", wide=False):
        a, b, w, x = arr[key]; pr = pre[key]
        t, h, tn, hn = case[pr + "t_bhpt"], case[pr + "h_bhpt"], case[pr + "t_nr"], case[pr + "h_nr"]
        bb = b * np.exp((p[0] + p[1] * x ** 1.5 + p[2] * x ** 2) * w)
        aa = a * np.exp((p[3] + p[4] * x) * w)
        bc = creative.cumulative_trapezoid(bb, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
        if np.min(np.diff(ts)) <= 0:
            return 1.0
        f = lambda x0: S._err_at_t0(x0, aa, ts, t, h, tn, hn)
        if st["t0"] is None or wide:
            g = np.linspace(-300, 200, 126); c0 = g[int(np.argmin([f(v) for v in g]))]; wd = 6
        else:
            c0, wd = st["t0"], 15
        r = minimize_scalar(f, bounds=(c0 - wd, c0 + wd), method="bounded", options={"xatol": 1e-3})
        st["t0"] = float(r.x)
        return float(r.fun)

    p0 = np.zeros(5)
    e0 = err(p0, "full", wide=True)
    st["t0"] = None; err(p0, "fit", wide=True)
    best = (np.inf, p0)
    for _ in range(3):
        r = minimize(err, best[1], method="Powell", options={"xtol": 1e-6, "ftol": 1e-9, "maxiter": 40})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    e1 = err(best[1], "full", wide=True)
    return dict(q=q, chi1=chi1, E_lead=e0, E_insp=e1, p=list(map(float, best[1])))


if __name__ == "__main__":
    with mp.get_context("fork").Pool(96) as pool:
        rows = pool.map(run, P6.T96)
    (OUT / "p20_insp_floor.json").write_text(json.dumps(rows, indent=1))
    E0 = np.array([r["E_lead"] for r in rows]); E1 = np.array([r["E_insp"] for r in rows])
    print(f"T96: leader {E0.mean():.3e} / {np.median(E0):.3e} / {E0.max():.3e} -> + per-case inspiral terms "
          f"{E1.mean():.3e} / {np.median(E1):.3e} / {E1.max():.3e}", flush=True)
    for r in rows:
        print(f"  q {r['q']:5.2f} chi1 {r['chi1']:+.1f}  {r['E_lead']:.2e} -> {r['E_insp']:.2e}  d = "
              + " ".join(f"{v:+.4f}" for v in r["p"]), flush=True)
