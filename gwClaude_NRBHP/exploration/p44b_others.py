"""Pass 44b: the installed models and Rink et al. on the pass-44 q = 9, 10 points (to separate model extrapolation from
NRHybSur3dq8's own extrapolation above q = 8).  Same caches (exploration/p44/cache), same scorers as t100_score.py.
    python p44b_others.py      -> exploration/p44/p44b.json
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0]]
import p44_highq as H                           # noqa: E402  (installs the p44 cache redirect)
S, FS, KEYS = H.S, H.FS, H.KEYS
import fit_flux_anc_spin_with_orbit as FO       # noqa: E402
import fit_scaling_PN_opt_creative as creative  # noqa: E402
from scipy.optimize import minimize_scalar      # noqa: E402
TH_FO = FO.load_theta()
D = json.loads(S.COEFFS.read_text()); TH9 = S.model9(); THS = np.array(D["theta_spin"], float)


def e_fo(k):
    return FO.mismatch(TH_FO, k[0], k[1], FS.build_case(*k))[0]


def e_pn(k):
    return S.full_mm(TH9, THS, k[0], k[1], FS.build_case(*k))


def e_rink(k):
    import BHPTNRSur2dq1e3 as ppsur
    c = FS.build_case(*k)
    t, hd = ppsur.generate_surrogate(q=k[0], spin1=k[1], calibrated=True, modes=[(2, 2)], neg_modes=False)
    t = np.asarray(t, float); h = np.asarray(hd[(2, 2)])
    si = np.unique(np.r_[np.arange(0, len(t), S.SRC_STRIDE), len(t) - 1]); ts = t - np.interp(S.T_ANCHOR, t, t)
    f = lambda x0: S._err_at_t0(x0, np.ones_like(ts[si]), ts[si], ts[si], h[si], c["fit_t_nr"], c["fit_h_nr"])
    g = np.linspace(-300, 200, 251); cen = float(g[int(np.argmin([f(v) for v in g]))])
    x = minimize_scalar(f, bounds=(cen - 6, cen + 6), method="bounded", options={"xatol": 1e-4}).x
    f2 = lambda x0: S._err_at_t0(x0, np.ones_like(ts), ts, ts, h, c["t_nr"], c["h_nr"])
    return float(minimize_scalar(f2, bounds=(x - 3, x + 3), method="bounded", options={"xatol": 1e-4}).fun)


def main():
    p40 = json.loads((HERE / "p44" / "p44.json").read_text())["cases"]
    out = {}
    with mp.get_context("fork").Pool(48) as pool:
        for name, fn in (("flux_orbit", e_fo), ("pn13", e_pn), ("rink", e_rink)):
            out[name] = dict(zip([f"{q:.6f}_{c:+.4f}" for q, c in KEYS], pool.map(fn, KEYS)))
    (HERE / "p44" / "p44b.json").write_text(json.dumps(out, indent=1))
    print("q    chi1 | pass 40 | flux_orbit | pn13 | Rink", flush=True)
    for q, c in KEYS:
        k = f"{q:.6f}_{c:+.4f}"
        print(f"{q:4g} {c:+.1f} | {p40[k]:.2e} | {out['flux_orbit'][k]:.2e} | {out['pn13'][k]:.2e} | {out['rink'][k]:.2e}", flush=True)


if __name__ == "__main__":
    main()
