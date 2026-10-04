"""STEP 1 of the spin extension: MEASURE alpha(x), beta(x) vs chi1, parameter-free.

No model, no fit.  For each (q, chi1) the two waveforms are generated independently
(ppBHPT: BHPTNRSur2dq1e3 with calibrated=False, i.e. raw 0PA; NR: NRHybSur3dq8 with
chi1 = [0,0,chi1], chi2 = 0 because the ppBHPT secondary is a non-spinning point
particle), then alpha and beta are read off by the merger-aligned equal-accumulated-
phase construction of beta_drift_tests/alpha_beta_of_omega_v2.py:

    beta  = omega_pp / omega_NR      at points paired by equal phase back from merger
    alpha = |h_NR| / |h_pp|          at the same paired points
    x     = (0.5 omega_pp)^(2/3)

Smoothing window set in PHYSICAL M on both sides (the v1 flaw).

THE TWO QUESTIONS this is built to answer, both about the ratio to the chi1 = 0 curve
at the same q,

    R_alpha(x; q, chi1) = alpha(x; q, chi1) / alpha(x; q, 0)     (same for beta)

  Q1  TIME-INDEPENDENCE: is R independent of x?   (the user's proposed ansatz assumes
      yes; Islam+ 2407.18319 CANNOT have tested this -- their alpha, beta are
      time-independent by construction, their Eq. 6.)
  Q2  SEPARABILITY: is R independent of q?  (their Eq. 8/9 assume yes -- the spin factor
      multiplies the 1/q polynomial.)

Islam's own prediction for the ratio is overlaid: with
alpha(q,chi) = 1 + P(q)(1 + S(chi)),  R_pred = [1 + P(q)(1+S)] / [1 + P(q)],
which is NOT simply the spin factor -- it depends on q, so their form predicts a
specific, mild Q2 violation.  Coefficients read from the surrogate h5.

Usage:  python spin_measure_alpha_beta.py [--win 20] [--regen]
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/work2/11702/omkarsridhar1401/stampede3/BHPTNRSurrogate/surrogates")
RES = ROOT / "spin_measure_results"; RES.mkdir(exist_ok=True)
H5 = "/work2/11702/omkarsridhar1401/stampede3/BHPTNRSurrogate/data/BHPTNRSur2dq1e3.h5"

QS = [3.0, 4.0, 5.0, 6.0, 8.0]
CHIS = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.8]
MODE = (2, 2)
F_LOW = 5e-3


def phase_and_freq(t, h, win_M):
    dt = float(np.median(np.diff(t)))
    win = int(round(win_M / dt)); win += (win % 2 == 0); win = max(win, 7)
    ph = np.unwrap(np.angle(h))
    if ph[-1] < ph[0]:
        ph = -ph
    ph_s = savgol_filter(ph, win, 3, mode="interp")
    w = np.abs(savgol_filter(ph, win, 3, mode="interp", deriv=1, delta=dt))
    im = int(np.argmax(np.abs(h)))
    return ph_s - ph_s[im], w, im


def measure(q, chi1, win_M):
    import BHPTNRSur2dq1e3 as ppsur
    import gwsurrogate
    global _NRSUR
    try:
        _NRSUR
    except NameError:
        _NRSUR = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    t_pp, hd = ppsur.generate_surrogate(q=q, spin1=chi1, calibrated=False,
                                        modes=[MODE], neg_modes=False)
    h_pp = hd[MODE]
    t_nr, hn, _ = _NRSUR(q, [0, 0, chi1], [0, 0, 0.0], dt=0.1, f_low=F_LOW)
    h_nr = hn[MODE]
    php, wpp, ip = phase_and_freq(np.asarray(t_pp, float), h_pp, win_M)
    phn, wnr, iN = phase_and_freq(np.asarray(t_nr, float), h_nr, win_M)
    sp, sn = slice(0, ip), slice(0, iN)
    lo = max(php[sp][0], phn[sn][0]); hi = min(php[sp][-1], phn[sn][-1])
    pg = np.linspace(lo * 0.98, hi * 0.98, 400)
    w_p = np.interp(pg, php[sp], wpp[sp]); A_p = np.interp(pg, php[sp], np.abs(h_pp)[sp])
    w_n = np.interp(pg, phn[sn], wnr[sn]); A_n = np.interp(pg, phn[sn], np.abs(h_nr)[sn])
    x = (0.5 * w_p) ** (2.0 / 3.0)
    return dict(x=x.tolist(), alpha=(A_n / A_p).tolist(), beta=(w_p / w_n).tolist())


def islam_coeffs():
    import h5py
    with h5py.File(H5, "r") as f:
        g = f["nr_calib_params"]["(2,2)"]
        return np.array(g["alpha"]), np.array(g["beta"])


def islam_ratio(coef, q, chi):
    A, B = coef[:4], coef[4:]
    P = sum(A[n] * q ** -(n + 1) for n in range(4))
    S = sum(B[n] * chi ** (n + 1) for n in range(2))
    return (1 + P * (1 + S)) / (1 + P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=20.0)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()
    cache = RES / f"curves_win{args.win:.0f}.json"
    data = {} if args.regen or not cache.exists() else json.loads(cache.read_text())
    for q in QS:
        for c in CHIS:
            k = f"{q:.1f}_{c:+.2f}"
            if k in data:
                continue
            data[k] = measure(q, c, args.win)
            cache.write_text(json.dumps(data))
            print(f"  measured q={q} chi1={c:+.2f}  "
                  f"x[{min(data[k]['x']):.3f},{max(data[k]['x']):.3f}]", flush=True)
    print(f"\nwrote {cache}  ({len(data)} points)")

    ac, bc = islam_coeffs()
    XREF = [0.06, 0.09, 0.12]
    for name, coef in (("alpha", ac), ("beta", bc)):
        print(f"\n===== {name}: R = {name}(x;q,chi)/{name}(x;q,0) =====")
        print(f"{'q':>4} {'chi1':>6} " + " ".join(f"{'R@x='+str(x):>10}" for x in XREF)
              + f" {'spread in x':>12} {'Islam R':>9}")
        for q in QS:
            b = data[f"{q:.1f}_+0.00"]
            xb, yb = np.array(b["x"]), np.array(b[name])
            ob = np.argsort(xb)
            for c in CHIS:
                if c == 0.0:
                    continue
                d = data[f"{q:.1f}_{c:+.2f}"]
                xd, yd = np.array(d["x"]), np.array(d[name])
                od = np.argsort(xd)
                lo = max(xb.min(), xd.min()); hi = min(xb.max(), xd.max(), 0.13)
                xg = np.linspace(max(lo, 0.05), hi, 200)
                R = np.interp(xg, xd[od], yd[od]) / np.interp(xg, xb[ob], yb[ob])
                vals = [np.interp(xr, xg, R) if xg.min() <= xr <= xg.max() else np.nan
                        for xr in XREF]
                spread = (R.max() - R.min()) / np.mean(R) * 100
                print(f"{q:>4g} {c:>+6.2f} " + " ".join(f"{v:10.4f}" for v in vals)
                      + f" {spread:11.2f}% {islam_ratio(coef, q, c):9.4f}")


if __name__ == "__main__":
    main()
