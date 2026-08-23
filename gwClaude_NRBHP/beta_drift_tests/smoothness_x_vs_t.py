"""Is alpha actually SMOOTHER in x than in t?  That premise is why the x-line exists
(the f-domain diagnostic saw alpha(f)/a_PP ~3x smoother than the time domain), so measure
it before choosing a form.

Operational definition of "smoother" = the one a model can represent with FEWER
COEFFICIENTS.  For each q, fit the measured alpha/anchor and beta/anchor with polynomials
of degree 1..8 in each candidate coordinate and report the relative RMS residual.

Two weightings, because the samples are uniform in accumulated GW PHASE (that is how the
NR/ppBHPT pairing is built), not in t or x:
  as-sampled  -- phase-uniform, arguably the right weighting for a waveform model
  t-uniform   -- resampled uniform in NR time, to check the answer is not a weighting artefact

Coordinates: t_NR, x = (M Omega_orb)^(2/3), ln x, and x^(-5/8) (~ the PN time-to-merger
variable, monotone in t but stretched like x).
"""
import sys, json, warnings
import numpy as np
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_PN_opt_creative_q_dep as qdep
import alpha_beta_of_omega_v2 as V2   # reuse phase_and_freq  (also re-runs its main; cheap)

QS = [2.0, 3.0, 5.0, 8.0]
WIN_M = 20.0
DEGS = [1, 2, 3, 4, 5, 6, 8]
store = dict(np.load(HERE / "nr_long_cache.npz"))


def paired(q, win_M=WIN_M):
    d = np.load(qdep.waveform_cache_path(q))
    t_pp = d["t_bhpt"]; h_pp = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    t_nr = store[f"q{q:.4f}_t"]; h_nr = store[f"q{q:.4f}_h"]
    php, wpp, ip, *_ = V2.phase_and_freq(t_pp, h_pp, win_M)
    phn, wnr, iN, *_ = V2.phase_and_freq(t_nr, h_nr, win_M)
    tn = t_nr - t_nr[iN]
    sp, sn = slice(0, ip), slice(0, iN)
    lo = max(php[sp][0], phn[sn][0]); hi = min(php[sp][-1], phn[sn][-1])
    pg = np.linspace(lo * 0.98, hi * 0.98, 300)
    w_p = np.interp(pg, php[sp], wpp[sp]); A_p = np.interp(pg, php[sp], np.abs(h_pp)[sp])
    w_n = np.interp(pg, phn[sn], wnr[sn]); A_n = np.interp(pg, phn[sn], np.abs(h_nr)[sn])
    t_at = np.interp(pg, phn[sn], tn[sn])
    return dict(x=(0.5 * w_p) ** (2 / 3), beta=w_p / w_n, alpha=A_n / A_p,
                t=t_at, anchor=(q / (1 + q)) ** 1.2)


def rms(coord, y, deg):
    """relative RMS residual of a degree-`deg` polynomial fit, coord mapped to [-1,1]."""
    u = 2 * (coord - coord.min()) / (coord.max() - coord.min()) - 1
    r = y - np.polyval(np.polyfit(u, y, deg), u)
    return float(np.sqrt(np.mean(r ** 2)) / np.mean(np.abs(y)))


for weighting in ("as-sampled (phase-uniform)", "t-uniform"):
    print(f"\n{'='*96}\nweighting: {weighting}")
    for name in ("alpha", "beta"):
        print(f"\n--- {name}/anchor,  relative RMS residual vs polynomial degree "
              f"(smoothing {WIN_M:g} M) ---")
        print(f"{'q':>4s} {'coord':>10s} " + "".join(f"{'deg '+str(d):>10s}" for d in DEGS))
        for q in QS:
            P = paired(q)
            y0 = P[name] / P["anchor"]; t0 = P["t"]; x0 = P["x"]
            if weighting == "t-uniform":
                tu = np.linspace(t0.min(), t0.max(), 300)
                o = np.argsort(t0)
                y = np.interp(tu, t0[o], y0[o]); x = np.interp(tu, t0[o], x0[o]); t = tu
            else:
                y, x, t = y0, x0, t0
            for lab, c in (("t_NR", t), ("x", x), ("ln x", np.log(x)),
                           ("x^-5/8", x ** -0.625)):
                print(f"{q:4.1f} {lab:>10s} " +
                      "".join(f"{rms(c, y, d):10.2e}" for d in DEGS), flush=True)
            print()
