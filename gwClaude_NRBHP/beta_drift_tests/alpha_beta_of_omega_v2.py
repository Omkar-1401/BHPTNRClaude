"""alpha(x), beta(x) parameter-free, merger-aligned -- with MATCHED smoothing.

Fixes a flaw in v1 (alpha_beta_of_omega.py): the savgol window was a fixed 401 SAMPLES,
but the ppBHPT cache has dt = 0.2 M and the regenerated NR has dt = 0.1 M, so the two
sides were smoothed over 80 M and 40 M respectively.  That asymmetry does NOT cancel in
beta = omega_pp/omega_NR, and it lands hardest near merger where the GW period is ~22 M
(i.e. ~3.6 vs ~1.8 cycles of smoothing) -- exactly where the excursion accumulates.

Here the window is set in PHYSICAL TIME (M) and converted to samples per waveform, and
omega comes from savgol's own analytic derivative rather than gradient-of-smoothed.
Runs several window lengths so the sensitivity is visible instead of assumed.
"""
import sys, json, warnings
import numpy as np
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_PN_opt_creative_q_dep as qdep

QS = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
WINDOWS_M = [10.0, 20.0, 40.0]
NRCACHE = HERE / "nr_long_cache.npz"
MODE = (2, 2)


def phase_and_freq(t, h, win_M):
    """phi (0 at |h| peak) and omega, smoothed over win_M MASS UNITS on both sides."""
    dt = float(np.median(np.diff(t)))
    win = int(round(win_M/dt));  win += (win % 2 == 0);  win = max(win, 7)
    ph = np.unwrap(np.angle(h))
    if ph[-1] < ph[0]:
        ph = -ph
    ph_s = savgol_filter(ph, win, 3, mode="interp")
    w = np.abs(savgol_filter(ph, win, 3, mode="interp", deriv=1, delta=dt))
    im = int(np.argmax(np.abs(h)))
    return ph_s - ph_s[im], w, im, win, win*dt


store = dict(np.load(NRCACHE))
out = {}
for win_M in WINDOWS_M:
    print(f"\n########## smoothing window = {win_M:g} M on BOTH sides ##########")
    print(f"{'q':>5s} {'pp win':>14s} {'NR win':>14s} {'beta excursion':>15s} "
          f"{'alpha excursion':>16s} {'beta(low x)/X1^6/5':>19s}")
    rows = {}
    for q in QS:
        d = np.load(qdep.waveform_cache_path(q))
        t_pp = d["t_bhpt"]; h_pp = d["h_bhpt_re"] + 1j*d["h_bhpt_im"]
        t_nr = store[f"q{q:.4f}_t"]; h_nr = store[f"q{q:.4f}_h"]
        php, wpp, ip, wp_n, wp_M = phase_and_freq(t_pp, h_pp, win_M)
        phn, wnr, iN, wn_n, wn_M = phase_and_freq(t_nr, h_nr, win_M)
        tp = t_pp - t_pp[ip]; tn = t_nr - t_nr[iN]
        sp, sn = slice(0, ip), slice(0, iN)
        lo = max(php[sp][0], phn[sn][0]); hi = min(php[sp][-1], phn[sn][-1])
        pg = np.linspace(lo*0.98, hi*0.98, 300)
        w_p = np.interp(pg, php[sp], wpp[sp]); A_p = np.interp(pg, php[sp], np.abs(h_pp)[sp])
        w_n = np.interp(pg, phn[sn], wnr[sn]); A_n = np.interp(pg, phn[sn], np.abs(h_nr)[sn])
        beta = w_p/w_n; alpha = A_n/A_p; x = (0.5*w_p)**(2.0/3.0)
        anc = (q/(1+q))**1.2
        db = (beta[-1]-beta[0])/beta[0]*100; da = (alpha[-1]-alpha[0])/alpha[0]*100
        print(f"{q:5.2f} {wp_n:6d}={wp_M:5.1f}M {wn_n:6d}={wn_M:5.1f}M {db:+14.2f}% "
              f"{da:+15.2f}% {beta[0]/anc:19.5f}")
        rows[q] = dict(x=x.tolist(), beta=beta.tolist(), alpha=alpha.tolist(),
                       anchor=anc, dbeta=db, dalpha=da)
    NU = lambda q: q/(1+q)**2
    for nm, key in (("beta", "dbeta"), ("alpha", "dalpha")):
        for use, lab in ((QS, "all 7"), ([q for q in QS if q >= 3], "q>=3")):
            dd = np.array([abs(rows[q][key]) for q in use]); nu = np.array([NU(q) for q in use])
            p = np.polyfit(np.log(nu), np.log(dd), 1)[0]
            print(f"   {nm:5s} {lab:6s}: |excursion| ~ nu^{p:.2f}")
    out[f"{win_M:g}"] = rows
json.dump(out, open(HERE/"alpha_beta_of_omega_v2.json", "w"))
print("\nwrote alpha_beta_of_omega_v2.json")
