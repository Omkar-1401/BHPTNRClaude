"""alpha(omega) and beta(omega), measured parameter-free with MERGER ALIGNMENT.

beta comes from phase matching: differentiate phi_NR(tau(t)) = phi_pp(t) to get
    beta = omega_pp(t) / omega_NR(tau(t))                 at corresponding times
and alpha is the amplitude ratio at those SAME corresponding times
    alpha = |h_NR(tau)| / |h_pp(t)| .
Nothing is fitted.  The one irreducible freedom -- the constant in
phi_NR(tau) = phi_pp(t) + dphi -- is fixed by MERGER ALIGNMENT: both waveforms have
t = 0 and phi = 0 at their own |h| peak.  Points are then paired by equal accumulated
phase measured back from merger.

The earlier attempt (alpha_beta_of_x.py) was limited by the CACHED NR waveform starting
at only t = -5000 M, giving a narrow x strip (0.11-0.14) in which beta's slope could not
be resolved.  Here NRHybSur3dq8 is regenerated with a lower f_low and NO truncation; the
ppBHPT side is reused from the cache (it is already the full ~30500 M window).
"""
import sys, json, warnings
import numpy as np
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_PN_opt_creative_q_dep as qdep

QS = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
F_LOW = 2.0e-3                      # cache used 5e-3 -> x ~ 0.063; this reaches lower
MODE = (2, 2)
NRCACHE = HERE / "nr_long_cache.npz"


def phase_and_freq(t, h):
    """phi (increasing, 0 at the |h| peak) and omega = dphi/dt, lightly smoothed."""
    ph = np.unwrap(np.angle(h))
    if ph[-1] < ph[0]:
        ph = -ph                                  # h = A exp(-i Phi): make phi increase
    n = len(t); win = min(401, n - (1 - n % 2)); win = win - 1 if win % 2 == 0 else win
    if win >= 11:
        ph = savgol_filter(ph, win, 3, mode="interp")
    im = int(np.argmax(np.abs(h)))
    return ph - ph[im], np.abs(np.gradient(ph, t)), im


def load_nr_long(q, store):
    key = f"q{q:.4f}"
    if key + "_t" in store:
        return store[key + "_t"], store[key + "_h"]
    import gwsurrogate
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    t, hd, _ = nrsur(q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=F_LOW)
    store[key + "_t"] = t; store[key + "_h"] = hd[MODE]
    return t, hd[MODE]


store = dict(np.load(NRCACHE)) if NRCACHE.exists() else {}
rows = {}
for q in QS:
    d = np.load(qdep.waveform_cache_path(q))
    t_pp = d["t_bhpt"]; h_pp = d["h_bhpt_re"] + 1j*d["h_bhpt_im"]
    t_nr, h_nr = load_nr_long(q, store)
    php, wpp, ip = phase_and_freq(t_pp, h_pp)
    phn, wnr, iN = phase_and_freq(t_nr, h_nr)
    tp = t_pp - t_pp[ip]; tn = t_nr - t_nr[iN]          # merger at 0 on both
    print(f"\n=== q={q:g}   pp window [{tp[0]:.0f}, {tp[-1]:.0f}] M,  "
          f"NR window [{tn[0]:.0f}, {tn[-1]:.0f}] M   (cache was -5000)")
    # inspiral, monotone phase, before merger
    sp = slice(0, ip); sn = slice(0, iN)
    php_i, wpp_i, App_i, tp_i = php[sp], wpp[sp], np.abs(h_pp)[sp], tp[sp]
    phn_i, wnr_i, Anr_i, tn_i = phn[sn], wnr[sn], np.abs(h_nr)[sn], tn[sn]
    lo = max(php_i[0], phn_i[0]); hi = min(php_i[-1], phn_i[-1])
    pg = np.linspace(lo*0.98, hi*0.98, 300)            # common accumulated-phase grid
    t_of_p  = np.interp(pg, php_i, tp_i);  w_p = np.interp(pg, php_i, wpp_i)
    A_p     = np.interp(pg, php_i, App_i)
    t_of_n  = np.interp(pg, phn_i, tn_i);  w_n = np.interp(pg, phn_i, wnr_i)
    A_n     = np.interp(pg, phn_i, Anr_i)
    beta = w_p/w_n; alpha = A_n/A_p; x_pp = (0.5*w_p)**(2.0/3.0)
    X1 = q/(1+q); anc = X1**1.2; nu = q/(1+q)**2
    print(f"   phase overlap: {hi-lo:.0f} rad ({(hi-lo)/(2*np.pi):.1f} cycles), "
          f"x_pp in [{x_pp[0]:.4f}, {x_pp[-1]:.4f}]")
    print(f"   {'t_pp':>8s} {'x_pp':>7s} {'beta':>8s} {'b/X1^6/5':>9s} "
          f"{'alpha':>8s} {'a/X1^6/5':>9s} {'1+(55/42)nu x':>14s}")
    for i in (0, 74, 149, 224, 299):
        print(f"   {t_of_p[i]:8.0f} {x_pp[i]:7.4f} {beta[i]:8.5f} {beta[i]/anc:9.5f} "
              f"{alpha[i]:8.5f} {alpha[i]/anc:9.5f} "
              f"{1+(55/42)*nu*x_pp[i]:14.5f}")
    print(f"   beta  over the range: {(beta[-1]-beta[0])/beta[0]*100:+.2f}%  "
          f"{'RISES' if beta[-1]>beta[0] else 'FALLS'}")
    print(f"   alpha over the range: {(alpha[-1]-alpha[0])/alpha[0]*100:+.2f}%")
    rows[q] = dict(x=x_pp.tolist(), beta=beta.tolist(), alpha=alpha.tolist(),
                   t=t_of_p.tolist(), anchor=anc)
np.savez(NRCACHE, **store)
json.dump({str(k): v for k, v in rows.items()}, open(HERE/"alpha_beta_of_omega.json","w"))
print("\nwrote alpha_beta_of_omega.json")
