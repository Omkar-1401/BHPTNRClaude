"""Which SHAPE functions can represent NR's beta(t) in the spin models?  No mismatch fit.

Question (user, 2026-09-27): would Jdot work as a shape function for beta, compared with the
E(t) drive of fluxanchored_spin / energy_anchored_spin (beta = beta_PP + P(nu,chi1)*E(t))?

Method.  Measured beta(t) = omega_pp / omega_NR at equal (2,2) phase, both phases zero at the
|h| peak (as beta_drift_tests/alpha_beta_of_omega_v2.py / chi_t_anchor_check.py), on the NR
clock.  Every regressor is a ppBHPT quantity -- what a model can actually use -- evaluated at
the phase-matched ppBHPT time.  Per case, least squares of beta_meas on each basis:
    {1, E}          the shipped shape (E = the models' own drive, spin_dep/flux_loss_cache)
    {1, J}          cumulative radiated J_z from the ppBHPT (2,2)
    {1, E, Edot}    + the models' despiked flux F(t) (flux_hat)
    {1, E, Jdot}    + normalised Jdot
    {1, J, Jdot}
Coefficients are free PER CASE, so this is an upper bound on what each shape can do (the
real models tie them across the grid through P(nu, chi1)).  Fit window t_NR in FIT_WIN,
uniform in NR time; the residual rms is reported by band.  `python beta_shape_projection.py 10`
ends the window at +10 M: the retrograde ringdown omega is not a single clean frequency
(q=3, chi1=-0.6 gives beta_meas 0.05-1.85 there), which swamps the lstsq otherwise.

Sensitivity: repeated with the NR phase offset shifted by +-1 rad, because "measured beta"
depends on the alignment choice.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

CASES = [(q, c) for q in (3.0, 8.0) for c in (-0.6, 0.0, 0.4, 0.8)]
WIN = 10.0
FIT_WIN = (-1000.0, float(sys.argv[1]) if len(sys.argv) > 1 else 60.0)
BANDS = [b for b in [(-1000, -300), (-300, -100), (-100, -30), (-30, 10), (10, 60)] if b[1] <= FIT_WIN[1]]
TAG = "" if FIT_WIN[1] == 60.0 else f"_to{FIT_WIN[1]:+.0f}"
SHIFTS = [0.0, -1.0, 1.0]
BASES = {"1,E": ["E"], "1,J": ["J"], "1,E,Edot": ["E", "Edot"],
         "1,E,Jdot": ["E", "Jdot"], "1,J,Jdot": ["J", "Jdot"]}


def phase_freq(t, h):
    dt = float(np.median(np.diff(t)))
    w = int(round(WIN / dt)); w += (w % 2 == 0); w = max(w, 7)
    ph = np.unwrap(np.angle(h))
    if ph[-1] < ph[0]:
        ph = -ph
    im = int(np.argmax(np.abs(h)))
    phs = savgol_filter(ph, w, 3, mode="interp")
    om = np.abs(savgol_filter(ph, w, 3, mode="interp", deriv=1, delta=dt))
    return phs - phs[im], om, im, w


out = {}
for q, c in CASES:
    d = np.load(HERE / "spin_wf_cache" / f"wf_q{q:.6f}_chi{c:+.3f}.npz")
    z = np.load(HERE / "flux_loss_cache" / f"loss_q{q:.6f}_chi{c:+.3f}.npz")
    tp = d["t_bhpt"]; hp = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    tn = d["t_nr"]; hn = d["h_nr_re"] + 1j * d["h_nr_im"]
    E = np.asarray(z["e"], float); F = np.asarray(z["f"], float)
    php, omp, ip, wpp = phase_freq(tp, hp)
    phn, omn, iN, _ = phase_freq(tn, hn)
    # ppBHPT J_z flux of the (2,2) mode (+ its m<0 partner), smoothed like omega
    hd = np.gradient(hp, tp)
    jd = savgol_filter(np.abs(np.imag(hp * np.conj(hd))), wpp, 3, mode="interp")
    J = cumulative_trapezoid(jd, tp, initial=0.0)
    degen = np.median(np.abs(savgol_filter(np.abs(hd) ** 2, wpp, 3, mode="interp")
                             / (omp * jd) - 1)[(tp - tp[ip] > -1000) & (tp - tp[ip] < -100)])
    reg = {"E": E, "J": J / J[-1], "Edot": F, "Jdot": jd / jd.max()}
    tnr = tn - tn[iN]
    mono = np.r_[True, np.diff(php) > 0]
    res_case = {}
    for sh in SHIFTS:
        m = (tnr >= FIT_WIN[0]) & (tnr <= FIT_WIN[1])
        tg = tnr[m]; pg = phn[m] + sh
        ok = (pg > php[mono][0]) & (pg < php[mono][-1])
        tg, pg = tg[ok], pg[ok]
        tpp = np.interp(pg, php[mono], tp[mono])                 # matched ppBHPT time
        y = np.interp(tpp, tp, omp) / np.interp(tg, tnr, omn)    # beta_meas
        rows = {}
        for name, cols in BASES.items():
            X = np.column_stack([np.ones_like(tg)] + [np.interp(tpp, tp, reg[k]) for k in cols])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = (X @ coef - y) / y
            rows[name] = {f"{a}..{b}": float(np.sqrt(np.mean(r[(tg >= a) & (tg < b)] ** 2)))
                          for a, b in BANDS}
            rows[name]["all"] = float(np.sqrt(np.mean(r ** 2)))
        res_case[sh] = rows
    out[f"q{q:g}_chi{c:+.1f}"] = {"degeneracy_insp": float(degen),
                                  "shifts": {str(k): v for k, v in res_case.items()}}
    print(f"\n=== q = {q:g}, chi1 = {c:+.1f}   (|2Edot/(omega Jdot)-1| inspiral median "
          f"{degen:.1e};  beta_meas range {y.min():.3f}-{y.max():.3f})")
    print(f"  {'basis':>9s} " + " ".join(f"{f'{a}..{b}':>11s}" for a, b in BANDS)
          + f" {'all':>7s}   all @ shift -1 / +1 rad")
    for name in BASES:
        r0 = res_case[0.0][name]
        print(f"  {name:>9s} " + " ".join(f"{r0[f'{a}..{b}']:11.2%}" for a, b in BANDS)
              + f" {r0['all']:7.2%}   {res_case[-1.0][name]['all']:6.2%} / "
                f"{res_case[1.0][name]['all']:6.2%}")

# grid summary: mean over cases of the rms by band, shift 0
print("\n=== MEAN over the 8 cases (shift 0)")
print(f"  {'basis':>9s} " + " ".join(f"{f'{a}..{b}':>11s}" for a, b in BANDS) + f" {'all':>7s}")
for name in BASES:
    vals = [np.mean([out[k]["shifts"]["0.0"][name][f"{a}..{b}"] for k in out]) for a, b in BANDS]
    al = np.mean([out[k]["shifts"]["0.0"][name]["all"] for k in out])
    print(f"  {name:>9s} " + " ".join(f"{v:11.2%}" for v in vals) + f" {al:7.2%}")
json.dump(out, open(HERE / f"beta_shape_projection{TAG}.json", "w"), indent=1)
print("\nwrote beta_shape_projection" + TAG + ".json")
