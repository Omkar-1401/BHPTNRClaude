"""pn_anc_spin_physical -- the pn-anchored spin calibration at DERIVABLE PN ORDERS.

A MODEL, not a test.  Promoted 2026-09-17 from `fit_pn_anchored_spin_pn15.py` on
STRUCTURAL grounds (see PROVENANCE below): every spin term sits at a PN order that can
be derived rather than fitted, which is the discipline the rest of the workspace already
runs on (`X1**(6/5)`, the 55/42 alpha slope, `beta_r`).  It supersedes nothing
automatically -- `fit_pn_anchored_spin.py` + `pn_anchored_spin_results/coeffs_spin.json`
remain on disk as the 13-coefficient predecessor.

===============================================================================
DEFINITIONS -- every symbol below, and where its value comes from.
Read this before the MODEL block; nothing in that block is defined after it.
===============================================================================

CALIBRATION BEING FITTED.  Two real functions of time that map a ppBHPT waveform onto
an NR one:

    h_NR(tau)  =  alpha(t) * exp(i*phi0) * h_pp(t),      dtau/dt = beta(t)

  h_pp   the ppBHPT (point-particle black-hole perturbation theory) waveform, the
         (2,2) mode of BHPTNRSur2dq1e3 evaluated at (q, chi1).  SOURCE: the surrogate
         package `BHPTNRSurrogate`, loaded in `fit_pn_anchored_spin.build_case`.
         Aligned-spin, domain q in [3, 1000], |chi| <= 0.8, dt = 0.1 m1,
         t in [-13500, 125] m1 (ppBHPT time is in PRIMARY-mass units).
  h_NR   the NR-proxy waveform, the (2,2) mode of NRHybSur3dq8 at the same (q, chi1).
         SOURCE: `gwsurrogate`, same function.  Used as the calibration TARGET; it is
         a surrogate to NR, not NR itself, and is valid only for q <= 8.
  t      the ppBHPT time coordinate, `case["t_bhpt"]`.
  tau    the NR time coordinate, `case["t_nr"]`.
  alpha  amplitude scaling, dimensionless, > 0.
  beta   time-coordinate rate, dimensionless, > 0.  A non-positive beta makes tau
         non-monotone, so the objective returns a 50.0 penalty there.
  phi0   a constant phase, solved ANALYTICALLY (not fitted) inside `S._err_at_t0`.
  t0     a constant time offset, a per-case NUISANCE parameter minimised by
         `minimize_scalar` per (q, chi1).  Not a model coefficient.

MASS-RATIO VARIABLES.  q = m1/m2 >= 1 is the mass ratio (input).

  X1   = q / (1 + q)          the larger body's mass fraction.
  nu   = q / (1 + q)**2       the symmetric mass ratio.
  base = X1 ** 1.2            = X1**(6/5), the NEWTONIAN CHIRP ANCHOR.  DERIVED, not
         fitted.  Its presence is why this model is not PN-free: it fixes the leading
         amplitude/rate scaling so the fitted coefficients only carry corrections.

DRIVE VARIABLES (the "where in the coalescence are we" coordinates).

  x      = (M*Omega)**(2/3), the standard PN frequency variable (note Eq. 6).  The ppBHPT
           phase gives x_pp = (0.5*|d(unwrap(angle(h)))/dt|)**(2/3) on the PRIMARY-mass
           clock (`fit_pn_anchored_spin.get_x`, clipped to [1e-8, 0.6], phase
           savgol-smoothed over 80 m1, polyorder 3); the model uses x = x_pp*X1^-0.8
           (note Eqs. 8, 12).  (Until 2026-09-26 x_pp was used directly.)  x ~ v**2, so x**n corresponds to nPN -- THIS IS THE FACT
           THE WHOLE MODEL TURNS ON.
  xc     = X1^-0.8 * min(x_pp, XCLIP) with XCLIP = 0.26 on x_pp (`S.XCLIP`).  The clip stops the x**2 and
           x**1.5 terms running away through merger, where the PN series has no
           business being evaluated.
  p_loss = 0.5 * (e_hat + j_hat), the mean of the radiated energy and angular momentum
           measured from the |h| peak and each divided by its magnitude at T_REF = -100 m1,
           so e_hat = j_hat = -1 at t = -100 m1, 0 at the peak, positive after
           (fit_scaling_wf_nu_q_dep.wf_loss_coordinates).  (Corrected 2026-09-26: this used
           to say "divided by its own final value so both run 0 -> 1", which is wrong.)
           COMPUTED in `fit_pn_anchored_spin.build_case`.  Used ONLY as the switch
           argument -- it is a monotone progress variable, not an energy.

SPIN VARIABLE.  chi1 is the dimensionless spin of the LARGER body, projected on the
orbital angular momentum; the smaller body is non-spinning.  Grid: `S.CHIS` =
[-0.6, -0.4, -0.2, 0.0, +0.2, +0.4, +0.6, +0.8].

THE MERGER-RINGDOWN ANCHOR (zero fitted parameters).

  beta_r_phys(q, chi1) = w220(chi1) * M_f / w220(chi_f)
         the ratio of the ppBHPT ringdown frequency to the true remnant's, i.e. the rate
         at which ppBHPT time must be stretched so the two ring down together.  pp side:
         the primary rings down, w220(chi1) in PRIMARY-mass units = the ppBHPT time axis.
         NR side: w220(chi_f)/M_f in total-mass units = the NR time axis.
         `fit_pn_anchored_spin.beta_r_phys`.  History: until 2026-09-24 a spurious
         (1+q)/q made it 17-29% too large (UNITS_AUDIT.md A1); until 2026-09-26 the pp side
         was 0.3737*w220_BCW(chi1)/w220_BCW(0) against the raw BCW fit on the NR side.
  w220(chi)    the Kerr (2,2,0) QNM frequency M*omega_220, EXACT (Leaver's continued
         fraction via the `qnm` package, `physics_common.w220_exact`; counter-rotating
         branch for chi < 0).  Replaced the Berti-Cardoso-Will fit on 2026-09-26.
  M_f, chi_f   the remnant mass and spin, from
         `surfinBH.LoadFits("NRSur3dq8Remnant")` evaluated at (q, [0,0,chi1], [0,0,0]).
         THIS IS THE ONLY NR-INFORMED OBJECT IN THE MODEL.  Everything else is either
         derived analytically or fitted to surrogate waveforms.

FIXED CONSTANTS (not fitted).

  PN_SLOPE  = 55/42 (`S.PN_SLOPE`), the derived 1PN coefficient of alpha's inspiral
              expansion.
  T_ANCHOR  = -100.0 in ppBHPT time (m1 units) (`S.T_ANCHOR`), the time at which tau is
              pinned to t so the
              cumulative integral of beta has a fixed zero.
  XCLIP     = 0.26 (`S.XCLIP`), defined above.

THE 11 FITTED COEFFICIENTS.  `NAMES`, in the order `theta` stores them:

  b1, b2    nonspinning beta inspiral shape, at 1PN and 2PN.
  a2        alpha inspiral shape at 2PN (its 1PN coefficient is FIXED at PN_SLOPE).
  p00, p01  the switch location, as p_loss = p00 + p01*nu.
  w0        the switch width in p_loss.
  r0        scale on the merger-ringdown beta branch, multiplying beta_r_phys.
  m0, m1    the merger-ringdown alpha branch, as base*(m0 + m1*nu).
  sSO       1.5PN SPIN-ORBIT coefficient, multiplying chi1 * xc**1.5.
  sSS       2PN SPIN-SPIN coefficient, multiplying chi1**2 * xc**2.

===============================================================================
MODEL
===============================================================================

    Sw  = sigmoid( (p_loss - (p00 + p01*nu)) / w0 )         the merger switch, in [0,1]

    beta = (1 - Sw) * base * ( 1 + nu*( b1*xc + b2*xc**2          # nonspinning
                                        + sSO * chi1    * xc**1.5  # 1.5PN spin-orbit
                                        + sSS * chi1**2 * xc**2 )) # 2PN   spin-spin
         +      Sw  * beta_r_phys(q, chi1) * r0

    alpha = (1 - Sw) * base * ( 1 + PN_SLOPE*nu*xc + nu*a2*xc**2 )
          +      Sw  * base * ( m0 + m1*nu )

alpha carries NO spin term: the spin dependence of alpha measured out consistent with
zero (see `spin_extension_plan.md` sec 7), so adding one would be fitting noise.
chi1 enters beta in exactly two places, both at a derivable PN order, plus beta_r_phys
where it enters with no fitted parameter at all.

`xc**1.5` is safe: x = x_pp * X1^(-4/5) > 0 everywhere by construction, with
x_pp = (m1*omega_22/2)**(2/3) from the ppBHPT phase (physics_common.x_scale).

===============================================================================
PROVENANCE -- why this replaced the 4-coefficient polynomial spin form
===============================================================================

The predecessor spent four coefficients on

    nu * [ (s11*chi1 + s12*chi1**2) * x  +  (s21*chi1 + s22*chi1**2) * x**2 ]

which places chi1 at 1PN and 2PN and chi1**2 at 1PN and 2PN.  Both 1PN placements are
wrong -- spin-orbit enters at 1.5PN, spin-spin at 2PN -- and, decisively, THERE IS NO
x**1.5 TERM ANYWHERE IN IT, so the leading spin-orbit effect had nowhere to live.

Measured consequences in the shipped 13-coefficient fit
(s11 = -0.0856, s12 = -0.2769, s21 = +13.3217, s22 = +8.1237):

  * the linear-in-chi1 piece `s11*x + s21*x**2` has local effective PN order
    d log f / d log x between 2.03 and 2.31 across x in [0.02, 0.26] -- i.e. it behaves
    as a 2PN term EVERYWHERE, never 1.5PN, because s11 is negligible against s21;
  * the spin-orbit / spin-spin ratio at chi1 = +0.8 flattens to 2.05 (the constant
    |s21|*0.8 / |s22|*0.64), so the form cannot separate two different PN orders;
    PN requires that ratio to grow as x**-0.5 toward low frequency;
  * this form puts 2.3x MORE spin correction at x = 0.26 and 2.2x LESS at x = 0.02
    than the PN-ordered one -- the spin effect was dragged onto the merger plateau,
    where the x**2 basis has the most leverage.

The PN-ordered form recovers eff. PN order 1.50 exactly, and an SO/SS ratio running
3.75 -> 9.57 as x falls from 0.26 to 0.04.

It also PARAMETER-MATCHES the spin sector to `fit_fluxanchored_spin.py` and
`fit_energy_anchored_spin.py`, which carry 2 spin coefficients each; the predecessor's
4 made that three-way comparison unfair.

WHAT IS NOT CHANGED from the predecessor: the drive (x, p_loss), the switch, beta_r,
alpha's spin-blindness, the 96-point grid, and the objective.

===============================================================================
OBJECTIVE AND GRID
===============================================================================

Objective: the MEAN over the grid of the per-case waveform mismatch

    (n1 + n2 - 2*|Z|) / (2*n1)

computed on the window t in [-5000.1, +100.0] M with a minimum coverage of 0.88,
maximised analytically over phi0 and numerically over t0.  Median and max are reported
as diagnostics but are NOT minimised -- a run whose median rises while its mean falls
is behaving as specified.

Grid: `S.TRAIN_Q` (12 values in [3, 8]) x `S.CHIS` (8 values in [-0.6, +0.8]) = 96
points, chi1 = 0 INCLUDED.  All 11 coefficients are fitted on this grid; nothing is
transplanted from a nonspinning 1d fit.

q = 2 IS NOT IN THE GRID and reading it is a double extrapolation: outside the training
range nu in [0.0988, 0.1875] (q in [3, 8]; q=2 is nu = 0.2222, 18.5% ABOVE the
top of it), AND outside BHPTNRSur2dq1e3's own q_min = 3.  See
history.md, open items.

Usage:
    python fit_pn_anc_spin_physical.py --promote          # import the pn15 result
    python fit_pn_anc_spin_physical.py --refit --budget 100
    python fit_pn_anc_spin_physical.py --refit --resume --budget 100
    python fit_pn_anc_spin_physical.py --report           # tables from stored coeffs
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os, shutil, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import fit_pn_anchored_spin as S

MODEL_NAME = "pn_anc_spin_physical"
RESULTS = HERE / "pn_anc_spin_physical_results"; RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs_physical.json"
PARTIAL = RESULTS / "partial_physical.json"

# --freeze9 writes here instead, so it never clobbers the promoted model.
FROZEN9 = RESULTS / "coeffs_frozen9.json"
FROZEN9_PARTIAL = RESULTS / "partial_frozen9.json"

# --q/--chi subset runs (small-grid checks) write here instead, so a subset fit never
# clobbers the full-grid coeffs_physical.json it seeds from.
SMALL = RESULTS / "coeffs_physical_small.json"
SMALL_PARTIAL = RESULTS / "partial_physical_small.json"

# where --promote imports from: the test run that produced this model
PN15 = HERE / "pn15_results"
PN15_FINAL, PN15_PARTIAL = PN15 / "coeffs_pn15.json", PN15 / "partial_pn15.json"

NAMES = ["b1", "b2", "a2", "p00", "p01", "w0", "r0", "m0", "m1", "sSO", "sSS"]
N_COEF = len(NAMES)
SNAMES = ["sSO", "sSS"]                      # the spin sector alone
PRED = HERE / "pn_anchored_spin_results" / "coeffs_spin.json"   # 13-coef predecessor
REF = {"pn_anchored+spin 13c, 9 frozen from 1d": 7.6858e-04,
       "fluxanchored+spin 9c on 2d": 1.4110e-03,
       "energy_anchored+spin 8c on 2d": 1.7311e-03}


def model_ab(th, q, chi1, x, p_loss):
    """alpha(t), beta(t) for the 11-coefficient PN-ordered spin model.

    th      the 11 coefficients in `NAMES` order
    q       mass ratio >= 1
    chi1    dimensionless spin of the larger body
    x       x_pp = (omega/2)**(2/3) from S.get_x, on the ppBHPT (m1) clock; converted to the
            note's x = x_pp*X1^-0.8 inside (2026-09-26)
    p_loss  0.5*(e_hat + j_hat), the switch argument
    """
    b1, b2, a2, p00, p01, w0, r0, m0, m1, sSO, sSS = th
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    xc = S.PC.x_scale(q) * np.minimum(x, S.XCLIP)   # note's x (Eq. 6); clip on x_pp (2026-09-26)
    Sw = S.creative.sigmoid((p_loss - (p00 + p01 * nu)) / max(w0, 1e-3))
    beta = ((1 - Sw) * base * (1 + nu * (b1 * xc + b2 * xc ** 2
                                         + sSO * chi1 * xc ** 1.5
                                         + sSS * chi1 ** 2 * xc ** 2))
            + Sw * S.beta_r_phys(q, chi1) * r0)
    alpha = ((1 - Sw) * base * (1 + S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2)
             + Sw * base * (m0 + m1 * nu))
    return alpha, beta


def shipped11(partial=False):
    """The model's stored coefficients.  partial=True reads the in-flight checkpoint."""
    p = PARTIAL if partial else COEFFS
    if not p.exists():
        sys.exit(f"no coefficients at {p} -- run --promote or --refit first")
    d = json.loads(p.read_text())
    if d.get("model_form") != S.MODEL_FORM:            # never SCORE a mismatched file (2026-09-26)
        sys.exit(f"{p.name} was fitted for model_form={d.get('model_form', 'unstamped')!r}, "
                 f"not {S.MODEL_FORM!r}; refit first (--refit).")
    return np.array(d["theta"], float), p


def mismatch(th, q, chi1, case, t0_seed):
    """Per-case mismatch and the t0 that achieves it.  t0 is a nuisance, not a coef."""
    t, h = case["t_bhpt"], case["h_bhpt"]
    tn, hn = case["t_nr"], case["h_nr"]
    alpha, beta = model_ab(th, q, chi1, case["x"], case["p_loss"])
    if np.any(beta <= 0) or not np.all(np.isfinite(np.r_[alpha, beta])):
        return 50.0, t0_seed
    bc = S.creative.cumulative_trapezoid(beta, t)
    ts = bc - float(np.interp(S.T_ANCHOR, t, bc))
    d = float(np.min(np.diff(ts)))
    if d <= 0:
        return 50.0 + 1e3 * (-d), t0_seed
    f = lambda v: S._err_at_t0(v, alpha, ts, t, h, tn, hn)
    if t0_seed is None:
        g = np.linspace(-250.0, 150.0, 81)
        t0_seed = float(g[int(np.argmin([f(v) for v in g]))])
    r = minimize_scalar(f, bounds=(t0_seed - 60.0, t0_seed + 60.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 60})
    return float(r.fun), float(r.x)


def seed_theta():
    """Shipped 9 nonspinning, plus sSO/sSS projected from the predecessor's 4 at the
    mid-inspiral reference x = 0.12:
        s11*x + s21*x**2  ->  sSO*x**1.5     (the chi1-linear part)
        s12*x + s22*x**2  ->  sSS*x**2       (the chi1**2 part)
    NOTE this projection is POOR -- it evaluates at mean 6.06e-03 against the
    predecessor's own 1.28e-03 -- which is itself evidence that x**2 was never a clean
    stand-in for x**1.5.  Kept only as a starting point for --refit from scratch.
    """
    th9 = S.shipped9()
    s11, s12, s21, s22 = json.loads(PRED.read_text())["theta_spin"]
    x = 0.12
    sSO = (s11 * x + s21 * x ** 2) / x ** 1.5
    sSS = (s12 * x + s22 * x ** 2) / x ** 2
    print(f"  seed projection at x={x}: sSO={sSO:+.4f}  sSS={sSS:+.4f}")
    return np.r_[th9, sSO, sSS]


def pn_structure(th):
    """Print the PN-ordering diagnostic that justifies this model, against the
    predecessor.  Uses only stored coefficients -- no waveform evaluation."""
    sSO, sSS = th[9], th[10]
    s11, s12, s21, s22 = json.loads(PRED.read_text())["theta_spin"]
    x = np.array([0.02, 0.04, 0.06, 0.10, 0.12, 0.16, 0.20, 0.26])
    SO_p, SO_n = s11 * x + s21 * x ** 2, sSO * x ** 1.5
    SS_p, SS_n = s12 * x + s22 * x ** 2, sSS * x ** 2
    # guard: a coefficient at exactly 0 would send log(|f|) to -inf
    order = lambda f: np.gradient(np.log(np.maximum(np.abs(f), 1e-300)),
                                  np.log(x))
    print("\nPN STRUCTURE.  linear-in-chi1 piece of beta's nu-bracket; eff. PN order is")
    print("d log f / d log x, and must be 1.50 for spin-orbit.")
    print(f"{'x':>6} {'predecessor':>13} {'effPN':>6} {'physical':>12} {'effPN':>6}"
          f" {'SO/SS pred':>11} {'SO/SS phys':>11}   (SO/SS at chi1=+0.8)")
    op, on = order(SO_p), order(SO_n)
    for i, xx in enumerate(x):
        rp = abs(SO_p[i] * 0.8) / abs(SS_p[i] * 0.64)
        rn = abs(SO_n[i] * 0.8) / abs(SS_n[i] * 0.64)
        print(f"{xx:6.3f} {SO_p[i]:+13.5f} {op[i]:6.2f} {SO_n[i]:+12.5f} {on[i]:6.2f}"
              f" {rp:11.2f} {rn:11.2f}")


def tables(errs, cases, th):
    """The per-(q, chi1) grid, the chi1-column comparison, and the q-U-shape check."""
    allm = {f"{q:.4f}_{c:+.1f}": float(e) for (q, c), e in zip(cases.keys(), errs)}
    print(f"\nper-case mismatch\n{'q':>8} " + " ".join(f"{c:>+9.1f}" for c in S.CHIS))
    for q in S.TRAIN_Q:
        print(f"{q:>8.4f} " + " ".join(
            f"{allm[f'{q:.4f}_{c:+.1f}']:9.2e}" for c in S.CHIS))
    print(f"\nALL 96: mean {np.mean(errs):.4e}  median {np.median(errs):.4e}  "
          f"max {errs.max():.4e}")
    pred = json.loads(PRED.read_text())
    print(f"\nper chi1 column, median over q, vs the 13-coefficient predecessor:")
    print(f"{'chi1':>6} {'physical(11c)':>14} {'predecessor(13c)':>17} {'ratio':>7}")
    for c in S.CHIS:
        new = np.median([allm[f"{q:.4f}_{c:+.1f}"] for q in S.TRAIN_Q])
        old = np.median([pred["mismatches"][f"{q:.4f}_{c:+.1f}"] for q in S.TRAIN_Q])
        print(f"{c:+6.1f} {new:14.4e} {old:17.4e} {new/old:6.2f}x")
    print("\nq-U-SHAPE at chi1 = +0.8 (the column the shipped form handled worst).")
    print(f"{'q':>6} {'physical':>12} {'predecessor':>13} {'ratio':>7}")
    for q in (3.0, 5.0, 8.0):
        k = f"{q:.4f}_+0.8"
        new, old = allm[k], pred["mismatches"][k]
        print(f"{q:6.1f} {new:12.4e} {old:13.4e} {new/old:6.2f}x")
    pn_structure(th)
    print("\nreference: " + ",  ".join(f"{k} {v:.4e}" for k, v in REF.items()))
    return allm


def write_coeffs(th, errs, allm, provenance, path=None, n_free=None, grid=None):
    path = COEFFS if path is None else path
    grid = {"q": S.TRAIN_Q, "chi1": S.CHIS} if grid is None else grid
    path.write_text(json.dumps({
        "model": {COEFFS: MODEL_NAME, SMALL: MODEL_NAME + "_small"}.get(
            path, MODEL_NAME + "_frozen9"),
        "model_form": S.MODEL_FORM,
        "description": "pn-anchored + spin with every spin term at a derivable PN order",
        "form": {
            "beta_inspiral": "base*(1 + nu*(b1*xc + b2*xc**2 "
                             "+ sSO*chi1*xc**1.5 + sSS*chi1**2*xc**2))",
            "beta_mr": "beta_r_phys(q, chi1) * r0",
            "alpha_inspiral": "base*(1 + (55/42)*nu*xc + nu*a2*xc**2)",
            "alpha_mr": "base*(m0 + m1*nu)",
            "switch": "sigmoid((p_loss - (p00 + p01*nu))/w0)",
            "alpha_spin": "NONE -- measured consistent with zero",
            "base": "X1**(6/5), the Newtonian chirp anchor, DERIVED not fitted",
        },
        "pn_orders": {"sSO": "1.5PN spin-orbit (chi1 * x**1.5)",
                      "sSS": "2PN spin-spin (chi1**2 * x**2)"},
        "n_coefficients": N_COEF,
        "n_free_parameters": N_COEF if n_free is None else n_free,
        "names": NAMES,
        "theta": [float(v) for v in th],
        "spin_param_names": SNAMES,
        "theta_spin": [float(th[9]), float(th[10])],
        "all_fitted_on_2d": True,
        "grid": {**grid, "chi0_included": 0.0 in grid["chi1"]},
        "objective": "MEAN over the 96-point grid of the waveform mismatch; median and "
                     "max are diagnostics and are NOT minimised",
        "beta_r_spin": ("DERIVED: beta_r = w220(chi1) * M_f / w220(chi_f), with w220 the EXACT "
                        "Kerr (2,2,0) QNM frequency (qnm package, Leaver continued fraction; "
                        "counter-rotating branch for chi < 0; physics_common.w220_exact).  pp "
                        "side w220(chi1) in m1 units (the primary rings down, ppBHPT clock); NR "
                        "side w220(chi_f)/M_f in M units with M_f, chi_f from surfinBH "
                        "NRSur3dq8Remnant at (q, [0,0,chi1], [0,0,0]).  No free parameter (r0 "
                        "scales it).  The remnant fit is the ONLY NR-informed object in the "
                        "model.  (Berti-Cardoso-Will fit used until 2026-09-26.)"),
        "sources": {
            "h_pp": "BHPTNRSur2dq1e3 (2,2) mode, q in [3,1000], |chi|<=0.8, dt=0.1",
            "h_NR": "NRHybSur3dq8 (2,2) mode -- an NR PROXY, valid q <= 8",
            "x": "x_pp*X1^-0.8 (note Eq. 6), x_pp = (0.5*|d(unwrap(angle(h)))/dt|)**(2/3) "
                 "on the ppBHPT (m1) clock, phase savgol-smoothed over 80 m1 (polyorder 3), "
                 "x_pp clipped [1e-8, 0.6] and at XCLIP=0.26; see fit_pn_anchored_spin.get_x",
            "p_loss": "0.5*(e_hat + j_hat), energy and angular momentum from the |h| peak, "
                      "each divided by its magnitude at T_REF=-100 m1 (-1 there, 0 at the "
                      "peak); see fit_scaling_wf_nu_q_dep.wf_loss_coordinates",
        },
        "mean": float(np.mean(errs)), "median": float(np.median(errs)),
        "max": float(errs.max()),
        "mismatches": allm, "reference": REF,
        "promoted_from": "fit_pn_anchored_spin_pn15.py",
        "predecessor": "pn_anchored_spin_results/coeffs_spin.json (13 coef, "
                       "9 transplanted from a nonspinning 1d fit, median 7.6858e-04)",
        "provenance": provenance,
        "q2_warning": "q = 2 is NOT in the grid. Reading it is a double extrapolation: "
                      "outside the fitted range nu in [0.0988, 0.1875] -- q=2 is "
                      "nu = 0.2222, 18.5% above its top -- AND outside "
                      "BHPTNRSur2dq1e3's "
                      "q_min = 3.",
    }, indent=2))
    print(f"\nwrote {path}")


def build_cases(qs=None, chis=None):
    t0 = time.time(); cases = {}
    for q in (S.TRAIN_Q if qs is None else qs):
        for c in (S.CHIS if chis is None else chis):
            cases[(q, c)] = S.build_case(q, c)
    print(f"cache ready in {time.time()-t0:.0f}s", flush=True)
    return cases


# Parallel scoring (2026-09-27): the 96 cases of one objective evaluation are independent, so
# --refit spreads them over a fork pool (~3.5 s -> ~0.2 s per evaluation).  Same per-case
# function, same t0 warm starts, so the result is identical to the serial loop.
_POOL = None
_PCASES: dict = {}


def _pwork(a):
    th, k, seed = a
    return mismatch(th, k[0], k[1], _PCASES[k], seed)


def start_pool(cases, nproc):
    global _POOL
    _PCASES.clear(); _PCASES.update(cases)
    _POOL = mp.get_context("fork").Pool(nproc)
    print(f"scoring {len(cases)} cases on {nproc} workers", flush=True)


def score_all(th, cases, nuis=None):
    nuis = {k: None for k in cases} if nuis is None else nuis
    if _POOL is not None and all(k in _PCASES for k in cases):
        res = _POOL.map(_pwork, [(th, k, nuis[k]) for k in cases])
        for k, (_, t0_) in zip(cases, res):
            nuis[k] = t0_
        return np.array([e for e, _ in res]), nuis
    out = []
    for k, case in cases.items():
        e, t0_ = mismatch(th, k[0], k[1], case, nuis[k])
        nuis[k] = t0_
        out.append(e)
    return np.array(out), nuis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true",
                    help="import the finished (or in-flight) pn15 coefficients, "
                         "re-score them under this model's name, and write "
                         "coeffs_physical.json")
    ap.add_argument("--freeze9", action="store_true",
                    help="BASIS-VS-BASIS TEST. Freeze the 9 nonspinning coefficients at "
                         "the shipped nonspinning pn_anchored values and fit ONLY "
                         "(sSO, sSS). This makes the model a STRICT spin extension -- "
                         "bit-identical to nonspinning pn_anchored at chi1 = 0 -- and "
                         "directly comparable to the predecessor's 4 polynomial spin "
                         "coefficients on the same frozen 9. Writes coeffs_frozen9.json")
    ap.add_argument("--report", action="store_true",
                    help="re-score the stored coefficients and print the tables")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="no longer required: --refit AUTO-RESUMES whenever "
                         "coeffs_physical.json exists. Kept for compatibility.")
    ap.add_argument("--fresh", action="store_true",
                    help="override the auto-resume and restart from the (poor) "
                         "x=0.12 seed projection")
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--nproc", type=int, default=0,
                    help="--refit: workers for scoring (0 = auto, cores-2 capped at 32; 1 = serial)")
    ap.add_argument("--q", type=float, nargs="*", default=None,
                    help="--refit on this q subset only (small-grid check); writes "
                         "coeffs_physical_small.json, never coeffs_physical.json")
    ap.add_argument("--chi", type=float, nargs="*", default=None,
                    help="--refit on this chi1 subset only (see --q)")
    args = ap.parse_args()
    subset = bool(args.q or args.chi)
    if subset and not args.refit:
        sys.exit("--q/--chi apply to --refit only")
    QS = args.q if args.q else S.TRAIN_Q
    CS = args.chi if args.chi else S.CHIS

    print(f"{MODEL_NAME}: pn-anchored + spin at derivable PN orders")
    print(f"  1.5PN spin-orbit (sSO*chi1*xc**1.5) + 2PN spin-spin (sSS*chi1**2*xc**2)")
    print(f"  {N_COEF} coefficients, ALL fitted on the 2d grid, chi1 = 0 INCLUDED")
    print(f"  grid: {len(QS)} q x {len(CS)} chi1 = {len(QS)*len(CS)} points"
          f"{' (SUBSET)' if subset else ''}\n", flush=True)

    if args.promote:
        src = PN15_FINAL if PN15_FINAL.exists() else PN15_PARTIAL
        if not src.exists():
            sys.exit(f"nothing to promote: neither {PN15_FINAL.name} nor "
                     f"{PN15_PARTIAL.name} exists yet")
        d = json.loads(src.read_text())
        assert d["names"] == NAMES, f"coefficient order mismatch in {src}"
        th = np.array(d["theta"], float)
        stage = "converged" if src is PN15_FINAL else f"IN-FLIGHT (eval {d['n_eval']})"
        print(f"promoting from {src.name} [{stage}]")
        print("  " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)), flush=True)
        shutil.copy2(src, RESULTS / f"imported_{src.name}")
        cases = build_cases()
        errs, _ = score_all(th, cases)
        allm = tables(errs, cases, th)
        write_coeffs(th, errs, allm,
                     f"promoted 2026-09-17 from {src.name} [{stage}] on structural "
                     f"grounds: every spin term at a derivable PN order. Job 3512518 "
                     f"(pass 1) / 3512521 (pass 2), fit_pn_anchored_spin_pn15.py.")
        return

    if args.report:
        th, src = shipped11()
        print(f"coefficients from {src.name}")
        print("  " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)), flush=True)
        cases = build_cases()
        errs, _ = score_all(th, cases)
        tables(errs, cases, th)
        return

    if args.freeze9:
        # BASIS-VS-BASIS TEST.  Only (sSO, sSS) are free; the 9 nonspinning
        # coefficients are pinned to the shipped nonspinning pn_anchored fit, exactly
        # as the 4-coefficient predecessor pins them.  So the ONLY difference from the
        # predecessor is the spin basis -- 2 PN-ordered terms against 4 polynomial
        # ones -- with the grid, objective, drive and frozen base all identical.
        #
        # It also restores strict-extension behaviour: at chi1 = 0 both spin terms
        # vanish and beta_r_phys reduces, so with the 9 frozen the model is
        # bit-identical to nonspinning pn_anchored there.
        th9 = S.shipped9()
        s2 = (np.array(json.loads(FROZEN9.read_text())["theta"][9:], float)
              if FROZEN9.exists() and not args.fresh else seed_theta()[9:])
        print(f"FREEZE9: only (sSO, sSS) are free; the 9 are pinned to the shipped "
              f"nonspinning fit")
        print("  frozen 9: " + ", ".join(f"{n}={v:+.4f}"
                                         for n, v in zip(NAMES[:9], th9)))
        print(f"  start:    sSO={s2[0]:+.4f}, sSS={s2[1]:+.4f}", flush=True)

        cases = build_cases()
        nuis = {k: None for k in cases}
        st = {"n": 0, "best": np.inf, "bx": np.asarray(s2, float),
              "t0": time.time()}

        def obj2(x2):
            th = np.r_[th9, x2]
            errs, _ = score_all(th, cases, nuis)
            m, md = float(np.mean(errs)), float(np.median(errs))
            st["n"] += 1; el = time.time() - st["t0"]; tag = ""
            if m < st["best"]:
                st["best"] = m; st["bx"] = np.array(x2, float); tag = "  <-- best"
                FROZEN9_PARTIAL.write_text(json.dumps(
                    {"n_eval": st["n"], "elapsed_min": el / 60,
                     "names": SNAMES, "theta_spin": st["bx"].tolist(),
                     "mean": m, "median": md}, indent=2))
            print(f"  [{st['n']:4d}] {el/60:6.1f}m mean={m:.4e} median={md:.4e} "
                  f"max={errs.max():.4e}  sSO={x2[0]:+.4f} sSS={x2[1]:+.4f}{tag}",
                  flush=True)
            return m

        r = minimize(obj2, np.asarray(s2, float), method="Powell",
                     options={"maxiter": args.maxiter, "xtol": 1e-5, "ftol": 1e-7})
        s2 = st["bx"]
        print(f"  {r.message} / {st['n']} evals", flush=True)

        th = np.r_[th9, s2]
        print(f"\nfitted 2 of 11: sSO={s2[0]:+.4f}, sSS={s2[1]:+.4f}")
        errs, _ = score_all(th, cases)
        allm = tables(errs, cases, th)

        # the comparison this run exists for
        pred = json.loads(PRED.read_text())
        print("\n" + "=" * 72)
        print("BASIS VS BASIS -- both on the SAME frozen 9 nonspinning coefficients")
        print("=" * 72)
        print(f"{'':34} {'spin coef':>9} {'mean':>11} {'median':>11} {'max':>11}")
        pv = np.array(list(pred["mismatches"].values()))
        print(f"{'predecessor, polynomial basis':34} {4:>9} {pv.mean():11.4e} "
              f"{np.median(pv):11.4e} {pv.max():11.4e}")
        print(f"{'this run, PN-ordered basis':34} {2:>9} {errs.mean():11.4e} "
              f"{np.median(errs):11.4e} {errs.max():11.4e}")
        print(f"{'ratio (PN / polynomial)':34} {'':>9} {errs.mean()/pv.mean():11.2f} "
              f"{np.median(errs)/np.median(pv):11.2f} {errs.max()/pv.max():11.2f}")
        print("\nThis is the ONLY clean basis-vs-basis comparison: grid, objective,")
        print("drive and frozen base are identical, so the difference is the basis")
        print("alone.  The all-11-fitted numbers confound the basis change with the")
        print("1d->2d refit, which is worth -16.4% on the mean by itself.")

        write_coeffs(th, errs, allm,
                     f"--freeze9 basis-vs-basis test: 9 pinned to the shipped "
                     f"nonspinning pn_anchored fit, only (sSO, sSS) fitted. "
                     f"{r.message} / {st['n']} evals.",
                     path=FROZEN9, n_free=2)
        return

    if not args.refit:
        sys.exit("pick one of --promote / --report / --refit / --freeze9")

    # AUTO-RESUME.  seed_theta() is a POOR starting point (mean 6.06e-03 against the
    # predecessor's 1.28e-03), so a continuation run that silently restarted from it
    # would throw away the whole previous pass.  Resuming is the default whenever
    # coefficients exist; --fresh is the explicit opt-out.
    resume = COEFFS.exists() and not args.fresh
    th = (np.array(json.loads(COEFFS.read_text())["theta"], float)
          if resume else seed_theta())
    if resume and json.loads(COEFFS.read_text()).get("model_form") != S.MODEL_FORM:
        print(f"  note: {COEFFS.name} is from another model_form -- used as a SEED only")
    print(f"start: {'RESUMED from ' + COEFFS.name if resume else 'FRESH seed'}")
    print("seed: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)), flush=True)

    cases = build_cases(QS, CS)
    nproc = args.nproc or max(1, min(32, (os.cpu_count() or 4) - 2))
    if min(nproc, len(cases)) > 1:
        start_pool(cases, min(nproc, len(cases)))
    partial = SMALL_PARTIAL if subset else PARTIAL
    if subset:
        print(f"SUBSET {len(QS)} q x {len(CS)} chi1 = {len(cases)} points; writes "
              f"{SMALL.name} (not {COEFFS.name})", flush=True)
    nuis = {k: None for k in cases}
    st = {"n": 0, "best": np.inf, "bx": th.copy(), "t0": time.time()}

    class _Budget(Exception):
        pass

    def obj(x):
        errs, _ = score_all(x, cases, nuis)
        m, md = float(np.mean(errs)), float(np.median(errs))
        st["n"] += 1; el = time.time() - st["t0"]; tag = ""
        if m < st["best"]:
            st["best"] = m; st["bx"] = np.array(x, float); tag = "  <-- best"
            partial.write_text(json.dumps({"n_eval": st["n"], "elapsed_min": el / 60,
                                           "names": NAMES, "theta": st["bx"].tolist(),
                                           "mean": m, "median": md}, indent=2))
        print(f"  [{st['n']:4d}] {el/60:6.1f}m mean={m:.4e} median={md:.4e} "
              f"max={errs.max():.4e}{tag}", flush=True)
        if el > args.budget * 60:
            raise _Budget
        return m

    obj(th)
    st["t0"] = time.time(); st["n"] = 0
    try:
        r = minimize(obj, th, method="Powell",
                     options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
        th = st["bx"]; note = f"CONVERGED / {st['n']} evals: {r.message}"
    except _Budget:
        th = st["bx"]
        note = (f"STOPPED ON BUDGET / {st['n']} evals (best mean {st['best']:.4e}); "
                f"rerun with --resume")
    print(f"  {note}", flush=True)

    print("\nfitted: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)))
    errs, _ = score_all(th, cases)
    if subset:
        allm = {f"{q:.4f}_{c:+.1f}": float(e) for (q, c), e in zip(cases.keys(), errs)}
        for k, v in allm.items():
            print(f"  {k}: {v:.4e}")
        print(f"SUBSET {len(errs)}: mean {np.mean(errs):.4e}  median "
              f"{np.median(errs):.4e}  max {errs.max():.4e}")
        write_coeffs(th, errs, allm, f"fit_pn_anc_spin_physical.py --refit SUBSET: {note}",
                     path=SMALL, grid={"q": QS, "chi1": CS})
        return
    allm = tables(errs, cases, th)
    write_coeffs(th, errs, allm, f"fit_pn_anc_spin_physical.py --refit: {note}")


if __name__ == "__main__":
    main()
