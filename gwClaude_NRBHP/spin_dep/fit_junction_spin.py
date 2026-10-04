"""fit_junction_spin -- pn-anchored with the logistic switch replaced by a C1 JUNCTION.

5 fitted coefficients (b1, b2, a2, sSO, sSS) against the 11 of
`fit_pn_anc_spin_physical.py`.  The 6 that go are p00, p01, w0 (the switch), r0, m0,
m1 (the merger-ringdown branch), plus the XCLIP device.  Nothing replaces them with
another fit: the merger-ringdown branch is fixed by QNM theory plus the four junction
conditions.

===========================================================================
MODEL
===========================================================================
    h_NR(tau) = alpha(t) h_pp(t),   dtau/dt = beta(t)

    t_* = the ppBHPT |h| peak.  PARAMETER-FREE, and already this workspace's
          alignment anchor.  Measurement (junction_diag.py, 2026-09-23) shows
          beta has its MAXIMUM and alpha its MINIMUM there, so it is also the
          physically right place to hand over.

    INSPIRAL  t <= t_*      (no XCLIP -- the branch is not evaluated past t_*)
      beta_I  = base [1 + nu  (b1 x + b2 x^2 + sSO chi1 x^1.5 + sSS chi1^2 x^2)
                        + nu^2 (b21 x^2 + sSS1 chi1^2 x^2)]

    The nu^2 ORDER (b21, sSS1) added 2026-09-23.  These are NOT regressed nu-forms:
    nu^2 is a known function of q and its coefficients are nu-FLAT constants, so the
    q-dependence stays IMPOSED and nothing is extrapolated -- unlike e.g. fluxanchored's
    fitted `alpha_F = A0 nu + A1 nu^2`, which is -18% at q=2.
    WHICH coefficients get the nu^2 term was MEASURED, not chosen: the per-q trajectory
    sweep (job 3532375, junction_floor.py --mode traj, 7 q across [3,8]) gives, for the
    per-q OPTIMAL coefficients,
        coef   spread%   lin R^2   monotone
        sSS     1127.9     0.985      yes      -> nu^2 term  (sign change -5.99 -> +4.58)
        b2       579.7     0.921      yes      -> nu^2 term  (sign change +3.22 -> -2.73)
        b1       531.6     0.856      NO       -> SKIPPED, a line tilts the wrong way
        a2        19.9     0.924      yes      -> SKIPPED, barely moves
        sSO       17.7     0.964      yes      -> SKIPPED, barely moves
    So two terms, not five.  `a2` being the most nu-STABLE coefficient in the model is
    also why the `a32` experiment was misdirected.

    SEEDING: b21 = sSS1 = 0, i.e. this fit starts BIT-IDENTICAL to the converged 5-coef
    model, and the global joint fit must find the nu^2 order on its own.  It is
    deliberately NOT seeded from the trajectory fits, for two reasons: (a) the
    trajectories are the EVIDENCE that these two need nu structure, so using them as the
    starting point would make the global result partly just the seed re-emerging; (b)
    picking a start to avoid the compensated basin the master sits in (master b2 = -5.44
    lies OUTSIDE the per-q range [-2.73, +3.22]) would engineer around a property of the
    model instead of reporting it.  cf. fit_pn_anchored_spin.py's "WHY freeze" note,
    where an independent refit landed in a worse basin and the response was to declare
    it, not to re-seed.  If this fit stalls in that basin, THAT is the result, and the
    honest follow-up is a multi-start with the spread reported.
      alpha_I = base [1 + (55/42) nu x + nu a2 x^2]

    NO x^1.5 TERM IN ALPHA, and this was TESTED, not overlooked.  The PN note's Eq. (57)
    does carry `a_{3/2}(nu) x^{3/2}`, so it was added as `a32` and globally refit
    (job 3532284, 2026-09-23).  REJECTED: mean 3.7310e-03 -> 3.6048e-03, a 3.4% gain
    for a whole coefficient, and the hypothesis (that `a2` was inflated by absorbing
    1.5PN structure and would SHRINK) is falsified -- `a2` went -17.01 -> -39.31 while
    `a32` came in at +10.86, a large opposing pair that reproduces the single old term
    (at x=0.26: +1.440 and -2.657, sum -1.217, vs -1.150 before).  x^1.5 and x^2 are not
    separable over this x-range; cf. `wf_nu_fluxes`'s degenerate alpha_F/alpha_G.  The
    rejected fit is kept at junction_spin_results/coeffs_junction_6coef_a32_REJECTED.json.

    MERGER-RINGDOWN  t > t_*,  dt_ = t - t_*
      beta_M  = beta_r + exp(-dt_/tau)(C + D dt_)
      alpha_M = exp(kappa dt_) [A + B exp(-dt_/tau)]

    JUNCTION CONDITIONS -- the four Darmois conditions, equivalently C1 continuity
    of the energy and angular-momentum fluxes (see the derivation in the session
    notes).  They FIX C, D, A, B; none is fitted:
      [beta]  = 0 -> C = beta_I(t_*) - beta_r
      [beta'] = 0 -> D = betadot_I(t_*) + C/tau
      [alpha] = 0 -> A + B = alpha_I(t_*)
      [alpha']= 0 -> B = tau (kappa alpha_I(t_*) - alphadot_I(t_*)),  A = alpha_I - B

    The (C + D dt_) form -- rather than a bare exponential -- is what lets beta
    arrive RISING (betadot_I(t_*) > 0), turn over, and still settle on beta_r.  A
    single decaying exponential cannot, which is why the earlier root-finding
    formulation failed at 13/24 grid points.

===========================================================================
DERIVED BLOCK -- zero fitted parameters
===========================================================================
  w220(chi), Q220(chi)                            exact Kerr (2,2,0) frequency and quality
           factor, physics_common.w220_exact / Q220_exact (qnm package, Leaver continued
           fraction; counter-rotating branch for chi < 0), via junction_diag.  Until
           2026-09-26 these were the Berti-Cardoso-Will fits 1.5251 - 1.1568 (1-chi)^0.1292
           and 0.7000 + 1.4187 (1-chi)^(-0.4990).
  M_f, chi_f                                      surfinBH NRSur3dq8Remnant

  w_pp   = w220(chi1)          the pp background IS the primary, mass m1, spin chi1, on
           the ppBHPT clock (primary-mass units) -- NO (1+q)/q.  Exact Kerr
           (physics_common.w220_exact) on both sides of beta_r since 2026-09-26; checked
           against the measured pp ringdown at q=5 to 0.1-0.7% for chi1 in [-0.2, 0.6].
           fit_pn_anchored_spin.beta_r_phys uses the same form (fixed 2026-09-24).
  beta_r = w_pp / (w220(chi_f)/M_f)               the ringdown time-stretch anchor
  tau    = tau_f_pp = (2 Q_f / w_f) / beta_r       fundamental damping time, converted
           from the NR axis to the ppBHPT axis (UNITS_AUDIT A3).
           *** THE CHOICE OF tau IS NOT SETTLED BY MEASUREMENT (corrected 2026-09-24). ***
           The T2 table reported median tau_meas/tau_f = 0.97 and concluded "the data
           picks the fundamental" -- but that compared a measurement made on pp-axis
           peak times against tau_f, an NR-axis duration.  Redone on a COMMON axis:
           tau_meas/tau_f_pp has median 0.90, range 0.64-9.6, against tau_f_pp 13-16 M
           and tau_ov_pp 6.3-8.0 M.  The measurement sits BETWEEN the two candidates,
           closer to the fundamental, and its scatter is far too wide to separate them.
           tau_f_pp is retained as the better of the two, NOT as a measured result.
  kappa  = (w_pp/2)(1/Q_pp - 1/Q_f)               alpha's ringdown log-slope.
           beta matches the two QNM REAL frequencies; it cannot also match the
           imaginary parts, and kappa is the leftover decay-rate mismatch that
           alpha must carry.  VERIFIED: sign correct in 21/21 clean grid points,
           median measured/derived 1.09.  Q_f > Q_pp always, so kappa > 0 and
           alpha RISES through ringdown -- which a constant alpha_mr cannot do.

Spin enters the whole merger-ringdown sector through chi_f(q,chi1) and chi1 with NO
fitted coefficient, which is the gap `history.md` flags ("alpha's merger spin
behaviour, never measured ... spin-blind by fiat").

===========================================================================
OBJECTIVE AND GRID -- identical to fit_pn_anc_spin_physical, for comparability
===========================================================================
Mean over the grid of the per-case waveform mismatch, window [-5000.1, +100] M,
phi0 analytic, t0 a per-case nuisance.  Grid = S.TRAIN_Q (12 q in [3,8]) x S.CHIS
(8 chi1 in [-0.6,+0.8]) = 96 points.

chi1 = -0.6 CAVEAT: the post-merger |h| envelope carries 15-16 genuine nulls there
(see the FREEZE_XC note in fit_pn_anchored_spin.py); junction_diag finds ZERO usable
post-merger peaks at chi1=-0.6 for all three q.  Those rows are fitted and scored
like any other, but the derived branch is not measurable there -- read them with care.
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os, sys, time as _time, warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import fit_pn_anchored_spin as S
import fit_scaling_PN_opt_creative as creative
import junction_diag as J

NAMES = ["b1", "b2", "a2", "sSO", "sSS", "b21", "sSS1"]
# job 3530305's converged 5-coef result, with the two nu^2 coefficients at ZERO.
SEED = np.array([0.40089, -5.43646, -17.00576, 3.33094, -0.45003, 0.0, 0.0])
RESULTS = HERE / "junction_spin_results"; RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs_junction.json"
PARTIAL = COEFFS.with_name("coeffs_junction_partial.json")

# False reproduces the 2026-09-23 runs (tau on the NR axis -- mis-specified by 16-34%).
TAU_PP = True

_DER: dict = {}


def der(q, chi1):
    k = (round(q, 6), round(chi1, 6))
    if k not in _DER:
        _DER[k] = J.derived(q, chi1)
    return _DER[k]


def model_ab(theta, q, chi1, t, x, i_star):
    """alpha(t), beta(t) with the C1 junction at t[i_star]."""
    b1, b2, a2, sSO, sSS, b21, sSS1 = theta
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    # x arrives as x_pp (ppBHPT clock); the PN terms (55/42 slope, x^1.5 spin-orbit, x^2) are
    # defined for the note's x = x_pp * X1^-0.8 (Eqs. 6, 8, 12).  Fixed 2026-09-26.
    x = S.PC.x_scale(q) * np.asarray(x, float)
    d = der(q, chi1)
    # tau must be a duration on the ppBHPT time axis, because dt_ below is pp time.
    # d["tau_f"] = 1/w_I,f is an NR-time duration; dividing by beta_r converts it.
    # kappa needs no conversion (already per unit pp time).  See junction_diag.derived.
    beta_r, kappa = d["beta_r"], d["kappa"]
    tau = d["tau_f_pp"] if TAU_PP else d["tau_f"]

    beta_I = base * (1 + nu * (b1 * x + b2 * x ** 2
                               + sSO * chi1 * x ** 1.5 + sSS * chi1 ** 2 * x ** 2)
                     + nu ** 2 * (b21 * x ** 2 + sSS1 * chi1 ** 2 * x ** 2))
    alpha_I = base * (1 + S.PN_SLOPE * nu * x + nu * a2 * x ** 2)
    dbI = np.gradient(beta_I, t); daI = np.gradient(alpha_I, t)

    bs, dbs = float(beta_I[i_star]), float(dbI[i_star])
    as_, das = float(alpha_I[i_star]), float(daI[i_star])

    C = bs - beta_r
    D = dbs + C / tau
    B = tau * (kappa * as_ - das)
    A = as_ - B

    dt_ = t - t[i_star]
    beta = beta_I.copy(); alpha = alpha_I.copy()
    m = dt_ > 0.0
    e = np.exp(-dt_[m] / tau)
    beta[m] = beta_r + e * (C + D * dt_[m])
    alpha[m] = np.exp(kappa * dt_[m]) * (A + B * e)
    return alpha, beta


def mismatch(theta, q, chi1, case, key, seed):
    p = "fit_" if key == "fit" else ""
    t, h = case[p + "t_bhpt"], case[p + "h_bhpt"]
    tn, hn = case[p + "t_nr"], case[p + "h_nr"]
    x = case[p + "x"]
    i_star = int(np.argmax(np.abs(h)))
    alpha, beta = model_ab(theta, q, chi1, t, x, i_star)
    if not np.all(np.isfinite(beta)) or np.min(beta) <= 1e-3:
        return 50.0, seed
    bc = creative.cumulative_trapezoid(beta, t)
    ts = bc - np.interp(S.T_ANCHOR, t, bc)
    dmin = np.min(np.diff(ts))
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), seed
    if not np.all(np.isfinite(alpha)) or np.max(np.abs(alpha)) > 1e3:
        return 50.0, seed
    grid = np.linspace(seed - 200, seed + 200, 81)
    e = [S._err_at_t0(x0, alpha, ts, t, h, tn, hn) for x0 in grid]
    c = grid[int(np.argmin(e))]
    r = minimize_scalar(lambda x0: S._err_at_t0(x0, alpha, ts, t, h, tn, hn),
                        bounds=(c - 6, c + 6), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 60})
    return float(r.fun), float(r.x)


def full_mm(theta, q, chi1, case, seed=-50.0):
    _, t0 = mismatch(theta, q, chi1, case, "fit", seed)
    e, _ = mismatch(theta, q, chi1, case, "full", t0)
    return e


# ---------------------------------------------------------------------------
# Grid points are independent, so the objective forks over them.  Measured
# 2026-09-23: 30 s/eval serial on skx (96 pairs x ~140 _err_at_t0 calls each),
# which makes a Powell run to convergence (~1000 evals) an 8 h job.  A fork Pool
# over the grid -- the pattern mismatch_heatmap_dense.py already uses here --
# takes that to ~2 s/eval on a 48-core node.  `_CASES` and the `_DER` cache are
# populated in the parent BEFORE the fork, so children inherit them copy-on-write
# and never touch surfinBH or the waveform cache themselves.
_CASES: dict = {}


def _work(arg):
    th, q, c = arg
    return full_mm(np.asarray(th, float), q, c, _CASES[(q, c)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--nproc", type=int, default=0, help="0 = auto (cores-2, cap 24)")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="wall-clock minutes; stop cleanly and keep best-so-far")
    ap.add_argument("--q", type=float, nargs="*", default=None)
    ap.add_argument("--chi", type=float, nargs="*", default=None)
    a = ap.parse_args()

    QS = a.q if a.q else S.TRAIN_Q
    CS = a.chi if a.chi else S.CHIS
    print(f"grid: {len(QS)} q x {len(CS)} chi1 = {len(QS)*len(CS)} points", flush=True)

    for q in QS:
        for c in CS:
            _CASES[(q, c)] = S.build_case(q, c)
            der(q, c)                     # populate the derived cache before forking
    cases = _CASES
    keys = [(q, c) for q in QS for c in CS]
    nproc = a.nproc or max(1, min(24, (os.cpu_count() or 4) - 2))
    nproc = min(nproc, len(keys))
    print(f"cases built; {len(keys)} grid points on {nproc} workers", flush=True)
    pool = mp.get_context("fork").Pool(nproc) if nproc > 1 else None

    def grid_mean(th):
        if pool is None:
            return float(np.mean([full_mm(th, q, c, cases[(q, c)]) for q, c in keys]))
        return float(np.mean(pool.map(_work, [(th, q, c) for q, c in keys])))

    # Warm start from whatever is on disk, refit or not, so a short skx-dev pass and a
    # long skx pass chain instead of each restarting from SEED.
    theta = SEED.copy()
    # Pick the BEST available warm start by its stored grid mean, not the last file read
    # (audit 2026-09-26: COEFFS used to override a better PARTIAL from a killed run).
    cands = []
    for src in (PARTIAL, COEFFS):
        if src.exists():
            try:
                d = json.loads(src.read_text())
                th = np.array(d["theta"], float)
                if len(th) < len(NAMES):       # older file: new nu^2 terms start at 0
                    th = np.r_[th, np.zeros(len(NAMES) - len(th))]
                if len(th) != len(NAMES):
                    raise ValueError(f"expected {len(NAMES)} coefficients, got {len(th)}")
                m = d.get("mean", d.get("mismatch", {}).get("mean", np.inf))
                cands.append((float(m), src.name, th))
            except Exception as e:
                print(f"  (ignoring {src.name}: {e})", flush=True)
    if cands:
        m, name, theta = min(cands, key=lambda c: c[0])
        print(f"warm start from {name} (stored mean {m:.4e}): "
              f"{np.array2string(theta, precision=4)}", flush=True)

    if a.refit:
        t0 = _time.time(); st = {"n": 0, "best": np.inf, "bx": theta.copy()}

        class _Budget(Exception):
            pass

        def obj(th):
            if a.budget and (_time.time() - t0) > a.budget * 60.0:
                raise _Budget
            v = grid_mean(np.asarray(th, float))
            st["n"] += 1
            if v < st["best"]:
                st["best"] = v; st["bx"] = np.asarray(th, float).copy()
                PARTIAL.write_text(json.dumps(
                    {"theta": st["bx"].tolist(), "mean": v, "n_eval": st["n"]}, indent=1))
            if st["n"] % 20 == 0:
                print(f"  [{st['n']:>4d}] {_time.time()-t0:>7.0f}s  mean={v:.6e}  "
                      f"best={st['best']:.6e}  {np.array2string(np.asarray(th), precision=4)}",
                      flush=True)
            return v

        try:
            r = minimize(obj, theta, method="Powell",
                         options={"maxiter": a.maxiter, "xtol": 1e-3, "ftol": 1e-4})
            theta = r.x
            print(f"\nrefit done: mean={r.fun:.6e}  "
                  f"{np.array2string(theta, precision=5)}", flush=True)
        except _Budget:
            theta = st["bx"]
            print(f"\nBUDGET {a.budget:.0f} min reached after {st['n']} evals; "
                  f"keeping best mean={st['best']:.6e}  "
                  f"{np.array2string(theta, precision=5)}", flush=True)

    if pool is not None:
        pool.close(); pool.join()
    allm = {}
    print(f"\n{'q':>8} " + " ".join(f"{c:>+9.1f}" for c in CS))
    for q in QS:
        row = [full_mm(theta, q, c, cases[(q, c)]) for c in CS]
        for c, v in zip(CS, row):
            allm[f"{q:.4f}_{c:+.1f}"] = v
        print(f"{q:>8.4f} " + " ".join(f"{v:>9.3e}" for v in row), flush=True)

    v = np.array(list(allm.values()))
    print(f"\nmean {v.mean():.4e}   median {np.median(v):.4e}   max {v.max():.4e}")
    if 0.0 in [float(c) for c in CS]:          # subset grids need not contain chi1 = 0 (fixed 2026-09-26)
        chi0 = np.array([allm[f"{q:.4f}_{0.0:+.1f}"] for q in QS])
        print(f"chi1=0 column: median {np.median(chi0):.4e}  max {chi0.max():.4e}")

    # Only a --refit on the FULL grid may replace coeffs_junction.json; anything else (a
    # rescore, a subset) goes to its own file (audit 2026-09-26: a subset rescore used to
    # overwrite the full-grid statistics).  The settings that define the model are recorded
    # so the file can be checked against the code later.
    full = (list(QS) == list(S.TRAIN_Q)) and (list(CS) == list(S.CHIS))
    out = COEFFS if (a.refit and full) else COEFFS.with_name(
        "coeffs_junction_rescore.json" if full else "coeffs_junction_subset.json")
    out.write_text(json.dumps({
        "model": "junction_spin",
        "model_form": S.MODEL_FORM + "; junction C1 at |h| peak, tau_f/beta_r",
        "n_coefficients": len(NAMES),
        "names": NAMES, "theta": theta.tolist(),
        "derived": "beta_r = w_pp(chi1)*M_f/w220(chi_f), kappa, "
                   + ("tau = tau_f/beta_r (pp axis)" if TAU_PP else "tau = tau_f (NR axis)")
                   + "; t_* = |h| peak",
        "settings": {"TAU_PP": bool(TAU_PP), "RESCALE_QNM": getattr(S, "RESCALE_QNM", getattr(J, "RESCALE_QNM", None)),
                     "beta_r_form": "w_pp(chi1)*M_f/w220(chi_f)  [2026-09-24 units fix]",
                     "refit": bool(a.refit), "full_grid": bool(full)},
        "grid": {"q": list(QS), "chi1": list(CS)},
        "mismatch": {"mean": float(v.mean()), "median": float(np.median(v)),
                     "max": float(v.max())},
        "per_case": allm}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
