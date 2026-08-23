"""
gw_remnant energy-driven calibration (gw_remnant_energy).

PHYSICALLY MOTIVATED, SWITCHLESS.  The only time-dependence in the scaling comes
from the mass the binary has lost to gravitational waves.  The cumulative
radiated energy E(t) is taken from `gw_remnant`
(https://github.com/tousifislam/gw_remnant), the same quantity plotted by
`RemnantMassCalculator.plot_mass_energy()`:

    dE/dt = (1/16 pi) * sum_{lm} |dh_lm/dt|^2          (Eq. 2 of 1802.04276)
    E(t)  = int_{t_0}^{t} dE/dt dt'                    (Eoft,  units of M)
    M(t)  = (M_initial - E_initial) - E(t)             (Moft,  Bondi mass)

evaluated on the *BHPT* waveform (the model input), with the (2,2) and (2,-2)
modes, E_initial = L_initial = 0 and M_initial = 1, so M(t) = 1 - E(t).

Referenced to the time-map anchor:

    dE(t)    = E(t) - E(T_ANCHOR)                >= 0, monotone
    m_hat(t) = M(t) / M(T_ANCHOR) = 1 - dE/M(T_ANCHOR)

Two model forms, both 6 parameters, NO logistic switch and NO angular-momentum
coordinate:

  form="linear"      alpha = alpha_i + alpha_E * dE
  (default)          beta  = beta_i  + beta_E  * dE

  form="mass_power"  alpha = alpha_i * m_hat ** alpha_p
                     beta  = beta_i  * m_hat ** beta_p

    tau(t)          = t0_nr + int_{T_ANCHOR}^{t} beta(t') dt'
    h_model(tau(t)) = alpha(t) * exp(i*phi0) * h_BHPT(t)

Why this is the physical ansatz.  BHPT evolves a *fixed* background mass; the
real binary sheds mass-energy, so the natural leading-order correction rescales
both the strain and the clock by M(t)/M_0.  That prediction is
alpha_E/alpha_i = beta_E/beta_i = -1/M(T_ANCHOR) (linear form), or
alpha_p = beta_p = 1 (mass_power form) — both are reported as diagnostics.

Why raw E(t) rather than a normalised flux coordinate.  E(t) carries its own
nu-scaling (E_rad ~ nu^2), so the couplings alpha_E, beta_E are expected to be
much flatter in nu than the coefficients of a peak-normalised coordinate.  They
are therefore left UNANCHORED in the regression: the PP limit is enforced by
dE -> 0, not by the coefficient.  Anchors alpha_i, beta_i -> 1 as nu -> 0.

Contrast with wf_nu_switchless (8 params, alpha_E/alpha_J and beta_E/beta_J on
peak-normalised e_hat, j_hat): that model converged per-q (median 5.1e-4) but
FAILED as a master (median 8.8e-3) because the two flux couplings are mutually
degenerate and scatter non-monotonically in nu.  Dropping dJ removes exactly
that degeneracy.

Training range [3, 8]; q < 3 is pure extrapolation, reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTNRSurrogate" / "surrogates"))

warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_wf_nu_q_dep as wfnu

from gw_remnant.remnant_calculators.remnant_mass_calculator import (
    RemnantMassCalculator,
)

# regression helpers are shared with the other nu-basis models
get_nu            = wfnu.get_nu
fit_poly_anchored = wfnu.fit_poly_anchored
eval_poly         = wfnu.eval_poly

MODE               = creative.MODE
T_ANCHOR           = creative.T_ANCHOR
MIN_COVERAGE       = qdep.MIN_COVERAGE
MASTER_OUTLIER_CUT = qdep.MASTER_OUTLIER_CUT
WAVEFORM_CACHE_DIR = qdep.WAVEFORM_CACHE_DIR

RESULTS_DIR = ROOT / "gw_remnant_energy_results"
RESULTS_DIR.mkdir(exist_ok=True)
MD_PATH = ROOT / "scaling_gw_remnant_energy.md"

SWITCHLESS_CACHE_PATH = ROOT / "wf_nu_switchless_results" / "per_q_cache.json"

LOW_Q = (2.75, 2.5, 2.25, 2.0)
DEGREES = (2, 3, 4)

FORMS = {
    "mult":       ["alpha_PP", "alpha_E", "beta_PP", "beta_E", "t0_nr", "phi0"],
    "linear":     ["alpha_i", "alpha_E", "beta_i", "beta_E", "t0_nr", "phi0"],
    "linear_nu":  ["alpha_i", "alpha_A", "beta_i", "beta_B", "t0_nr", "phi0"],
    "mass_power": ["alpha_i", "alpha_p", "beta_i", "beta_p", "t0_nr", "phi0"],
}

# Waveforms below this q are outside BHPTNRSur1dq1e4's declared domain
# (BHPTNRSur1dq1e4.py: X_min = log10(2.5)); the surrogate only prints a warning
# and extrapolates its own splines in log q.  Reported, but not treated as a gate.
BHPT_Q_MIN = 2.5

# form "linear_nu": identical model to "linear", but dE is divided by a power of
# nu before it multiplies the coupling, so that the fitted coefficient is the
# quantity that actually regresses.  Measured on the [3,8] per-q solutions of
# form "linear":
#     alpha_E * nu    varies by 1.20x   (alpha_E alone: 2.28x)
#     beta_E  * nu^2  is smooth         (beta_E alone changes SIGN, 79x spread)
# i.e. the amplitude correction per unit radiated energy scales as 1/nu and the
# clock drift as 1/nu^2.  Since dE ~ nu^2, that says the beta drift amplitude is
# nu-independent and the alpha correction scales as nu.
# nu is normalised by its value at the middle of the training range so the
# rescaled couplings stay O(1) and Nelder-Mead sees the same conditioning as the
# unscaled form (at q=5 the two parameterisations coincide exactly).
NU_POW = {"alpha": 1.0, "beta": 2.0}
NU_REF = 5.0 / 36.0          # get_nu(5.0)

# nu -> 0 (test-particle) anchors.  Couplings are FREE: dE -> 0 and m_hat -> 1
# already enforce the PP limit, so pinning them would double-count.
PP_ANCHORS = {
    # form "mult": alpha = alpha_PP(nu) * (1 + alpha_E(nu)*E(t)).  Only the
    # prefactors are anchored -- the correction term vanishes on its own because
    # E_rad -> 0 as nu -> 0 (measured E_tot ~ nu^2.31 over q in [3,10], while
    # alpha_E ~ -1.18/nu, so the product goes as nu^1.3).  The PP limit is
    # therefore carried by the coordinate, not imposed on the coefficient.
    "mult":       {"alpha_PP": 1.0, "alpha_E": None, "beta_PP": 1.0,
                   "beta_E": None, "t0_nr": None, "phi0": None},
    "linear":     {"alpha_i": 1.0, "alpha_E": None, "beta_i": 1.0,
                   "beta_E": None, "t0_nr": None, "phi0": None},
    "linear_nu":  {"alpha_i": 1.0, "alpha_A": None, "beta_i": 1.0,
                   "beta_B": None, "t0_nr": None, "phi0": None},
    "mass_power": {"alpha_i": 1.0, "alpha_p": None, "beta_i": 1.0,
                   "beta_p": None, "t0_nr": None, "phi0": None},
}


def results_paths(form: str) -> tuple[Path, Path]:
    suffix = "" if form == "linear" else f"_{form}"
    return (RESULTS_DIR / f"per_q_cache{suffix}.json",
            RESULTS_DIR / f"coeffs{suffix}.json")


# ---------------------------------------------------------------------------
# gw_remnant energy coordinates
# ---------------------------------------------------------------------------

def gwr_energy_coordinates(t_bhpt: np.ndarray, h_bhpt: np.ndarray, q: float) -> dict:
    """
    Cumulative radiated energy E(t) and Bondi mass M(t) from gw_remnant.

    The BHPT (2,2) mode plus its (2,-2) partner (h_{2,-2} = conj(h_{2,2}) for a
    non-precessing binary) are handed to `RemnantMassCalculator`, which is the
    object behind `plot_mass_energy()`.  E_initial = L_initial = 0 so that Eoft
    is the pure radiated energy and Moft = 1 - Eoft.
    """
    h_dict = {MODE: h_bhpt, (MODE[0], -MODE[1]): np.conj(h_bhpt)}
    calc = RemnantMassCalculator(
        time=t_bhpt, h_dict=h_dict, q=q,
        E_initial=0.0, L_initial=0.0, M_initial=1.0, use_filter=False,
    )

    e_oft = np.asarray(calc.Eoft, dtype=float)     # radiated energy, units of M
    m_oft = np.asarray(calc.Moft, dtype=float)     # Bondi mass, units of M

    e_anchor = float(np.interp(T_ANCHOR, t_bhpt, e_oft))
    m_anchor = float(np.interp(T_ANCHOR, t_bhpt, m_oft))
    if m_anchor <= 0.0:
        raise ValueError(f"Non-positive Bondi mass at T_ANCHOR for q={q}.")

    # Peak-normalised scale used by wf_nu_switchless, kept only so its cached
    # solutions can be converted into seeds for this parameterisation.
    merger_index = int(np.argmax(np.abs(h_bhpt)))
    e_ref = abs(e_anchor - float(e_oft[merger_index]))

    d_e = e_oft - e_anchor
    nu = get_nu(q)
    nu_hat = nu / NU_REF

    return {
        "e_oft":    e_oft,
        "m_oft":    m_oft,
        "dE":       d_e,
        "dE_a":     d_e / nu_hat ** NU_POW["alpha"],   # form "linear_nu"
        "dE_b":     d_e / nu_hat ** NU_POW["beta"],
        "nu":       nu,
        "nu_hat":   nu_hat,
        "m_hat":    m_oft / m_anchor,
        "e_anchor": e_anchor,
        "m_anchor": m_anchor,
        "e_rad":    float(calc.E_rad),
        "e_ref":    e_ref,
    }


def load_case(q: float, source_stride: int = 3, nr_stride: int = 5) -> dict:
    """Load the shared waveform cache and attach gw_remnant energy coordinates."""
    d = np.load(qdep.waveform_cache_path(q))
    t_bhpt = d["t_bhpt"]
    h_bhpt = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    t_nr = d["t_nr"]
    h_nr = d["h_nr_re"] + 1j * d["h_nr_im"]
    meta = {
        "nu":            float(d["nu"]),
        "t_bhpt_merger": float(d["t_bhpt_merger"]),
        "t_nr_merger":   float(d["t_nr_merger"]),
    }

    losses_full = gwr_energy_coordinates(t_bhpt, h_bhpt, q)

    src_idx = np.unique(np.r_[np.arange(0, len(t_bhpt), source_stride), len(t_bhpt) - 1])
    nr_idx  = np.unique(np.r_[np.arange(0, len(t_nr), nr_stride), len(t_nr) - 1])

    return {
        "q":          q,
        "t_bhpt":     t_bhpt,
        "h_bhpt":     h_bhpt,
        "t_nr":       t_nr,
        "h_nr":       h_nr,
        "meta":       meta,
        "losses":     losses_full,
        "fit_t_bhpt": t_bhpt[src_idx],
        "fit_h_bhpt": h_bhpt[src_idx],
        "fit_losses": qdep.subsample_losses(losses_full, src_idx),
        "fit_t_nr":   t_nr[nr_idx],
        "fit_h_nr":   h_nr[nr_idx],
    }


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def alpha_beta(params, losses, form):
    a_i, a_c, b_i, b_c = (float(v) for v in params[:4])
    if form == "mult":
        # RAW gw_remnant E(t): Eoft[0] == 0 exactly, so alpha_PP is the scaling
        # at the start of the surrogate window and no reference subtraction is
        # needed to make the PP limit come out right.
        e = losses["e_oft"]
        return a_i * (1.0 + a_c * e), b_i * (1.0 + b_c * e)
    if form == "linear":
        dE = losses["dE"]
        return a_i + a_c * dE, b_i + b_c * dE
    if form == "linear_nu":
        return a_i + a_c * losses["dE_a"], b_i + b_c * losses["dE_b"]
    m_hat = losses["m_hat"]
    return a_i * m_hat ** a_c, b_i * m_hat ** b_c


def evaluate_model(params, t_bhpt, h_bhpt, t_nr, h_nr, losses,
                   form="linear", min_coverage=MIN_COVERAGE, analytic_phi0=True):
    """
    Mismatch of the scaled BHPT waveform against NR.

    With `analytic_phi0` (the default) the constant phase is not a fitted
    parameter: for h_model = exp(i*phi0) * h0 the overlap is
    Re[exp(-i*phi0) * sum(h_ref * conj(h0))], which is maximised in closed form
    at phi0 = arg(sum(h_ref * conj(h0))), and the norms do not depend on phi0.
    Fitting it instead is not just wasteful — the (t0_nr, phi0) alignment valley
    is so flat that the per-q optima scatter by O(pi) between adjacent q, which
    no amount of unwrapping makes regressable in nu.
    """
    t0_nr = float(params[4])

    alpha, beta = alpha_beta(params, losses, form)

    if np.any(beta <= 0) or not np.all(np.isfinite(beta)):
        return {"error": 50.0}
    if not np.all(np.isfinite(alpha)):
        return {"error": 50.0}

    beta_cum   = creative.cumulative_trapezoid(beta, t_bhpt)
    anchor_val = float(np.interp(T_ANCHOR, t_bhpt, beta_cum))
    tau = t0_nr + beta_cum - anchor_val

    if not np.all(np.diff(tau) > 0):
        return {"error": 50.0}

    t_min = max(t_nr[0], tau[0])
    t_max = min(t_nr[-1], tau[-1])
    if t_max - t_min < 0:
        return {"error": 50.0}

    nr_mask  = (t_nr >= t_min) & (t_nr <= t_max)
    coverage = nr_mask.sum() / len(t_nr)
    if coverage < min_coverage:
        return {"error": 50.0}

    t_common = t_nr[nr_mask]
    h_ref    = h_nr[nr_mask]

    h_bhpt_r = np.interp(t_common, tau, h_bhpt.real)
    h_bhpt_i = np.interp(t_common, tau, h_bhpt.imag)
    alpha_c  = np.interp(t_common, tau, alpha)

    h0 = alpha_c * (h_bhpt_r + 1j * h_bhpt_i)

    z = np.sum(h_ref * h0.conjugate())
    if analytic_phi0:
        phi0 = float(np.angle(z))
        sd = float(np.abs(z))
    else:
        phi0 = float(params[5])
        sd = float(np.real(z * np.exp(-1j * phi0)))

    h_model = h0 * np.exp(1j * phi0)

    n1 = np.sum(np.abs(h_ref) ** 2)
    n2 = np.sum(np.abs(h0) ** 2)

    return {
        "error":    float(((n1 + n2) - 2.0 * sd) / (2.0 * n1)),
        "coverage": float(coverage),
        "phi0":     phi0,
        "tau":      tau,
        "alpha":    alpha,
        "beta":     beta,
        "h_ref":    h_ref,
        "h_model":  h_model,
        "common":   nr_mask,
    }


# ---------------------------------------------------------------------------
# Per-q optimisation
# ---------------------------------------------------------------------------

def canonicalize(params, form):
    """
    Remove the two gauge freedoms before regression.

    1. Sign symmetry: h_model = alpha*exp(i*phi0)*h is invariant under
       alpha -> -alpha, phi0 -> phi0 + pi.  Nelder-Mead lands on either branch
       (e.g. q=4.154 came back with alpha_i = -0.755), which puts a step
       discontinuity into an otherwise smooth alpha_i(nu).  Force alpha_i > 0.
    2. phi0 branch: reduce to (-pi, pi] so np.unwrap has a slowly-varying
       sequence to work with instead of arbitrary 2*pi offsets.
    """
    p = np.array(params, dtype=float)
    if p[0] < 0:
        p[0] = -p[0]
        if form in ("linear", "linear_nu"):
            p[1] = -p[1]        # alpha = alpha_i + alpha_E*dE flips wholesale
        p[5] += np.pi           # mass_power: alpha_i*m_hat**p, exponent unchanged
    p[5] = creative.principal_phase(p[5])
    return p


def _load_switchless_cache():
    if not SWITCHLESS_CACHE_PATH.exists():
        return {}
    return json.loads(SWITCHLESS_CACHE_PATH.read_text())


def _switchless_seed(p8, e_ref, form):
    """
    Convert a wf_nu_switchless solution into this parameterisation.

    switchless: [alpha_i, alpha_E, alpha_J, beta_i, beta_E, beta_J, t0_nr, phi0]
    on peak-normalised coordinates with dE_sw = dE_raw / e_ref (and dJ_sw of very
    similar shape), so the combined coupling maps as (c_E + c_J) / e_ref.
    """
    a_i, a_E, a_J, b_i, b_E, b_J, t0, ph = (float(v) for v in p8)
    a_c = (a_E + a_J) / max(e_ref, 1e-12)
    b_c = (b_E + b_J) / max(e_ref, 1e-12)
    if form == "mass_power":
        # alpha_i*(1 - dE/M)^p ~ alpha_i - alpha_i*p*dE/M  =>  p = -a_c*M/alpha_i
        a_c = -a_c / max(abs(a_i), 1e-6)
        b_c = -b_c / max(abs(b_i), 1e-6)
    return np.array([a_i, a_c, b_i, b_c, t0, ph])


def _linear_to_linear_nu(p6, nu_hat):
    """Exact reparameterisation: same model, rescaled couplings."""
    p = np.array(p6, dtype=float)
    p[1] *= nu_hat ** NU_POW["alpha"]
    p[3] *= nu_hat ** NU_POW["beta"]
    return p


def _linear_to_mult(p6, e_anchor):
    """
    Exact reparameterisation of form "linear" into form "mult".

        linear:  alpha_i + alpha_E*(E - E_anchor)
        mult:    alpha_PP*(1 + alpha_E'*E)
      =>  alpha_PP = alpha_i - alpha_E*E_anchor,   alpha_E' = alpha_E/alpha_PP
    """
    a_i, a_E, b_i, b_E, t0, ph = (float(v) for v in p6)
    a_pp = a_i - a_E * e_anchor
    b_pp = b_i - b_E * e_anchor
    return np.array([
        a_pp, a_E / a_pp if a_pp else 0.0,
        b_pp, b_E / b_pp if b_pp else 0.0,
        t0, ph,
    ])


def _seeds(q, case, form, n_random=120):
    meta   = case["meta"]
    losses = case["fit_losses"]
    m_anchor = losses["m_anchor"]
    nu_hat   = losses["nu_hat"]
    seeds = []

    # forms "linear_nu" and "mult" are the same model as "linear" written
    # differently; if that fit already ran, its solutions transform exactly and
    # are the ideal seeds.
    if form in ("linear_nu", "mult"):
        lin_path, _ = results_paths("linear")
        if lin_path.exists():
            lin = json.loads(lin_path.read_text())
            for qk in (q, 3.0, 5.0, 8.0):
                key = f"{qk:.10f}"
                if key in lin:
                    p = lin[key]["params"]
                    seeds.append(_linear_to_linear_nu(p, nu_hat) if form == "linear_nu"
                                 else _linear_to_mult(p, losses["e_anchor"]))

    sw_cache = _load_switchless_cache()
    for qk in (q, 3.0, 5.0, 8.0):
        key = f"{qk:.10f}"
        if key in sw_cache:
            s = _switchless_seed(sw_cache[key]["params"], losses["e_ref"],
                                 "linear" if form == "linear_nu" else form)
            if form == "linear_nu":
                s = _linear_to_linear_nu(s, nu_hat)
            seeds.append(s)

    scale = (q / (1.0 + q)) / (5.0 / 6.0)

    # Physical prediction: pure mass rescaling of both strain and clock.
    for a0, b0 in ((scale, scale), (scale, 1.0), (1.0, 1.0)):
        t0 = meta["t_nr_merger"] - b0 * (meta["t_bhpt_merger"] - T_ANCHOR)
        if form == "mass_power":
            a_c, b_c = 1.0, 1.0
        elif form == "mult":
            a_c, b_c = -1.0, -1.0     # pure mass rescaling: 1 + a_c*E = M(t)/M0
        else:
            a_c, b_c = -a0 / m_anchor, -b0 / m_anchor
            if form == "linear_nu":
                a_c *= nu_hat ** NU_POW["alpha"]
                b_c *= nu_hat ** NU_POW["beta"]
        seeds.append(np.array([a0, a_c, b0, b_c, t0, 0.0]))

    rng = np.random.default_rng(int(q * 977) % (2 ** 31))
    # linear couplings live on ~1/E ~ O(10-100); exponents on O(1-30).
    # linear_nu coincides with linear at q=5, so the same ranges cover it.
    c_lo, c_hi = (-10.0, 30.0) if form == "mass_power" else (-120.0, 40.0)
    for _ in range(n_random):
        b_i = float(scale * rng.uniform(0.6, 1.15))
        seeds.append(np.array([
            float(scale * rng.uniform(0.5, 1.15)),        # alpha_i
            float(rng.uniform(c_lo, c_hi)),               # alpha coupling
            b_i,                                          # beta_i
            float(rng.uniform(c_lo * 0.35, c_hi * 0.35)), # beta coupling
            float(meta["t_nr_merger"] - b_i * (meta["t_bhpt_merger"] - T_ANCHOR)),
            float(rng.uniform(-np.pi, np.pi)),            # phi0
        ]))
    return seeds


def optimize_case(q, form="linear", source_stride=3, nr_stride=5,
                  top_n=5, maxiter=9000, analytic_phi0=True):
    case = load_case(q, source_stride, nr_stride)
    n_free = 5 if analytic_phi0 else 6

    def fit_err(free):
        params = np.empty(6)
        params[:n_free] = free
        if analytic_phi0:
            params[5] = 0.0
        return evaluate_model(
            params,
            case["fit_t_bhpt"], case["fit_h_bhpt"],
            case["fit_t_nr"],   case["fit_h_nr"],
            case["fit_losses"], form, analytic_phi0=analytic_phi0,
        )["error"]

    ranked = []
    for s in _seeds(q, case, form):
        e = fit_err(s[:n_free])
        if e < 10.0:
            ranked.append((e, s[:n_free].copy()))
    ranked.sort(key=lambda x: x[0])

    if not ranked:
        print(f"  q={q:.4g}  {form}: NO valid seeds", flush=True)
        return {"q": q, "error": 50.0, "params": [0.0] * 6, "coverage": float("nan")}

    best_err  = ranked[0][0]
    best_free = ranked[0][1].copy()
    for _, start in ranked[:top_n]:
        res = minimize(fit_err, start, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-13})
        if res.fun < best_err:
            best_err  = float(res.fun)
            best_free = res.x.copy()

    best_params = np.zeros(6)
    best_params[:n_free] = best_free
    best_params = canonicalize(best_params, form)

    full = evaluate_model(
        best_params,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"],   case["h_nr"],
        case["losses"], form, analytic_phi0=analytic_phi0,
    )
    if analytic_phi0:
        # reported only; not a fitted or regressed quantity
        best_params[5] = full.get("phi0", 0.0)
    print(f"  q={q:.4g}  {form}: fit={best_err:.4g}  full={full.get('error', 50.):.4g}",
          flush=True)
    return {
        "q":        q,
        "error":    float(full.get("error", 50.0)),
        "params":   [float(v) for v in best_params],
        "coverage": float(full.get("coverage", float("nan"))),
        "m_anchor": float(case["losses"]["m_anchor"]),
        "e_rad":    float(case["losses"]["e_rad"]),
    }


def _optimize_worker(args):
    return optimize_case(*args)


# ---------------------------------------------------------------------------
# nu regression
# ---------------------------------------------------------------------------

def fit_master(rows, degree, form, analytic_phi0=True):
    names   = FORMS[form]
    anchors = PP_ANCHORS[form]
    nu_arr  = np.array([get_nu(r["q"]) for r in rows])
    p_arr   = np.array([r["params"] for r in rows], dtype=float)
    coeffs  = {}
    for i, name in enumerate(names):
        if name == "phi0":
            # solved in closed form at evaluation time; never regressed
            coeffs[name] = np.zeros(degree + 1)
            if not analytic_phi0:
                coeffs[name] = fit_poly_anchored(
                    nu_arr, np.unwrap(p_arr[:, i]), degree, anchors[name])
            continue
        coeffs[name] = fit_poly_anchored(nu_arr, p_arr[:, i].copy(), degree, anchors[name])
    return coeffs


def master_params(q, coeffs, form):
    names  = FORMS[form]
    params = np.array([eval_poly(q, coeffs[name]) for name in names], dtype=float)
    params[0] = float(np.clip(params[0], 0.05, 2.5))   # alpha_i
    params[2] = float(np.clip(params[2], 0.2,  1.6))   # beta_i
    return params


# ---------------------------------------------------------------------------
# Master evaluation
# ---------------------------------------------------------------------------

def polish_nuisance(params, case, form, analytic_phi0=True):
    """
    Re-fit the alignment freedoms only.  With analytic phi0 that is the single
    time offset t0_nr; the phase is already optimal at every evaluation.
    """
    free = [params[4]] if analytic_phi0 else [params[4], params[5]]

    def obj(x):
        trial = params.copy()
        trial[4] = x[0]
        if not analytic_phi0:
            trial[5] = x[1]
        return evaluate_model(
            trial,
            case["fit_t_bhpt"], case["fit_h_bhpt"],
            case["fit_t_nr"],   case["fit_h_nr"],
            case["fit_losses"], form, analytic_phi0=analytic_phi0,
        )["error"]

    res = minimize(obj, free, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-7, "fatol": 1e-9})
    out = params.copy()
    out[4] = float(res.x[0])
    if not analytic_phi0:
        out[5] = float(res.x[1])
    return out


def eval_master_at_q(q, coeffs, form, source_stride, nr_stride, polish=True,
                     analytic_phi0=True):
    params = master_params(q, coeffs, form)
    case   = load_case(q, source_stride, nr_stride)

    raw = evaluate_model(params, case["t_bhpt"], case["h_bhpt"],
                         case["t_nr"], case["h_nr"], case["losses"], form,
                         analytic_phi0=analytic_phi0)
    out = {"q": q, "error_raw": float(raw.get("error", 50.0)),
           "params": [float(v) for v in params]}

    if polish:
        params = polish_nuisance(params, case, form, analytic_phi0)
        ev = evaluate_model(params, case["t_bhpt"], case["h_bhpt"],
                            case["t_nr"], case["h_nr"], case["losses"], form,
                            analytic_phi0=analytic_phi0)
        out["params"] = [float(v) for v in params]
    else:
        ev = raw

    out["error"]    = float(ev.get("error", 50.0))
    out["coverage"] = float(ev.get("coverage", float("nan")))
    return out


def _master_worker(args):
    q, coeffs_by_deg, degree, form, source_stride, nr_stride, polish, aphi = args
    return eval_master_at_q(q, coeffs_by_deg[degree], form,
                            source_stride, nr_stride, polish, aphi)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def summarize(values):
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"min": float("nan"), "median": float("nan"),
                "mean": float("nan"), "max": float("nan")}
    return {"min": float(np.min(a)), "median": float(np.median(a)),
            "mean": float(np.mean(a)), "max": float(np.max(a))}


def select_degree(stats_by_deg, low_by_deg):
    """
    Pick the degree that is best in-range without wrecking q<3 extrapolation.

    Score = in-range max (uniformity is what the master is for) with a hard veto
    on any degree failing the 1e-2 gate at the low-q points that are still inside
    the BHPT surrogate's domain (q >= BHPT_Q_MIN).  q = 2.25 and 2.0 are reported
    but not used as a gate: there the BHPT *input* is extrapolated too, so a
    failure there does not cleanly indict the regression.
    """
    ok = [d for d in stats_by_deg
          if np.isfinite(low_by_deg[d]["max_in_domain"])
          and low_by_deg[d]["max_in_domain"] <= 1e-2]
    pool = ok if ok else list(stats_by_deg)
    return min(pool, key=lambda d: stats_by_deg[d]["max"])


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def q_key(q):
    return f"{q:.10f}"


def get_cached_q_values():
    import re
    pat = re.compile(r"waveforms_q(.+)\.npz")
    qs = []
    for p in WAVEFORM_CACHE_DIR.glob("waveforms_q*.npz"):
        m = pat.match(p.name)
        if m:
            qs.append(float(m.group(1)))
    return sorted(qs)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def coeff_table_lines(coeffs, degree, form):
    header = "| parameter | anchor | " + " | ".join(f"c{i}" for i in range(degree + 1)) + " |"
    sep    = "|:---|:---|" + "|".join("---:" for _ in range(degree + 1)) + "|"
    lines  = [header, sep]
    for name in FORMS[form]:
        anc = PP_ANCHORS[form][name]
        anc_str = f"{anc:.4g}" if anc is not None else "free"
        row = " | ".join(f"{v:.12g}" for v in coeffs[name])
        lines.append(f"| {name} | {anc_str} | {row} |")
    return lines


def write_markdown(form, rows, fit_rows, masters, lows, coeffs_by_deg,
                   selected_degree, args_ns):
    names    = FORMS[form]
    selected = masters[selected_degree]
    sel_low  = lows[selected_degree]

    s_perq = summarize([r["error"] for r in rows
                        if np.isfinite(r.get("coverage", float("nan"))) and r["error"] < 10.0])
    s_fit  = summarize([r["error"] for r in fit_rows])
    s_sel  = summarize([r["error"] for r in selected])
    q_at_max = max(selected, key=lambda r: r["error"])["q"]

    if form == "mult":
        eqs = ["alpha(t, nu) = alpha_PP(nu) * (1 + alpha_E(nu) * E(t))",
               "beta(t, nu)  = beta_PP(nu)  * (1 + beta_E(nu)  * E(t))"]
        pred_note = ("E(t) is the RAW gw_remnant Eoft (zero at the start of the "
                     "surrogate window), so alpha_PP is the scaling there.  Pure "
                     "mass rescaling -- alpha ~ M(t)/M0 = 1 - E(t) -- predicts "
                     "alpha_E = beta_E = -1.")
    elif form == "linear":
        eqs = ["alpha(t) = alpha_i + alpha_E * dE(t)",
               "beta(t)  = beta_i  + beta_E  * dE(t)"]
        pred_note = ("Pure mass rescaling predicts alpha_E/alpha_i = beta_E/beta_i "
                     "= -1/M(T_ANCHOR) ~ -1.006.")
    elif form == "linear_nu":
        eqs = ["nu_hat   = nu(q) / nu(5)",
               "alpha(t) = alpha_i + alpha_A * dE(t) / nu_hat",
               "beta(t)  = beta_i  + beta_B  * dE(t) / nu_hat**2"]
        pred_note = ("Identical model to form=`linear` with alpha_A = alpha_E*nu_hat "
                     "and beta_B = beta_E*nu_hat**2; the nu powers were read off the "
                     "`linear` per-q solutions and make the couplings regress.")
    else:
        eqs = ["alpha(t) = alpha_i * m_hat(t) ** alpha_p",
               "beta(t)  = beta_i  * m_hat(t) ** beta_p"]
        pred_note = "Pure mass rescaling predicts alpha_p = beta_p = 1."

    lines = [
        f"# gw_remnant energy-driven scaling fit (gw_remnant_energy, form=`{form}`)",
        "",
        f"Switchless, {5 if not args_ns.fit_phi0 else 6} free parameters.  The *only* "
        "time-dependence is the mass the binary has radiated away.  No "
        "angular-momentum coordinate, no logistic gate.",
        "",
        ("`phi0` is solved in closed form at every evaluation "
         "(`phi0 = arg(sum(h_NR * conj(h_model)))`) rather than fitted: the "
         "(t0_nr, phi0) alignment valley is flat enough that per-q optima scatter "
         "by O(pi) between adjacent q, which is unregressable in nu."
         if not args_ns.fit_phi0 else
         "`phi0` fitted as a free parameter (diagnostic run)."),
        "",
        "## Loss coordinate (gw_remnant)",
        "",
        "```python",
        "# RemnantMassCalculator(t_bhpt, {(2,2): h22, (2,-2): conj(h22)}, q,",
        "#                       E_initial=0, L_initial=0, M_initial=1)",
        "dE/dt = (1/16*pi) * sum_lm |dh_lm/dt|**2      # calc.E_dot",
        "E(t)  = cumtrapz(dE/dt, t)                    # calc.Eoft, units of M",
        "M(t)  = 1 - E(t)                              # calc.Moft (Bondi mass)",
        "dE(t)    = E(t) - E(T_ANCHOR)",
        "m_hat(t) = M(t) / M(T_ANCHOR)",
        "```",
        "",
        "## Model equations",
        "",
        "```python",
        *eqs,
        "tau(t) = t0_nr + integral_{T_ANCHOR}^{t} beta(t') dt'",
        "h_model(tau(t)) = alpha(t) * exp(i*phi0) * h_BHPT(t)",
        "```",
        "",
        pred_note,
        "",
        "## Regression",
        "",
        "```python",
        "nu(q) = q / (1 + q)**2",
        "# Anchored at nu->0:  " + ", ".join(
            n for n in names if PP_ANCHORS[form][n] is not None),
        "# Free:               " + ", ".join(
            n for n in names if PP_ANCHORS[form][n] is None),
        "```",
        "",
        "The couplings are deliberately UNANCHORED: dE -> 0 and m_hat -> 1 in the",
        "test-particle limit already enforce the PP limit, and E(t) carries its own",
        "nu-scaling (E_rad ~ nu^2), so the couplings should be nearly nu-flat.",
        "",
        f"Training grid: {len(rows)} q values in "
        f"[{min(r['q'] for r in rows):.4g}, {max(r['q'] for r in rows):.4g}]; "
        f"{len(fit_rows)} rows used in the regression.",
        f"Selected polynomial degree: **{selected_degree}**",
        "",
        "## Selected master coefficients",
        "",
        *coeff_table_lines(coeffs_by_deg[selected_degree], selected_degree, form),
        "",
        "## Error summary (in-range [3, 8])",
        "",
        "| model | min | median | mean | max | q at max |",
        "|:---|---:|---:|---:|---:|---:|",
        f"| per-q independent | {s_perq['min']:.6g} | {s_perq['median']:.6g}"
        f" | {s_perq['mean']:.6g} | {s_perq['max']:.6g} | |",
        f"| per-q used in regression | {s_fit['min']:.6g} | {s_fit['median']:.6g}"
        f" | {s_fit['mean']:.6g} | {s_fit['max']:.6g} | |",
    ]
    for deg in sorted(masters):
        s = summarize([r["error"] for r in masters[deg]])
        tag = " **(selected)**" if deg == selected_degree else ""
        lines.append(
            f"| degree-{deg} master{tag} | {s['min']:.6g} | {s['median']:.6g}"
            f" | {s['mean']:.6g} | {s['max']:.6g} |"
            f" {max(masters[deg], key=lambda r: r['error'])['q']:.6g} |")

    lines += [
        "",
        "## q < 3 extrapolation",
        "",
        "`raw` = master polynomial evaluated as-is; `polished` = t0_nr and phi0 "
        "re-fitted (alignment freedoms only, all shape parameters extrapolated).",
        "",
        "| q | nu | BHPT input | "
        + " | ".join(f"deg-{d} raw / polished" for d in sorted(lows)) + " |",
        "|---:|---:|:---|" + "|".join("---:" for _ in lows) + "|",
    ]
    for i, q in enumerate(LOW_Q):
        cells = []
        for d in sorted(lows):
            r = lows[d]["rows"][i]
            cells.append(f"{r['error_raw']:.4g} / {r['error']:.4g}")
        dom = "in domain" if q >= BHPT_Q_MIN - 1e-9 else "**extrapolated**"
        lines.append(f"| {q:.4g} | {get_nu(q):.4f} | {dom} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"`BHPTNRSur1dq1e4` is only declared valid for q >= {BHPT_Q_MIN} "
        "(`X_min = log10(2.5)`), and the bound is warned about rather than "
        "enforced.  At q = 2.25 and 2.0 the BHPT *input* is therefore an "
        "extrapolation of the surrogate's own splines in log q, on top of the "
        "coefficient extrapolation -- so **q = 2.5 is the honest low-q gate** and "
        "the two points below it are reported for continuity only.",
        "",
        f"Selected degree-{selected_degree}: worst in-domain low-q error "
        f"**{sel_low['max_in_domain']:.4g}** at q={sel_low['q_at_max_in_domain']:.4g}; "
        f"worst including out-of-domain {sel_low['max']:.4g} at q={sel_low['q_at_max']:.4g}.",
        "",
        "## Per-q results",
        "",
        "| q | nu | per-q E | master E | coverage | in fit | "
        + " | ".join(names[:4]) + " | ratio_a | ratio_b |",
        "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    sel_by_q = {q_key(r["q"]): r for r in selected}
    fit_keys = {q_key(r["q"]) for r in fit_rows}
    for row in rows:
        key = q_key(row["q"])
        p   = row["params"]
        mr  = sel_by_q.get(key, {"error": float("nan"), "coverage": float("nan")})
        if form == "mult":
            # alpha_E*nu is the flat diagnostic (~ -1.18); beta_E should sit near
            # -1 if the clock really is a mass rescaling
            ra, rb = p[1] * get_nu(row["q"]), p[3]
        elif form == "mass_power":
            ra, rb = p[1], p[3]
        else:
            ra = p[1] / p[0] if p[0] else float("nan")
            rb = p[3] / p[2] if p[2] else float("nan")
        lines.append(
            f"| {row['q']:.6g} | {get_nu(row['q']):.4f} | {row['error']:.4g}"
            f" | {mr['error']:.4g} | {mr['coverage']:.4f}"
            f" | {'yes' if key in fit_keys else ''}"
            f" | {p[0]:.5g} | {p[1]:.5g} | {p[2]:.5g} | {p[3]:.5g}"
            f" | {ra:.4g} | {rb:.4g} |"
        )
    lines += [
        "",
        ("`ratio_a` = alpha_E*nu (flat to a few % if the coupling really goes as "
         "1/nu), `ratio_b` = beta_E itself (compare against -1 for a pure mass "
         "rescaling of the clock)." if form == "mult" else
         "`ratio_a`, `ratio_b` are alpha_E/alpha_i and beta_E/beta_i (linear form) "
         "or the exponents themselves (mass_power form) — compare against the pure "
         "mass-rescaling prediction quoted above."),
        "",
        f"Command: `python {Path(__file__).name} --form {form} "
        f"--q-min {args_ns.q_min} --q-max {args_ns.q_max}`",
        "",
    ]
    md_path = MD_PATH if form == "linear" else MD_PATH.with_name(
        f"scaling_gw_remnant_energy_{form}.md")
    md_path.write_text("\n".join(lines))
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--form", choices=sorted(FORMS), default="linear")
    parser.add_argument("--q-min",              type=float, default=3.0)
    parser.add_argument("--q-max",              type=float, default=8.0)
    parser.add_argument("--source-stride",      type=int,   default=3)
    parser.add_argument("--nr-stride",          type=int,   default=5)
    parser.add_argument("--top-n",              type=int,   default=5)
    parser.add_argument("--maxiter",            type=int,   default=9000)
    parser.add_argument("--master-outlier-cut", type=float, default=MASTER_OUTLIER_CUT)
    parser.add_argument("--workers",            type=int,   default=4)
    parser.add_argument("--force",              action="store_true")
    parser.add_argument("--fit-phi0", action="store_true",
                        help="fit phi0 as a free parameter instead of solving "
                             "for it analytically (diagnostic only)")
    args = parser.parse_args()

    form = args.form
    aphi = not args.fit_phi0
    cache_path, coeff_path = results_paths(form)
    if args.fit_phi0:
        cache_path  = cache_path.with_name(cache_path.stem + "_fitphi0.json")
        coeff_path  = coeff_path.with_name(coeff_path.stem + "_fitphi0.json")
    print(f"[{form}] phi0: {'analytic' if aphi else 'fitted'} "
          f"({5 if aphi else 6} free parameters)", flush=True)

    all_q = [q for q in get_cached_q_values()
             if args.q_min - 1e-9 <= q <= args.q_max + 1e-9]
    if not all_q:
        raise RuntimeError("No waveforms found in .cache/q_dep/.")
    print(f"[{form}] training grid: {len(all_q)} q values in "
          f"[{all_q[0]:.4g}, {all_q[-1]:.4g}]", flush=True)

    # Phase 1 — per-q optimisation
    cache = {} if args.force or not cache_path.exists() else json.loads(cache_path.read_text())
    todo  = [q for q in all_q if q_key(q) not in cache]
    if todo:
        print(f"Optimising {len(todo)} q values ({args.workers} workers) ...", flush=True)
        with Pool(args.workers) as pool:
            results = pool.map(_optimize_worker, [
                (q, form, args.source_stride, args.nr_stride, args.top_n,
                 args.maxiter, aphi)
                for q in todo])
        for row in results:
            cache[q_key(row["q"])] = row
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    else:
        print("All per-q results already cached.", flush=True)

    rows = sorted([v for v in cache.values()
                   if all_q[0] - 1e-9 <= v["q"] <= all_q[-1] + 1e-9],
                  key=lambda r: r["q"])
    # applied on load as well as at fit time, so caches written before the
    # gauge fix are repaired without re-optimising
    for r in rows:
        r["params"] = [float(v) for v in canonicalize(r["params"], form)]

    # Phase 2 — nu regression
    fit_rows = [r for r in rows
                if np.isfinite(r.get("coverage", float("nan")))
                and r["error"] < args.master_outlier_cut]
    if len(fit_rows) < 8:
        raise RuntimeError(f"Too few valid rows ({len(fit_rows)}) for regression.")
    print(f"Fitting nu polynomials on {len(fit_rows)} rows ...", flush=True)
    coeffs_by_deg = {d: fit_master(fit_rows, d, form, aphi) for d in DEGREES}

    # Phase 3 — master evaluation in range and below it
    print("Evaluating master model (in-range + q<3) ...", flush=True)
    jobs = [(r["q"], coeffs_by_deg, d, form, args.source_stride, args.nr_stride,
             True, aphi) for d in DEGREES for r in rows]
    jobs += [(q, coeffs_by_deg, d, form, args.source_stride, args.nr_stride,
              True, aphi) for d in DEGREES for q in LOW_Q]
    with Pool(args.workers) as pool:
        out = pool.map(_master_worker, jobs)

    n_in = len(rows)
    masters, lows = {}, {}
    for i, d in enumerate(DEGREES):
        masters[d] = out[i * n_in:(i + 1) * n_in]
    base = len(DEGREES) * n_in
    for i, d in enumerate(DEGREES):
        lrows = out[base + i * len(LOW_Q): base + (i + 1) * len(LOW_Q)]
        worst = max(lrows, key=lambda r: r["error"])
        in_dom = [r for r in lrows if r["q"] >= BHPT_Q_MIN - 1e-9]
        worst_in = max(in_dom, key=lambda r: r["error"])
        lows[d] = {"rows": lrows, "max": worst["error"], "q_at_max": worst["q"],
                   "max_in_domain": worst_in["error"],
                   "q_at_max_in_domain": worst_in["q"]}

    for d in DEGREES:
        s = summarize([r["error"] for r in masters[d]])
        by_q = {f"{r['q']:.4g}": r["error"] for r in lows[d]["rows"]}
        print(f"  degree {d}: in-range median={s['median']:.6g} max={s['max']:.6g}"
              f" | q2.5={by_q.get('2.5', float('nan')):.6g}"
              f" q2.25={by_q.get('2.25', float('nan')):.6g}"
              f" q2={by_q.get('2', float('nan')):.6g}", flush=True)

    selected_degree = select_degree(
        {d: summarize([r["error"] for r in masters[d]]) for d in DEGREES}, lows)
    print(f"selected_degree={selected_degree}", flush=True)

    coeff_path.write_text(json.dumps({
        "form": form,
        "analytic_phi0": aphi,
        "n_free_params": 5 if aphi else 6,
        "param_names": FORMS[form],
        "pp_anchors": PP_ANCHORS[form],
        "selected_degree": selected_degree,
        "coeffs": {str(d): {k: [float(x) for x in v] for k, v in c.items()}
                   for d, c in coeffs_by_deg.items()},
        "in_range": {str(d): summarize([r["error"] for r in masters[d]]) for d in DEGREES},
        "low_q": {str(d): {f"{r['q']:.4g}": {"raw": r["error_raw"], "polished": r["error"]}
                           for r in lows[d]["rows"]} for d in DEGREES},
        "training_range": [args.q_min, args.q_max],
        "n_training_q": len(rows),
    }, indent=2))

    md_path = write_markdown(form, rows, fit_rows, masters, lows,
                             coeffs_by_deg, selected_degree, args)
    print(f"Wrote {md_path}\nWrote {coeff_path}", flush=True)


if __name__ == "__main__":
    main()
