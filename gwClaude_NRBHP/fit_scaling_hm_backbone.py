"""
Higher-mode extension of gwr_energy_stiff via the (2,2) backbone transfer.

STEP 1 (this script): the ZERO-PARAMETER diagnostic.

Section 6.3 of `alpha_beta_pn_scaling_note_revised.pdf` gives a construction that
transfers the calibrated quadrupole onto every higher mode using only leading-PN
inter-mode mass factors, with no new fitted freedom (note Eq. 43):

    alpha_backbone_lm(t, q) = alpha_22(t, q) * C_lm(nu) * beta(t, q)^(-(l + eps_lm - 2)/3)
                              * rho_lm(t, q)

    eps_lm = (l + m) mod 2
    C_lm   = | X2^(l + eps_lm - 1) + (-1)^m X1^(l + eps_lm - 1) |      (note Eq. 39)
    rho_lm -> 1 in the point-particle limit                            (the residual)

with X1 = q/(1+q), X2 = 1/(1+q).  The beta^(-.../3) factor comes from x_NR/x_pp =
beta^(-2/3) (note Eq. 42), i.e. it is the frequency rescaling the common time map
already implies -- not a new parameter.

Verified against the note's Eq. (40) for the diagonal modes:
    C_33 = Delta,   C_44 = 1 - 3 nu,   C_55 = Delta (1 - 2 nu),   C_22 = 1
(Delta = X1 - X2 = (q-1)/(q+1)).  Because C_22 = 1 and its exponent is 0, the
backbone reproduces the (2,2) model EXACTLY -- a built-in consistency check against
the numbers in scaling_gwr_energy_stiff.md.

What is shared and what is not
------------------------------
beta(t) and the time map tau(t) = t0_nr + int beta dt' are COMMON to all modes: that
is the structure of the original BHPTNRSur1dq1e4 calibration (note Eq. 1, a single
beta(q) with mode-dependent alpha_l(q)) and of the reference implementation in
BHPTutils (`AlphaBetaOptimizer._optimize_func`: one beta scales the time axis, one
alpha per mode scales the strain).  So this extension adds NO new time-map freedom
and cannot degrade the (2,2) result.  Per mode we allow only the constant phase, and
that is solved analytically (arg of the complex overlap), exactly as for (2,2).

Two numbers are reported per mode:
  * rho = 1        -- the pure prediction, no fitting whatsoever;
  * rho = const    -- the best single multiplicative constant per mode, ALSO in closed
                      form (k = |z| / n2 minimises the mismatch), so still no
                      optimiser.  The fitted k measures how much of the residual is a
                      constant amplitude offset, i.e. how much a 1-parameter-per-mode
                      residual would buy.  The note's Eq. (49)-(51) comparison
                      predicts a 5-10% shortfall in the leading coefficient, so k is
                      expected around 0.9-1.1.

Modes: the BHPT/NR intersection.  BHPTNRSur1dq1e4 provides (2,2),(2,1),(3,1),(3,2),
(3,3),(4,2),(4,3),(4,4),(5,3),(5,4),(5,5),(6,4)...(10,9); NRHybSur3dq8 provides
(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,5).  Intersection = 9
modes.  The note warns that off-diagonal modes (spherical-spheroidal mixing) should
be tested separately rather than assumed to follow the l=m hierarchy, so they are
reported in a separate block.

Usage:  python fit_scaling_hm_backbone.py [--q 3 5 8 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_gw_remnant_energy as gwre
import fit_scaling_gwr_energy_stiff as stiff

FORM = "mult"
RESULTS = ROOT / "hm_backbone_results"
RESULTS.mkdir(exist_ok=True)
CACHE_DIR = ROOT / ".cache" / "hm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODES_DIAG = [(2, 2), (3, 3), (4, 4), (5, 5)]
MODES_OFFDIAG = [(2, 1), (3, 1), (3, 2), (4, 2), (4, 3)]
MODES = MODES_DIAG + MODES_OFFDIAG


# ---------------------------------------------------------------------------
# mass factors from the note (Eqs. 39, 40, 43)
# ---------------------------------------------------------------------------

def eps_lm(l, m):
    return (l + m) % 2


def C_lm(l, m, q):
    """|X2^(l+eps-1) + (-1)^m X1^(l+eps-1)|  -- note Eq. (39)."""
    X1 = q / (1.0 + q)
    X2 = 1.0 / (1.0 + q)
    k = l + eps_lm(l, m) - 1
    return abs(X2 ** k + (-1.0) ** m * X1 ** k)


def beta_exponent(l, m):
    """-(l + eps_lm - 2)/3  -- the exponent of beta in note Eq. (43)."""
    return -(l + eps_lm(l, m) - 2) / 3.0


def check_note_eq40():
    """C_lm must reproduce the note's closed forms for the diagonal modes."""
    out = []
    for q in (2.0, 3.0, 5.0, 8.0):
        nu = q / (1.0 + q) ** 2
        D = (q - 1.0) / (q + 1.0)
        want = {(2, 2): 1.0, (3, 3): D, (4, 4): 1.0 - 3 * nu, (5, 5): D * (1.0 - 2 * nu)}
        for mode, w in want.items():
            got = C_lm(*mode, q)
            out.append((q, mode, got, w, abs(got - w)))
    return out


# ---------------------------------------------------------------------------
# multi-mode waveform cache
# ---------------------------------------------------------------------------

def hm_cache_path(q):
    return CACHE_DIR / f"waveforms_hm_q{q:.10f}.npz"


def generate_and_cache_hm(q):
    path = hm_cache_path(q)
    if path.exists():
        return path
    import gwsurrogate
    import BHPTNRSur1dq1e4 as bhptsur

    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    t_bhpt, h_bhpt = bhptsur.generate_surrogate(
        q=q, calibrated=False, modes=MODES, neg_modes=False)
    t_nr, h_nr, _ = nrsur(q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)

    nr_mask = (t_nr >= qdep.NR_T_START) & (t_nr <= qdep.NR_T_END)
    t_nr = t_nr[nr_mask]

    store = {"t_bhpt": t_bhpt, "t_nr": t_nr, "q": np.float64(q),
             "nu": np.float64(q / (1.0 + q) ** 2)}
    for (l, m) in MODES:
        store[f"bhpt_{l}{m}_re"] = h_bhpt[(l, m)].real
        store[f"bhpt_{l}{m}_im"] = h_bhpt[(l, m)].imag
        store[f"nr_{l}{m}_re"] = h_nr[(l, m)][nr_mask].real
        store[f"nr_{l}{m}_im"] = h_nr[(l, m)][nr_mask].imag
    np.savez(path, **store)
    print(f"cached {len(MODES)}-mode waveforms q={q:.6g}", flush=True)
    return path


def load_hm(q):
    d = np.load(generate_and_cache_hm(q))
    bhpt = {(l, m): d[f"bhpt_{l}{m}_re"] + 1j * d[f"bhpt_{l}{m}_im"] for l, m in MODES}
    nr = {(l, m): d[f"nr_{l}{m}_re"] + 1j * d[f"nr_{l}{m}_im"] for l, m in MODES}
    return {"t_bhpt": d["t_bhpt"], "t_nr": d["t_nr"], "h_bhpt": bhpt, "h_nr": nr,
            "nu": float(d["nu"])}


# ---------------------------------------------------------------------------
# the (2,2) model: alpha_22(t), beta(t), tau(t) from gwr_energy_stiff
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "E_deg3":        "energy, uniform deg-3 (14c)",
    "flux_deg3":     "energy-flux, uniform deg-3 (14c)",
    "E_anchored":    "energy, PN-anchored (6c)",
    "flux_anchored": "energy-flux, PN-anchored (7c)",
    "stiff":         "energy, stiffened (9c)",
    "anchored":      "energy, PN-anchored (6c)",
}


def read_stiff_coeffs(model="stiff"):
    """
    Quadrupole coefficients.  `model` selects which (2,2) base carries the modes.

    Returns a dict with `params_at` (q -> the 4-vector alpha_PP, a_c, beta_PP, b_c) and
    `coord`, the coordinate alpha couples to -- `e_oft` for the energy models,
    `flux_hat` = Edot/max(Edot) for the flux models.  The backbone itself is unchanged
    and still fits nothing; only the quadrupole it transfers differs.
    """
    if model in ("E_deg3",):
        import fit_scaling_gwr_energy_global as GG
        d = json.loads((ROOT / "gwr_energy_global_results" / "coeffs.json").read_text())
        cf = {k: np.asarray(v, float) for k, v in d.items() if not k.startswith("_")}
        return {"name": model, "coord": "e_oft",
                "params_at": lambda q: GG.params_at(q, cf)}
    if model in ("flux_deg3",):
        import fit_scaling_gwr_energy_flux as FL
        cf = {k: np.asarray(v, float)
              for k, v in json.loads(FL.COEFFS.read_text())["coeffs"].items()}
        return {"name": model, "coord": "flux_hat",
                "params_at": lambda q: FL.params_at(q, cf)}
    if model in ("flux_anchored",):
        import fit_scaling_gwr_energy_fluxanchored as FA
        d = json.loads(FA.COEFFS.read_text())
        th = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                             [float(d["b"])], np.asarray(d["P"], float)])
        deg = int(d["aF_degree"])
        return {"name": model, "coord": "flux_hat",
                "params_at": lambda q: FA.params_at(q, th, deg)}
    if model in ("anchored", "E_anchored"):
        import fit_scaling_gwr_energy_anchored as anch
        d = json.loads(anch.COEFFS.read_text())
        th = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                             [float(d["b"])], np.asarray(d["P"], float)])
        deg = int(d["alphaE_degree"])
        return {"name": model, "coord": "e_oft",
                "params_at": lambda q: anch.params_at(q, th, deg)}
    if model in ("beta_monotone", "beta_monotone_E"):
        # P(nu) = |a|*max(0, nu-nu_c) >= 0, so beta never decreases at any q.
        # Two alpha bases: Edot (parent fluxanchored, 7c) and E (parent anchored, 6c).
        import fit_scaling_gwr_beta_monotone as BM
        Ebase = model.endswith("_E")
        d = json.loads((BM.RESULTS / ("coeffs_Ebase.json" if Ebase
                                      else "coeffs.json")).read_text())
        th = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                             [float(d["b"]), float(d["P_a"]), float(d["P_nu_c"])]])
        na = int(d.get("alpha_deg", d.get("aF_degree")))
        adeg = na - 1 if Ebase else na
        bs = "E" if Ebase else "flux"
        return {"name": model, "coord": "e_oft" if Ebase else "flux_hat",
                "params_at": lambda q: BM.params_at(q, th, adeg, bs)}
    d = json.loads(stiff.COEFFS.read_text())
    cf = {k: np.asarray(v, float) for k, v in d.items() if not k.startswith("_")}
    return {"name": "stiff", "coord": "e_oft",
            "params_at": lambda q: stiff.params_at(q, cf)}


def _evaluate(p, case, coord, t0):
    """gwre.evaluate_model generalised over alpha's coordinate (e_oft or flux_hat)."""
    a_pp, a_c, b_pp, b_c = p[:4]
    los = case["losses"]
    alpha = a_pp * (1.0 + a_c * los[coord])
    beta = b_pp * (1.0 + b_c * los["e_oft"])
    if np.any(beta <= 0) or not np.all(np.isfinite(np.r_[alpha, beta])):
        return {"error": 50.0}
    bc = creative.cumulative_trapezoid(beta, case["t_bhpt"])
    tau = t0 + bc - float(np.interp(gwre.T_ANCHOR, case["t_bhpt"], bc))
    if not np.all(np.diff(tau) > 0):
        return {"error": 50.0}
    t_nr, h_nr = case["t_nr"], case["h_nr"]
    m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
    tc, href = t_nr[m], h_nr[m]
    h0 = np.interp(tc, tau, alpha) * (np.interp(tc, tau, case["h_bhpt"].real)
                                      + 1j * np.interp(tc, tau, case["h_bhpt"].imag))
    z = np.sum(href * h0.conjugate())
    n1 = np.sum(np.abs(href) ** 2); n2 = np.sum(np.abs(h0) ** 2)
    return {"error": float((n1 + n2 - 2.0 * abs(z)) / (2.0 * n1)),
            "tau": tau, "alpha": alpha, "beta": beta, "common": m}


def quadrupole_model(q, coeffs, perq=False):
    """
    alpha_22(t), beta(t) on the BHPT grid, plus the polished shared time map.

    perq=False (default): the (2,2) parameters come from the SHIPPED 9 master
      coefficients, fitted on [3,8] only.  At q < 3 that is an extrapolation, which is
      what the reported numbers are.
    perq=True: the (2,2) parameters are re-optimised at this q alone (the `per-q floor`
      of scaling_gwr_energy_stiff.md).  Used only to separate error INHERITED from the
      (2,2) coefficient extrapolation from error intrinsic to the mode transfer.
      Not a shippable model -- it uses NR data at that q.
    """
    import fit_scaling_gwr_energy_flux as FL
    from scipy.optimize import minimize_scalar
    qdep.generate_and_cache_waveform(q)
    case = FL.add_flux(gwre.load_case(q, source_stride=3, nr_stride=5))
    coord = coeffs["coord"]
    if perq:
        if coord != "e_oft":
            raise ValueError("--perq is only defined for the E-coupled per-q form")
        params = gwre.optimize_case(q, FORM, top_n=5, maxiter=9000)["params"][:4]
    else:
        params = coeffs["params_at"](q)[:4]
    f = lambda t0: _evaluate(params, case, coord, t0)["error"]
    t0 = float(minimize_scalar(f, bounds=(-200.0, 40.0), method="bounded",
                               options={"xatol": 1e-3}).x)
    ev = _evaluate(params, case, coord, t0)
    return {"params": np.r_[params, t0, 0.0], "case": case, "alpha22": ev["alpha"],
            "beta": ev["beta"], "tau": ev["tau"], "error22": ev["error"]}


# ---------------------------------------------------------------------------
# per-mode mismatch under the shared time map
# ---------------------------------------------------------------------------

def mode_mismatch(t_common_src, tau, alpha_lm, h_bhpt_lm, t_nr, h_nr_lm,
                  return_curves=False):
    """
    Mismatch of alpha_lm * h_bhpt_lm(tau) against NR, with the constant phase
    analytic.  Returns both rho=1 and best-constant-rho errors; both closed form.

    error(k) = (n1 + k^2 n2 - 2 k |z|) / (2 n1),  z = sum(h_ref conj(h0))
      k = 1            -> the pure prediction
      k = |z| / n2     -> the minimiser, giving error = (n1 - |z|^2/n2) / (2 n1)
    """
    t_min = max(t_nr[0], tau[0])
    t_max = min(t_nr[-1], tau[-1])
    nr_mask = (t_nr >= t_min) & (t_nr <= t_max)
    t_common = t_nr[nr_mask]
    h_ref = h_nr_lm[nr_mask]

    h_r = np.interp(t_common, tau, h_bhpt_lm.real)
    h_i = np.interp(t_common, tau, h_bhpt_lm.imag)
    a_c = np.interp(t_common, tau, alpha_lm)
    h0 = a_c * (h_r + 1j * h_i)

    z = np.sum(h_ref * h0.conjugate())
    n1 = np.sum(np.abs(h_ref) ** 2)
    n2 = np.sum(np.abs(h0) ** 2)
    if n1 <= 0 or n2 <= 0 or not np.isfinite(n1 + n2):
        return {"err_rho1": np.nan, "err_rhoc": np.nan, "k": np.nan,
                "coverage": float(nr_mask.sum() / len(t_nr)), "power": 0.0}

    az = float(np.abs(z))
    out = {
        "err_rho1": float((n1 + n2 - 2.0 * az) / (2.0 * n1)),
        "err_rhoc": float((n1 - az ** 2 / n2) / (2.0 * n1)),
        "k": float(az / n2),
        "coverage": float(nr_mask.sum() / len(t_nr)),
        "power": float(n1),
    }
    if return_curves:
        # the rho=1 model with its analytic constant phase, for plotting only;
        # the numbers above are unaffected.
        out["t_common"] = t_common
        out["h_ref"] = h_ref
        out["h_model"] = h0 * np.exp(1j * float(np.angle(z)))
    return out


def run_q(q, coeffs, verbose=True, perq=False):
    qm = quadrupole_model(q, coeffs, perq=perq)
    data = load_hm(q)
    tau, beta, alpha22 = qm["tau"], qm["beta"], qm["alpha22"]

    # alpha_22/beta live on the (2,2) case grid; the hm cache uses the raw BHPT grid.
    t_src = qm["case"]["t_bhpt"]
    t_hm = data["t_bhpt"]
    tau_h = np.interp(t_hm, t_src, tau)
    beta_h = np.interp(t_hm, t_src, beta)
    a22_h = np.interp(t_hm, t_src, alpha22)

    rows = {}
    tot_power = sum(np.sum(np.abs(data["h_nr"][mo]) ** 2) for mo in MODES)
    for mo in MODES:
        l, m = mo
        a_lm = a22_h * C_lm(l, m, q) * beta_h ** beta_exponent(l, m)
        r = mode_mismatch(t_hm, tau_h, a_lm, data["h_bhpt"][mo],
                          data["t_nr"], data["h_nr"][mo])
        r["C_lm"] = C_lm(l, m, q)
        r["beta_exp"] = beta_exponent(l, m)
        r["power_frac"] = r["power"] / tot_power if tot_power > 0 else 0.0
        rows[mo] = r

    if verbose:
        print(f"\n=== q = {q:g}   nu = {data['nu']:.4f}   "
              f"(2,2) reference mathcalE = {qm['error22']:.4e} ===")
        print(f"{'mode':>7} {'C_lm':>8} {'beta_exp':>9} {'power':>9} "
              f"{'E(rho=1)':>11} {'E(rho=c)':>11} {'k':>8}")
        for label, group in (("diagonal", MODES_DIAG), ("off-diagonal", MODES_OFFDIAG)):
            print(f"  -- {label} --")
            for mo in group:
                r = rows[mo]
                print(f"{str(mo):>7} {r['C_lm']:8.4f} {r['beta_exp']:9.3f} "
                      f"{r['power_frac']:9.2e} {r['err_rho1']:11.4e} "
                      f"{r['err_rhoc']:11.4e} {r['k']:8.4f}")
    return {"q": q, "nu": data["nu"], "error22": qm["error22"], "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, nargs="+", default=[3.0, 5.0, 8.0, 2.0])
    ap.add_argument("--model", default="stiff",
                    choices=("stiff", "anchored", "E_deg3", "flux_deg3",
                             "E_anchored", "flux_anchored"),
                    help="which (2,2) base carries the higher modes")
    ap.add_argument("--perq", action="store_true",
                    help="use per-q-optimised (2,2) params instead of the master "
                         "coefficients (diagnostic only: uses NR at that q)")
    args = ap.parse_args()

    bad = [r for r in check_note_eq40() if r[4] > 1e-12]
    print(f"C_lm vs note Eq. (40) closed forms: "
          f"{'MISMATCH ' + str(bad) if bad else 'exact for all diagonal modes'}")

    coeffs = read_stiff_coeffs(args.model)
    out = [run_q(q, coeffs, perq=args.perq) for q in args.q]

    payload = {"note": "zero-parameter (2,2)-backbone transfer, note Eq. 43",
               "modes_diagonal": [list(m) for m in MODES_DIAG],
               "modes_offdiagonal": [list(m) for m in MODES_OFFDIAG],
               "results": [{"q": r["q"], "nu": r["nu"], "error22": r["error22"],
                            "rows": {f"{l}{m}": v for (l, m), v in r["rows"].items()}}
                           for r in out]}
    payload["quadrupole_source"] = "per-q optimum" if args.perq else "master coefficients"
    sfx = "" if args.model == "stiff" else f"_{args.model}"
    name = (f"backbone_rho1{sfx}_perq.json" if args.perq
            else f"backbone_rho1{sfx}.json")
    (RESULTS / name).write_text(json.dumps(payload, indent=2))
    print(f"\nWritten: {RESULTS / name}")


if __name__ == "__main__":
    main()
