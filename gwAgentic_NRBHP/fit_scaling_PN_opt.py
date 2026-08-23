from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import minimize

import fit_scaling_PN as base


Q = base.Q
MODE = base.MODE
T_REF = base.T_REF
TARGET = 1.0e-3
DESIRED = 1.0e-4

# Parameter order:
# p_start, p_width, t_start_nr,
# beta_left_const, beta_right_edge, beta_right_e, beta_right_j,
# alpha_left_const, alpha_right_edge, alpha_right_e, alpha_right_j,
# phi0
#
# The default accepted parameters are overwritten by running this script with
# --refit when a better smooth-transition solution is found.
ACCEPTED_PN_OPT_PARAMS = np.array(
    [
        -9.580164641299050e-01,
        5.714341410385353e-02,
        -6.745674687459955e01,
        8.044622792936059e-01,
        8.260432840217982e-01,
        1.768939802390595e-03,
        1.681229941065323e-03,
        8.098825628023710e-01,
        8.010433347009299e-01,
        2.536494098668636e-01,
        -3.793024530185873e-01,
        1.313892422375683e00,
    ],
    dtype=float,
)

OLD_DISCONTINUOUS_PARAMS = np.array(
    [
        -9.314282404272071e-01,
        -6.053514496114817e01,
        8.044673521530664e-01,
        8.257969772447457e-01,
        1.164640369011219e-03,
        2.627172998911345e-03,
        8.098917475324235e-01,
        8.089500323855249e-01,
        2.521632187139067e-01,
        -3.770847459785052e-01,
        1.315302905461986e00,
    ],
    dtype=float,
)

PREVIOUS_PN_SEED = np.array(
    [
        -1.049129116009673e00,
        -9.971550514754898e01,
        8.030470800477922e-01,
        8.128584306270605e-01,
        -3.301182718867643e-02,
        5.523103049275632e-02,
        8.140831180521366e-01,
        8.315690350856837e-01,
        2.373765252714617e-01,
        -3.460310858141835e-01,
        1.372921745805103e00,
    ],
    dtype=float,
)


def unpack(params: np.ndarray) -> dict[str, float]:
    (
        p_start,
        p_width,
        t_start_nr,
        beta_left,
        beta_right_edge,
        beta_right_e,
        beta_right_j,
        alpha_left,
        alpha_right_edge,
        alpha_right_e,
        alpha_right_j,
        phi0,
    ) = [float(v) for v in params]
    return {
        "p_start": p_start,
        "p_width": p_width,
        "p_end": p_start + p_width,
        "t_start_nr": t_start_nr,
        "beta_left": beta_left,
        "beta_right_edge": beta_right_edge,
        "beta_right_e": beta_right_e,
        "beta_right_j": beta_right_j,
        "alpha_left": alpha_left,
        "alpha_right_edge": alpha_right_edge,
        "alpha_right_e": alpha_right_e,
        "alpha_right_j": alpha_right_j,
        "phi0": phi0,
    }


def old_unpacked(params: np.ndarray) -> dict[str, float]:
    (
        p_cut,
        t_cut_nr,
        beta_left,
        beta_right_cut,
        beta_right_e,
        beta_right_j,
        alpha_left,
        alpha_right_cut,
        alpha_right_e,
        alpha_right_j,
        phi0,
    ) = [float(v) for v in params]
    return {
        "p_cut": p_cut,
        "t_cut_nr": t_cut_nr,
        "beta_left": beta_left,
        "beta_right_cut": beta_right_cut,
        "beta_right_e": beta_right_e,
        "beta_right_j": beta_right_j,
        "alpha_left": alpha_left,
        "alpha_right_cut": alpha_right_cut,
        "alpha_right_e": alpha_right_e,
        "alpha_right_j": alpha_right_j,
        "phi0": phi0,
    }


def feature_derivatives(losses: dict) -> tuple[np.ndarray, np.ndarray]:
    p_loss = losses["p_loss"]
    d_e_dp = np.gradient(losses["e_hat"], p_loss, edge_order=2)
    d_j_dp = np.gradient(losses["j_hat"], p_loss, edge_order=2)
    return d_e_dp, d_j_dp


def hermite_transition(
    z: np.ndarray,
    width: float,
    y_left: float,
    y_right: float,
    right_slope: float,
) -> np.ndarray:
    h00 = 2.0 * z**3 - 3.0 * z**2 + 1.0
    h01 = -2.0 * z**3 + 3.0 * z**2
    h11 = z**3 - z**2
    return h00 * y_left + h01 * y_right + h11 * width * right_slope


def old_model_arrays(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    losses: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = old_unpacked(params)
    p_loss = losses["p_loss"]
    e_hat = losses["e_hat"]
    j_hat = losses["j_hat"]
    e_cut = float(np.interp(p["p_cut"], p_loss, e_hat))
    j_cut = float(np.interp(p["p_cut"], p_loss, j_hat))
    de = e_hat - e_cut
    dj = j_hat - j_cut
    left = p_loss <= p["p_cut"]

    beta = np.empty_like(t_bhpt, dtype=float)
    alpha = np.empty_like(t_bhpt, dtype=float)
    beta[left] = p["beta_left"]
    alpha[left] = p["alpha_left"]
    beta[~left] = p["beta_right_cut"] + p["beta_right_e"] * de[~left] + p["beta_right_j"] * dj[~left]
    alpha[~left] = p["alpha_right_cut"] + p["alpha_right_e"] * de[~left] + p["alpha_right_j"] * dj[~left]

    beta_integral = base.cumulative_trapezoid(beta, t_bhpt)
    beta_integral_cut = float(np.interp(p["p_cut"], p_loss, beta_integral))
    tau = p["t_cut_nr"] + beta_integral - beta_integral_cut
    h_scaled = alpha * np.exp(1j * p["phi0"]) * h_bhpt
    return tau, h_scaled, alpha, beta, left


def model_arrays(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    losses: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    p = unpack(params)
    p_loss = losses["p_loss"]
    e_hat = losses["e_hat"]
    j_hat = losses["j_hat"]
    d_e_dp, d_j_dp = feature_derivatives(losses)

    e_end = float(np.interp(p["p_end"], p_loss, e_hat))
    j_end = float(np.interp(p["p_end"], p_loss, j_hat))
    de = e_hat - e_end
    dj = j_hat - j_end

    pre = p_loss <= p["p_start"]
    post = p_loss >= p["p_end"]
    transition = ~(pre | post)

    beta = np.empty_like(t_bhpt, dtype=float)
    alpha = np.empty_like(t_bhpt, dtype=float)
    beta[pre] = p["beta_left"]
    alpha[pre] = p["alpha_left"]
    beta[post] = (
        p["beta_right_edge"]
        + p["beta_right_e"] * de[post]
        + p["beta_right_j"] * dj[post]
    )
    alpha[post] = (
        p["alpha_right_edge"]
        + p["alpha_right_e"] * de[post]
        + p["alpha_right_j"] * dj[post]
    )

    if np.any(transition):
        z = (p_loss[transition] - p["p_start"]) / p["p_width"]
        beta_slope_end = (
            p["beta_right_e"] * float(np.interp(p["p_end"], p_loss, d_e_dp))
            + p["beta_right_j"] * float(np.interp(p["p_end"], p_loss, d_j_dp))
        )
        alpha_slope_end = (
            p["alpha_right_e"] * float(np.interp(p["p_end"], p_loss, d_e_dp))
            + p["alpha_right_j"] * float(np.interp(p["p_end"], p_loss, d_j_dp))
        )
        beta[transition] = hermite_transition(
            z,
            p["p_width"],
            p["beta_left"],
            p["beta_right_edge"],
            beta_slope_end,
        )
        alpha[transition] = hermite_transition(
            z,
            p["p_width"],
            p["alpha_left"],
            p["alpha_right_edge"],
            alpha_slope_end,
        )

    beta_integral = base.cumulative_trapezoid(beta, t_bhpt)
    beta_integral_start = float(np.interp(p["p_start"], p_loss, beta_integral))
    tau = p["t_start_nr"] + beta_integral - beta_integral_start
    h_scaled = alpha * np.exp(1j * p["phi0"]) * h_bhpt
    masks = {"pre": pre, "transition": transition, "post": post}
    return tau, h_scaled, alpha, beta, masks


def validate_basic(params: np.ndarray) -> bool:
    p = unpack(params)
    if not (-3.0 <= p["p_start"] <= 0.05):
        return False
    if not (0.015 <= p["p_width"] <= 0.45):
        return False
    if not (-3.0 <= p["p_end"] <= 0.20):
        return False
    if not (-800.0 <= p["t_start_nr"] <= 30.0):
        return False
    for key in ("beta_left", "beta_right_edge"):
        if not (0.2 <= p[key] <= 1.6):
            return False
    for key in ("alpha_left", "alpha_right_edge"):
        if not (0.05 <= p[key] <= 2.5):
            return False
    slope_keys = [
        "beta_right_e",
        "beta_right_j",
        "alpha_right_e",
        "alpha_right_j",
    ]
    return all(-1.2 <= p[key] <= 1.2 for key in slope_keys)


def evaluate_model(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    losses: dict,
    min_coverage: float = 0.94,
) -> dict:
    if not validate_basic(params):
        return {"error": 50.0}
    tau, h_scaled, alpha, beta, masks = model_arrays(params, t_bhpt, h_bhpt, losses)
    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    if np.count_nonzero(use) < 2000:
        return {"error": 50.0}
    if np.any(np.diff(tau[use]) <= 0.0):
        return {"error": 50.0}
    if np.any(alpha[use] <= 0.0) or np.any(beta[use] <= 0.0):
        return {"error": 50.0}

    common = (t_nr >= tau[use][0]) & (t_nr <= tau[use][-1])
    coverage = np.count_nonzero(common) / len(t_nr)
    if coverage < min_coverage:
        return {"error": 50.0 + (min_coverage - coverage)}

    h_model = base.interp_complex(tau[use], h_scaled[use], t_nr[common])
    return {
        "error": base.mathcalE_error(h_nr[common], h_model),
        "coverage": coverage,
        "common": common,
        "h_ref": h_nr[common],
        "h_model": h_model,
        "tau": tau[use],
        "alpha": alpha[use],
        "beta": beta[use],
        "p_loss": losses["p_loss"][use],
        "e_hat": losses["e_hat"][use],
        "j_hat": losses["j_hat"][use],
        "pre_source": masks["pre"][use],
        "transition_source": masks["transition"][use],
        "post_source": masks["post"][use],
    }


def smooth_start_from_old(
    p_start: float,
    p_width: float,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    losses: dict,
    old_params: np.ndarray = OLD_DISCONTINUOUS_PARAMS,
) -> np.ndarray:
    old_p = old_unpacked(old_params)
    tau_old, _h_old, alpha_old, beta_old, _left = old_model_arrays(
        old_params, t_bhpt, h_bhpt, losses
    )
    p_loss = losses["p_loss"]
    p_end = p_start + p_width
    e_end = float(np.interp(p_end, p_loss, losses["e_hat"]))
    j_end = float(np.interp(p_end, p_loss, losses["j_hat"]))
    e_old = float(np.interp(old_p["p_cut"], p_loss, losses["e_hat"]))
    j_old = float(np.interp(old_p["p_cut"], p_loss, losses["j_hat"]))

    return np.array(
        [
            p_start,
            p_width,
            float(np.interp(p_start, p_loss, tau_old)),
            float(np.interp(p_start, p_loss, beta_old)),
            old_p["beta_right_cut"]
            + old_p["beta_right_e"] * (e_end - e_old)
            + old_p["beta_right_j"] * (j_end - j_old),
            old_p["beta_right_e"],
            old_p["beta_right_j"],
            float(np.interp(p_start, p_loss, alpha_old)),
            old_p["alpha_right_cut"]
            + old_p["alpha_right_e"] * (e_end - e_old)
            + old_p["alpha_right_j"] * (j_end - j_old),
            old_p["alpha_right_e"],
            old_p["alpha_right_j"],
            old_p["phi0"],
        ],
        dtype=float,
    )


def candidate_starts(
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    losses: dict,
) -> list[np.ndarray]:
    starts: list[np.ndarray] = [ACCEPTED_PN_OPT_PARAMS.copy()]

    for p_start, p_width in (
        (-1.12, 0.12),
        (-1.08, 0.12),
        (-1.04, 0.10),
        (-1.02, 0.10),
        (-1.00, 0.10),
        (-1.00, 0.12),
        (-0.98, 0.08),
        (-0.98, 0.12),
        (-0.96, 0.08),
        (-0.94, 0.08),
        (-0.92, 0.06),
        (-1.15, 0.18),
    ):
        start = smooth_start_from_old(p_start, p_width, t_bhpt, h_bhpt, losses)
        starts.append(start)
        for alpha_shift in (-0.004, 0.0, 0.004):
            for beta_shift in (-0.004, 0.0, 0.004):
                shifted = start.copy()
                shifted[3] += beta_shift
                shifted[7] += alpha_shift
                starts.append(shifted)

    combined = PREVIOUS_PN_SEED
    for p_start, p_width in ((-1.10, 0.14), (-1.05, 0.12), (-1.00, 0.10)):
        p_cut_old = combined[0]
        p_end = p_start + p_width
        e_old = float(np.interp(p_cut_old, losses["p_loss"], losses["e_hat"]))
        j_old = float(np.interp(p_cut_old, losses["p_loss"], losses["j_hat"]))
        e_end = float(np.interp(p_end, losses["p_loss"], losses["e_hat"]))
        j_end = float(np.interp(p_end, losses["p_loss"], losses["j_hat"]))
        starts.append(
            np.array(
                [
                    p_start,
                    p_width,
                    combined[1],
                    combined[2],
                    combined[3] + combined[4] * (e_end - e_old) + combined[5] * (j_end - j_old),
                    combined[4],
                    combined[5],
                    combined[6],
                    combined[7] + combined[8] * (e_end - e_old) + combined[9] * (j_end - j_old),
                    combined[8],
                    combined[9],
                    combined[10],
                ],
                dtype=float,
            )
        )
    return starts


def fit_params(
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    losses: dict,
    top_n: int,
    maxiter: int,
) -> tuple[np.ndarray, float, dict]:
    sample = np.unique(np.r_[np.arange(0, len(t_nr), 5), len(t_nr) - 1])
    t_fit = t_nr[sample]
    h_fit = h_nr[sample]

    ranked = []
    for start in candidate_starts(t_bhpt, h_bhpt, losses):
        ev = evaluate_model(start, t_bhpt, h_bhpt, t_fit, h_fit, losses, min_coverage=0.90)
        if ev["error"] < 5.0:
            ranked.append((float(ev["error"]), start))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        raise RuntimeError("No usable smooth PN-loss opt starting point found.")

    best_params = ranked[0][1]
    best_fit_error = ranked[0][0]
    best_eval = evaluate_model(best_params, t_bhpt, h_bhpt, t_nr, h_nr, losses, min_coverage=0.88)
    print(f"ranked_smooth_pn_opt_starts={len(ranked)}")
    for index, (start_error, start) in enumerate(ranked[:top_n]):
        result = minimize(
            lambda params: evaluate_model(
                params, t_bhpt, h_bhpt, t_fit, h_fit, losses, min_coverage=0.88
            )["error"],
            start,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1.0e-8, "fatol": 1.0e-10},
        )
        full = evaluate_model(result.x, t_bhpt, h_bhpt, t_nr, h_nr, losses, min_coverage=0.88)
        print(
            f"run={index} start={start_error:.15g} fit={result.fun:.15g} "
            f"full={full['error']:.15g} nit={result.nit}"
        )
        if full["error"] < best_eval["error"]:
            best_params = result.x
            best_fit_error = float(result.fun)
            best_eval = full

    return best_params, best_fit_error, best_eval


def endpoint_values(params: np.ndarray, eval_data: dict) -> dict[str, float]:
    p = unpack(params)
    alpha = eval_data["alpha"]
    beta = eval_data["beta"]
    pre = eval_data["pre_source"]
    transition = eval_data["transition_source"]
    post = eval_data["post_source"]
    return {
        "alpha_pre_start": float(alpha[pre][0]),
        "alpha_start": p["alpha_left"],
        "alpha_end": p["alpha_right_edge"],
        "alpha_final": float(alpha[post][-1]),
        "beta_pre_start": float(beta[pre][0]),
        "beta_start": p["beta_left"],
        "beta_end": p["beta_right_edge"],
        "beta_final": float(beta[post][-1]),
        "transition_samples": int(np.count_nonzero(transition)),
    }


def local_error(h_ref: np.ndarray, h_model: np.ndarray, mask: np.ndarray) -> float:
    if np.count_nonzero(mask) == 0:
        return float("nan")
    return base.mathcalE_error(h_ref[mask], h_model[mask])


def write_markdown(
    params: np.ndarray,
    fit_error: float,
    eval_data: dict,
    t_bhpt: np.ndarray,
    t_nr: np.ndarray,
    losses: dict,
    meta: dict,
) -> None:
    if eval_data["error"] >= TARGET:
        raise RuntimeError(
            f"Smooth PN-loss opt fit error {eval_data['error']:.15g} exceeds {TARGET:g}"
        )

    p = unpack(params)
    common_t = t_nr[eval_data["common"]]
    e_start = float(np.interp(p["p_start"], losses["p_loss"], losses["e_hat"]))
    j_start = float(np.interp(p["p_start"], losses["p_loss"], losses["j_hat"]))
    e_end = float(np.interp(p["p_end"], losses["p_loss"], losses["e_hat"]))
    j_end = float(np.interp(p["p_end"], losses["p_loss"], losses["j_hat"]))
    t_bhpt_start = float(np.interp(p["p_start"], losses["p_loss"], t_bhpt))
    t_bhpt_end = float(np.interp(p["p_end"], losses["p_loss"], t_bhpt))
    t_end_nr = float(np.interp(p["p_end"], eval_data["p_loss"], eval_data["tau"]))

    pre_mask = common_t <= p["t_start_nr"]
    trans_mask = (common_t > p["t_start_nr"]) & (common_t < t_end_nr)
    post_mask = common_t >= t_end_nr
    pre_err = local_error(eval_data["h_ref"], eval_data["h_model"], pre_mask)
    trans_err = local_error(eval_data["h_ref"], eval_data["h_model"], trans_mask)
    post_err = local_error(eval_data["h_ref"], eval_data["h_model"], post_mask)
    endpoints = endpoint_values(params, eval_data)
    target_line = (
        "The requested `mathcalE ~ 1e-4` target is reached."
        if eval_data["error"] < DESIRED
        else "The fit does not reach `mathcalE ~ 1e-4`, but remains below `1e-3`."
    )

    lines = [
        "# Smooth optimized PN-loss q=5 (2,2) scaling fit",
        "",
        "This replaces the discontinuous constant-left PN-loss opt fit with a smooth",
        "constant-to-linear transition. Before the transition, `alpha` and `beta` are",
        "constant. Across the transition, a cubic Hermite interpolation matches the",
        "constant branch value and the post-transition branch value and slope along",
        "`p_loss`. After the transition, the fit is linear in the PN radiated-energy",
        "and radiated-angular-momentum coordinates.",
        "",
        "The BHPT waveform is the raw surrogate output with `calibrated=False`; the",
        "NR reference is `NRHybSur3dq8`.",
        "",
        "```python",
        f"q = {Q}",
        "nu = q / (1 + q)**2",
        f"nu = {meta['nu']:.15g}",
        f"t_bhpt_merger = {meta['t_bhpt_merger']:.15g}",
        f"t_nr_merger = {meta['t_nr_merger']:.15g}",
        f"t_ref = {T_REF:.15g}",
        "```",
        "",
        "PN-loss coordinates:",
        "",
        "```python",
        "omega_gw = abs(d unwrap(arg(h_BHPT_22)) / dt_bhpt)",
        "x_pn = (omega_gw / 2)**(2/3)",
        "F_E = PN_exprs.dEdt(nu, x_pn)",
        "F_J = PN_exprs.dJdt(1.0, x_pn, nu)",
        "DeltaE_PN(t) = integral_from_merger_to_t F_E dt",
        "DeltaJ_PN(t) = integral_from_merger_to_t F_J dt",
        "Ehat = DeltaE_PN / abs(DeltaE_PN(t_ref))",
        "Jhat = DeltaJ_PN / abs(DeltaJ_PN(t_ref))",
        "p_loss = 0.5 * (Ehat + Jhat)",
        "```",
        "",
        "Optimized PN-loss transition:",
        "",
        "```python",
        f"p_start = {p['p_start']:.15g}",
        f"p_width = {p['p_width']:.15g}",
        f"p_end = {p['p_end']:.15g}",
        f"Ehat_start = {e_start:.15g}",
        f"Jhat_start = {j_start:.15g}",
        f"Ehat_end = {e_end:.15g}",
        f"Jhat_end = {j_end:.15g}",
        f"t_bhpt_start = {t_bhpt_start:.15g}",
        f"t_bhpt_end = {t_bhpt_end:.15g}",
        f"t_start_nr = {p['t_start_nr']:.15g}",
        f"t_end_nr = {t_end_nr:.15g}",
        f"DeltaE_PN(t_ref) = {-losses['e_ref']:.15g}",
        f"DeltaJ_PN(t_ref) = {-losses['j_ref']:.15g}",
        "```",
        "",
        "For `p_loss <= p_start`, the pre-transition branch is constant:",
        "",
        "```python",
        f"alpha_left = {p['alpha_left']:.15g}",
        f"beta_left = {p['beta_left']:.15g}",
        "```",
        "",
        "For `p_loss >= p_end`:",
        "",
        "```python",
        "dE = Ehat - Ehat_end",
        "dJ = Jhat - Jhat_end",
        f"alpha_right = {p['alpha_right_edge']:.15g} + ({p['alpha_right_e']:.15g}) * dE + ({p['alpha_right_j']:.15g}) * dJ",
        f"beta_right = {p['beta_right_edge']:.15g} + ({p['beta_right_e']:.15g}) * dE + ({p['beta_right_j']:.15g}) * dJ",
        "```",
        "",
        "For `p_start < p_loss < p_end`, use cubic Hermite interpolation in",
        "`z = (p_loss - p_start) / p_width` between the constant values and the",
        "right-branch edge values. The left endpoint slope is zero, and the right",
        "endpoint slope is the derivative of the right branch along the PN-loss path.",
        "",
        "The time map is the numerical integral of `beta` anchored at `p_start`:",
        "",
        "```python",
        "tau(t_bhpt) = t_start_nr + integral_from_t_bhpt_start_to_t_bhpt beta(t') dt'",
        "```",
        "",
        "Phase rotation:",
        "",
        "```python",
        f"phi0 = {base.principal_phase(p['phi0']):.15g}  # radians, principal value",
        f"phi0_unwrapped = {p['phi0']:.15g}",
        "```",
        "",
        "Applied as:",
        "",
        "```python",
        "h_model(tau(t_bhpt)) = alpha(Ehat, Jhat) * exp(1j * phi0) * h_BHPT(t_bhpt)",
        "```",
        "",
        "Diagnostics:",
        "",
        f"- q: `{Q}`",
        f"- mode: `{MODE}`",
        f"- common NR support: `[{common_t[0]:.15g}, {common_t[-1]:.15g}]`",
        f"- common-support coverage of requested NR window: `{eval_data['coverage']:.15g}`",
        f"- optimizer downsampled `mathcalE`: `{fit_error:.15g}`",
        f"- full-window `mathcalE`: `{eval_data['error']:.15g}`",
        f"- pre-transition local `mathcalE`: `{pre_err:.15g}`",
        f"- transition local `mathcalE`: `{trans_err:.15g}`",
        f"- post-transition local `mathcalE`: `{post_err:.15g}`",
        f"- alpha endpoint values `(pre_start, start, end, final)`: `({endpoints['alpha_pre_start']:.15g}, {endpoints['alpha_start']:.15g}, {endpoints['alpha_end']:.15g}, {endpoints['alpha_final']:.15g})`",
        f"- beta endpoint values `(pre_start, start, end, final)`: `({endpoints['beta_pre_start']:.15g}, {endpoints['beta_start']:.15g}, {endpoints['beta_end']:.15g}, {endpoints['beta_final']:.15g})`",
        f"- source samples in smooth transition: `{endpoints['transition_samples']}`",
        "",
        target_line,
        "The main caveat remains that the 2PN flux expressions are used as an IMR",
        "coordinate through merger-ringdown, where they should be interpreted as a",
        "physically motivated proxy rather than a controlled PN approximation.",
        "",
    ]
    (base.ROOT / "scaling_PN_opt.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=3200)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Run the local optimizer again instead of recording the accepted best run.",
    )
    args = parser.parse_args()

    t_bhpt, h_bhpt, t_nr, h_nr = base.load_waveforms()
    meta = base.physical_metadata(t_bhpt, h_bhpt, t_nr, h_nr)
    losses = base.pn_loss_coordinates(t_bhpt, h_bhpt, meta)
    if args.refit:
        params, fit_error, eval_data = fit_params(
            t_bhpt, h_bhpt, t_nr, h_nr, losses, args.top_n, args.maxiter
        )
    else:
        params = ACCEPTED_PN_OPT_PARAMS.copy()
        eval_data = evaluate_model(params, t_bhpt, h_bhpt, t_nr, h_nr, losses, min_coverage=0.88)
        sample = np.unique(np.r_[np.arange(0, len(t_nr), 5), len(t_nr) - 1])
        fit_error = evaluate_model(
            params,
            t_bhpt,
            h_bhpt,
            t_nr[sample],
            h_nr[sample],
            losses,
            min_coverage=0.88,
        )["error"]
    write_markdown(params, fit_error, eval_data, t_bhpt, t_nr, losses, meta)

    p = unpack(params)
    print(f"mathcalE={eval_data['error']:.15g}")
    print(f"p_start={p['p_start']:.15g}")
    print(f"p_end={p['p_end']:.15g}")
    print(f"t_start_nr={p['t_start_nr']:.15g}")
    print(f"phi0={base.principal_phase(p['phi0']):.15g}")
    print("params=" + np.array2string(params, precision=15))


if __name__ == "__main__":
    main()
