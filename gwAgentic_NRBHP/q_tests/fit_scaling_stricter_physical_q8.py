from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
import fit_scaling_stricter_physical as base  # noqa: E402


Q = 8
MODE = (2, 2)
TARGET = 3.0e-4

# A q=8-specific single-segment physical alignment. This is deliberately not
# the q=5 stricter fit; it is only used to put the two-segment optimizer in the
# correct q=8 basin. Use --refit-linear to recompute this seed.
LINEAR_SEED = np.array(
    [
        8.61613084e-01,  # alpha0
        -2.78742980e-05,  # alpha1
        8.66001650e-01,  # beta0
        -3.49089417e-06,  # beta1
        2.90826119e00,  # Delta_tau
        -7.55750995e00,  # phi0
    ],
    dtype=float,
)


def configure_base() -> None:
    base.Q = Q
    base.TARGET = TARGET


def linear_model_error(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    meta: dict,
    min_coverage: float = 0.90,
) -> float:
    alpha0, alpha1, beta0, beta1, delta_tau, phi0 = params
    nu = meta["nu"]
    dt_bhpt = t_bhpt - meta["t_bhpt_merger"]
    theta = nu * dt_bhpt
    alpha = alpha0 + alpha1 * theta
    beta = beta0 + beta1 * theta
    tau = (
        meta["t_nr_merger"]
        + delta_tau
        + beta0 * dt_bhpt
        + 0.5 * beta1 * nu * dt_bhpt**2
    )
    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    if np.count_nonzero(use) < 1000:
        return 99.0
    if np.any(np.diff(tau[use]) <= 0.0):
        return 99.0
    if np.any(alpha[use] <= 0.0) or np.any(beta[use] <= 0.0):
        return 99.0

    common = (t_nr >= tau[use][0]) & (t_nr <= tau[use][-1])
    if np.count_nonzero(common) / len(t_nr) < min_coverage:
        return 99.0

    h_model = base.interp_complex(
        tau[use], alpha[use] * np.exp(1j * phi0) * h_bhpt[use], t_nr[common]
    )
    return base.mathcalE_error(h_nr[common], h_model)


def phase_seed_linear(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    meta: dict,
) -> np.ndarray:
    seeded = params.copy()
    seeded[5] = 0.0
    alpha0, alpha1, beta0, beta1, delta_tau, _phi0 = seeded
    nu = meta["nu"]
    dt_bhpt = t_bhpt - meta["t_bhpt_merger"]
    theta = nu * dt_bhpt
    alpha = alpha0 + alpha1 * theta
    tau = (
        meta["t_nr_merger"]
        + delta_tau
        + beta0 * dt_bhpt
        + 0.5 * beta1 * nu * dt_bhpt**2
    )
    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    if np.count_nonzero(use) < 1000 or np.any(np.diff(tau[use]) <= 0.0):
        return params
    if np.any(alpha[use] <= 0.0):
        return params

    common = (t_nr >= tau[use][0]) & (t_nr <= tau[use][-1])
    if np.count_nonzero(common) < 1000:
        return params
    h_model = base.interp_complex(tau[use], alpha[use] * h_bhpt[use], t_nr[common])
    seeded[5] = base.principal_phase(
        float(np.angle(np.sum(h_nr[common] * h_model.conjugate())))
    )
    return seeded


def generic_linear_starts() -> list[np.ndarray]:
    starts = []
    for alpha0 in (0.55, 0.65, 0.75, 0.85, 1.0):
        for alpha1 in (-1.0e-4, 0.0, 1.0e-4):
            for beta0 in (0.6, 0.7, 0.8, 0.9, 1.0):
                for beta1 in (-2.0e-5, 0.0, 2.0e-5):
                    for delta_tau in (-100.0, 0.0, 100.0, 300.0):
                        starts.append(
                            np.array(
                                [alpha0, alpha1, beta0, beta1, delta_tau, 0.0],
                                dtype=float,
                            )
                        )
    return starts


def fit_linear_seed(
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    meta: dict,
) -> np.ndarray:
    sample = np.unique(np.r_[np.arange(0, len(t_nr), 5), len(t_nr) - 1])
    t_fit = t_nr[sample]
    h_fit = h_nr[sample]

    ranked = []
    for start in generic_linear_starts():
        seeded = phase_seed_linear(start, t_bhpt, h_bhpt, t_fit, h_fit, meta)
        err = linear_model_error(seeded, t_bhpt, h_bhpt, t_fit, h_fit, meta)
        if err < 10.0:
            ranked.append((err, seeded))
    ranked.sort(key=lambda item: item[0])

    best_params = ranked[0][1]
    best_error = 99.0
    for _err, start in ranked[:5]:
        result = minimize(
            lambda params: linear_model_error(
                params, t_bhpt, h_bhpt, t_fit, h_fit, meta
            ),
            start,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-10},
        )
        full_error = linear_model_error(result.x, t_bhpt, h_bhpt, t_nr, h_nr, meta)
        if full_error < best_error:
            best_error = full_error
            best_params = result.x
    print(f"linear_seed_mathcalE={best_error:.15g}")
    print("linear_seed=" + np.array2string(best_params, precision=15))
    return best_params


def linear_tau(theta: float, linear_seed: np.ndarray, meta: dict) -> float:
    _alpha0, _alpha1, beta0, beta1, delta_tau, _phi0 = linear_seed
    dt_bhpt = theta / meta["nu"]
    return (
        meta["t_nr_merger"]
        + delta_tau
        + beta0 * dt_bhpt
        + 0.5 * beta1 * meta["nu"] * dt_bhpt**2
    )


def strict_start(
    theta_cut: float,
    linear_seed: np.ndarray,
    meta: dict,
    alpha_jump: float = 0.0,
    beta_jump: float = 0.0,
    alpha_right_slope: float | None = None,
    beta_right_slope: float | None = None,
) -> np.ndarray:
    alpha0, alpha1, beta0, beta1, _delta_tau, phi0 = linear_seed
    if alpha_right_slope is None:
        alpha_right_slope = alpha1
    if beta_right_slope is None:
        beta_right_slope = beta1
    alpha_cut = alpha0 + alpha1 * theta_cut
    beta_cut = beta0 + beta1 * theta_cut
    return np.array(
        [
            theta_cut,
            linear_tau(theta_cut, linear_seed, meta),
            beta_cut,
            beta1,
            beta_cut + beta_jump,
            beta_right_slope,
            alpha_cut,
            alpha1,
            alpha_cut + alpha_jump,
            alpha_right_slope,
            phi0,
        ],
        dtype=float,
    )


def q8_strict_starts(linear_seed: np.ndarray, meta: dict) -> list[np.ndarray]:
    alpha1 = linear_seed[1]
    beta1 = linear_seed[3]
    starts = []
    for theta_cut in (-90.0, -75.0, -65.0, -55.0, -45.0, -35.0, -28.0, -22.0, -16.0):
        if linear_tau(theta_cut, linear_seed, meta) >= -100.0:
            continue
        for alpha_jump in (0.0, 0.02, -0.02, 0.05):
            for beta_jump in (0.0, -0.02, 0.02):
                for alpha_right_slope in (alpha1, -1.0e-3, -2.0e-3, 5.0e-4):
                    for beta_right_slope in (beta1, 2.0e-4, 5.0e-4, 1.0e-3, -2.0e-4):
                        starts.append(
                            strict_start(
                                theta_cut,
                                linear_seed,
                                meta,
                                alpha_jump,
                                beta_jump,
                                alpha_right_slope,
                                beta_right_slope,
                            )
                        )
    return starts


def fit_q8_strict(
    linear_seed: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    meta: dict,
    top_n: int,
    maxiter: int,
) -> tuple[np.ndarray, float, dict]:
    sample = np.unique(np.r_[np.arange(0, len(t_nr), 5), len(t_nr) - 1])
    t_fit = t_nr[sample]
    h_fit = h_nr[sample]

    ranked = []
    for start in q8_strict_starts(linear_seed, meta):
        ev = base.evaluate_model(start, t_bhpt, h_bhpt, t_fit, h_fit, meta, min_coverage=0.90)
        if ev["error"] < 5.0:
            ranked.append((float(ev["error"]), start))
    ranked.sort(key=lambda item: item[0])
    print(f"ranked_two_segment_starts={len(ranked)}")

    best_params = ranked[0][1]
    best_fit_error = ranked[0][0]
    best_eval = base.evaluate_model(best_params, t_bhpt, h_bhpt, t_nr, h_nr, meta)
    for index, (start_error, start) in enumerate(ranked[:top_n]):
        result = minimize(
            base.bounded_objective,
            start,
            args=(t_bhpt, h_bhpt, t_fit, h_fit, meta),
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-10},
        )
        full = base.evaluate_model(result.x, t_bhpt, h_bhpt, t_nr, h_nr, meta)
        print(
            f"run={index} start={start_error:.15g} fit={result.fun:.15g} "
            f"full={full['error']:.15g} nit={result.nit}"
        )
        if full["error"] < best_eval["error"]:
            best_params = result.x
            best_fit_error = float(result.fun)
            best_eval = full

    return best_params, best_fit_error, best_eval


def write_q8_markdown(params: np.ndarray, fit_error: float, eval_data: dict, t_nr: np.ndarray, meta: dict) -> None:
    if eval_data["error"] >= TARGET:
        raise RuntimeError(
            f"q=8 stricter physical fit error {eval_data['error']:.15g} exceeds {TARGET:g}"
        )

    p = base.unpack(params)
    common_t = t_nr[eval_data["common"]]
    theta_cut = p["theta_cut"]
    t_bhpt_cut = meta["t_bhpt_merger"] + theta_cut / meta["nu"]
    theta_cut_nr = meta["nu"] * (p["t_cut_nr"] - meta["t_nr_merger"])
    left_mask = common_t <= p["t_cut_nr"]
    right_mask = common_t > p["t_cut_nr"]
    left_err = base.mathcalE_error(eval_data["h_ref"][left_mask], eval_data["h_model"][left_mask])
    right_err = base.mathcalE_error(eval_data["h_ref"][right_mask], eval_data["h_model"][right_mask])
    endpoints = base.endpoint_values(params, eval_data)

    lines = [
        "# Stricter physical q=8 (2,2) piecewise-linear scaling fit",
        "",
        "This q=8 run uses the same two-segment physical ansatz as the q=5 stricter",
        "physical run, but it does not use the q=5 generic start grid as the final",
        "fitting procedure. The q=5 start grid is not generally applicable: at q=8 the",
        "best q=5-style starting point had `mathcalE ~= 0.61` before polishing, so the",
        "optimizer starts in the wrong basin and can spend a long time improving a bad",
        "alignment.",
        "",
        "For this run I first used a q=8-specific single-segment physical alignment,",
        "then seeded the two-segment fit from that q=8 alignment and polished the best",
        "two-segment starts with Nelder-Mead.",
        "",
        "The BHPT waveform is the raw surrogate output with `calibrated=False`; the NR",
        "reference is `NRHybSur3dq8`.",
        "",
        "```python",
        f"q = {Q}",
        "nu = q / (1 + q)**2",
        f"nu = {meta['nu']:.15g}",
        "",
        f"t_bhpt_merger = {meta['t_bhpt_merger']:.15g}",
        f"t_nr_merger = {meta['t_nr_merger']:.15g}",
        "theta = nu * (t_bhpt - t_bhpt_merger)",
        "Theta_nr = nu * (t_nr - t_nr_merger)",
        "```",
        "",
        "The optimized source split and mapped NR cutoff are:",
        "",
        "```python",
        f"theta_cut = {theta_cut:.15g}",
        f"t_bhpt_cut = t_bhpt_merger + theta_cut / nu  # {t_bhpt_cut:.15g}",
        f"Theta_cut_nr = {theta_cut_nr:.15g}",
        f"t_cut_nr = t_nr_merger + Theta_cut_nr / nu  # {p['t_cut_nr']:.15g}",
        "```",
        "",
        "For `theta <= theta_cut`:",
        "",
        "```python",
        "dtheta = theta - theta_cut",
        f"alpha_left(theta) = {p['alpha_left_cut']:.15g} + ({p['alpha_left_slope']:.15g}) * dtheta",
        f"beta_left(theta) = {p['beta_left_cut']:.15g} + ({p['beta_left_slope']:.15g}) * dtheta",
        "dt_bhpt = t_bhpt - t_bhpt_cut",
        f"tau_left(t_bhpt) = t_cut_nr + {p['beta_left_cut']:.15g} * dt_bhpt + 0.5 * ({p['beta_left_slope']:.15g}) * nu * dt_bhpt**2",
        "```",
        "",
        "For `theta > theta_cut`:",
        "",
        "```python",
        "dtheta = theta - theta_cut",
        f"alpha_right(theta) = {p['alpha_right_cut']:.15g} + ({p['alpha_right_slope']:.15g}) * dtheta",
        f"beta_right(theta) = {p['beta_right_cut']:.15g} + ({p['beta_right_slope']:.15g}) * dtheta",
        "dt_bhpt = t_bhpt - t_bhpt_cut",
        f"tau_right(t_bhpt) = t_cut_nr + {p['beta_right_cut']:.15g} * dt_bhpt + 0.5 * ({p['beta_right_slope']:.15g}) * nu * dt_bhpt**2",
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
        "h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)",
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
        f"- `t <= t_cut_nr` local `mathcalE`: `{left_err:.15g}`",
        f"- `t > t_cut_nr` local `mathcalE`: `{right_err:.15g}`",
        f"- alpha endpoint values `(left_start, left_cut, right_cut, right_end)`: `({endpoints['alpha_left_start']:.15g}, {endpoints['alpha_left_cut']:.15g}, {endpoints['alpha_right_cut']:.15g}, {endpoints['alpha_right_end']:.15g})`",
        f"- beta endpoint values `(left_start, left_cut, right_cut, right_end)`: `({endpoints['beta_left_start']:.15g}, {endpoints['beta_left_cut']:.15g}, {endpoints['beta_right_cut']:.15g}, {endpoints['beta_right_end']:.15g})`",
        "",
        "No extrapolated BHPT samples are used. The evaluated source samples are the raw",
        "BHPT samples whose fitted `tau(t_bhpt)` lies inside the NR comparison support,",
        "and the NR grid is then restricted to that common support.",
        "",
    ]
    (ROOT / "scaling_stricter_physical_q_8.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-linear", action="store_true", help="recompute the q=8 single-segment seed")
    parser.add_argument("--top-n", type=int, default=4, help="number of ranked two-segment starts to polish")
    parser.add_argument("--maxiter", type=int, default=1600, help="Nelder-Mead max iterations per start")
    args = parser.parse_args()

    configure_base()
    t_bhpt, h_bhpt, t_nr, h_nr = base.load_waveforms()
    meta = base.physical_metadata(t_bhpt, h_bhpt, t_nr, h_nr)
    linear_seed = (
        fit_linear_seed(t_bhpt, h_bhpt, t_nr, h_nr, meta)
        if args.refit_linear
        else LINEAR_SEED.copy()
    )
    params, fit_error, eval_data = fit_q8_strict(
        linear_seed, t_bhpt, h_bhpt, t_nr, h_nr, meta, args.top_n, args.maxiter
    )
    write_q8_markdown(params, fit_error, eval_data, t_nr, meta)

    p = base.unpack(params)
    print(f"mathcalE={eval_data['error']:.15g}")
    print(f"theta_cut={p['theta_cut']:.15g}")
    print(f"t_cut_nr={p['t_cut_nr']:.15g}")
    print(f"phi0={base.principal_phase(p['phi0']):.15g}")
    print("params=" + np.array2string(params, precision=15))


if __name__ == "__main__":
    main()
