from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.optimize import minimize

import gwsurrogate


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.append(str(REPO_ROOT / "BHPTNRSurrogate" / "surrogates"))
import BHPTNRSur1dq1e4 as bhptsur  # noqa: E402


Q = 5
MODE = (2, 2)
NR_T_START = -5000.1
NR_T_END = 100.0
TARGET = 3.0e-4


def mathcalE_error(h1: np.ndarray, h2: np.ndarray) -> float:
    n1_sqr = np.sum(np.abs(h1) ** 2)
    n2_sqr = np.sum(np.abs(h2) ** 2)
    sdot = np.real(np.sum(h1 * h2.conjugate()))
    return float(((n1_sqr + n2_sqr) - 2.0 * sdot) / (2.0 * n1_sqr))


def interp_complex(t_src: np.ndarray, h_src: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    return np.interp(t_eval, t_src, h_src.real) + 1j * np.interp(
        t_eval, t_src, h_src.imag
    )


def principal_phase(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def load_waveforms():
    t_bhpt, h_bhpt = bhptsur.generate_surrogate(
        q=Q, calibrated=False, modes=[MODE], neg_modes=False
    )
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    t_nr, h_nr, _ = nrsur(Q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)
    nr_mask = (t_nr >= NR_T_START) & (t_nr <= NR_T_END)
    return t_bhpt, h_bhpt[MODE], t_nr[nr_mask], h_nr[MODE][nr_mask]


def physical_metadata(
    t_bhpt: np.ndarray, h_bhpt: np.ndarray, t_nr: np.ndarray, h_nr: np.ndarray
) -> dict:
    return {
        "nu": Q / (1.0 + Q) ** 2,
        "t_bhpt_merger": float(t_bhpt[np.argmax(np.abs(h_bhpt))]),
        "t_nr_merger": float(t_nr[np.argmax(np.abs(h_nr))]),
    }


def unpack(params: np.ndarray) -> dict[str, float]:
    (
        theta_cut,
        t_cut_nr,
        beta_left_cut,
        beta_left_slope,
        beta_right_cut,
        beta_right_slope,
        alpha_left_cut,
        alpha_left_slope,
        alpha_right_cut,
        alpha_right_slope,
        phi0,
    ) = [float(v) for v in params]
    return {
        "theta_cut": theta_cut,
        "t_cut_nr": t_cut_nr,
        "beta_left_cut": beta_left_cut,
        "beta_left_slope": beta_left_slope,
        "beta_right_cut": beta_right_cut,
        "beta_right_slope": beta_right_slope,
        "alpha_left_cut": alpha_left_cut,
        "alpha_left_slope": alpha_left_slope,
        "alpha_right_cut": alpha_right_cut,
        "alpha_right_slope": alpha_right_slope,
        "phi0": phi0,
    }


def model_arrays(
    params: np.ndarray, t_bhpt: np.ndarray, h_bhpt: np.ndarray, meta: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = unpack(params)
    nu = meta["nu"]
    theta = nu * (t_bhpt - meta["t_bhpt_merger"])
    theta_delta = theta - p["theta_cut"]
    s_cut = meta["t_bhpt_merger"] + p["theta_cut"] / nu
    ds = t_bhpt - s_cut
    left = theta <= p["theta_cut"]

    tau = np.empty_like(t_bhpt, dtype=float)
    beta = np.empty_like(t_bhpt, dtype=float)
    alpha = np.empty_like(t_bhpt, dtype=float)

    tau[left] = (
        p["t_cut_nr"]
        + p["beta_left_cut"] * ds[left]
        + 0.5 * p["beta_left_slope"] * nu * ds[left] ** 2
    )
    tau[~left] = (
        p["t_cut_nr"]
        + p["beta_right_cut"] * ds[~left]
        + 0.5 * p["beta_right_slope"] * nu * ds[~left] ** 2
    )
    beta[left] = p["beta_left_cut"] + p["beta_left_slope"] * theta_delta[left]
    beta[~left] = p["beta_right_cut"] + p["beta_right_slope"] * theta_delta[~left]
    alpha[left] = p["alpha_left_cut"] + p["alpha_left_slope"] * theta_delta[left]
    alpha[~left] = p["alpha_right_cut"] + p["alpha_right_slope"] * theta_delta[~left]

    h_scaled = alpha * np.exp(1j * p["phi0"]) * h_bhpt
    return tau, h_scaled, alpha, beta, theta


def validate_basic(params: np.ndarray, meta: dict) -> bool:
    p = unpack(params)
    s_cut = meta["t_bhpt_merger"] + p["theta_cut"] / meta["nu"]
    if not (-120.0 <= p["theta_cut"] <= -10.0):
        return False
    if not (-800.0 <= p["t_cut_nr"] <= -100.001):
        return False
    if not (-900.0 <= s_cut <= -70.0):
        return False
    if not (0.25 <= p["beta_left_cut"] <= 1.5):
        return False
    if not (0.25 <= p["beta_right_cut"] <= 1.5):
        return False
    if not (-1.0e-4 <= p["beta_left_slope"] <= 1.0e-4):
        return False
    if not (-1.0e-2 <= p["beta_right_slope"] <= 1.0e-2):
        return False
    if not (0.05 <= p["alpha_left_cut"] <= 2.5):
        return False
    if not (0.05 <= p["alpha_right_cut"] <= 2.5):
        return False
    if not (-2.0e-2 <= p["alpha_left_slope"] <= 2.0e-2):
        return False
    if not (-2.0e-2 <= p["alpha_right_slope"] <= 2.0e-2):
        return False
    return True


def evaluate_model(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    meta: dict,
    min_coverage: float = 0.94,
) -> dict:
    if not validate_basic(params, meta):
        return {"error": 25.0}

    tau, h_scaled, alpha, beta, theta = model_arrays(params, t_bhpt, h_bhpt, meta)
    use = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    if np.count_nonzero(use) < 2000:
        return {"error": 25.0}

    tau_use = tau[use]
    if np.any(np.diff(tau_use) <= 0.0):
        return {"error": 25.0}
    if np.any(alpha[use] <= 0.0) or np.any(beta[use] <= 0.0):
        return {"error": 25.0}

    common = (t_nr >= tau_use[0]) & (t_nr <= tau_use[-1])
    coverage = np.count_nonzero(common) / len(t_nr)
    if coverage < min_coverage:
        return {"error": 25.0 + (min_coverage - coverage)}

    h_model = interp_complex(tau_use, h_scaled[use], t_nr[common])
    err = mathcalE_error(h_nr[common], h_model)
    return {
        "error": err,
        "coverage": coverage,
        "common": common,
        "h_ref": h_nr[common],
        "h_model": h_model,
        "tau": tau_use,
        "alpha": alpha[use],
        "beta": beta[use],
        "theta": theta[use],
    }


def phase_seed(params: np.ndarray, t_bhpt, h_bhpt, t_nr, h_nr, meta) -> np.ndarray:
    seeded = params.copy()
    seeded[10] = 0.0
    ev = evaluate_model(seeded, t_bhpt, h_bhpt, t_nr, h_nr, meta, min_coverage=0.90)
    if ev["error"] >= 20.0:
        return params
    h_model = ev["h_model"]
    seeded[10] = principal_phase(float(np.angle(np.sum(ev["h_ref"] * h_model.conjugate()))))
    return seeded


def generic_starts() -> list[np.ndarray]:
    starts = []
    for theta_cut, t_cut_nr in (
        (-65.0, -420.0),
        (-50.0, -310.0),
        (-38.0, -240.0),
        (-32.0, -190.0),
        (-28.0, -170.0),
        (-22.0, -140.0),
    ):
        for beta_right_slope in (0.0, 1.0e-3, 2.0e-3, 3.0e-3):
            starts.append(
                np.array(
                    [
                        theta_cut,
                        t_cut_nr,
                        0.805,
                        0.0,
                        0.81,
                        beta_right_slope,
                        0.81,
                        0.0,
                        0.84,
                        -2.5e-3,
                        0.0,
                    ],
                    dtype=float,
                )
            )
    return starts


def bounded_objective(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    meta: dict,
) -> float:
    ev = evaluate_model(params, t_bhpt, h_bhpt, t_nr, h_nr, meta, min_coverage=0.94)
    return float(ev["error"])


def fit_params(t_bhpt, h_bhpt, t_nr, h_nr, meta) -> tuple[np.ndarray, float]:
    sample = np.unique(np.r_[np.arange(0, len(t_nr), 5), len(t_nr) - 1])
    t_fit = t_nr[sample]
    h_fit = h_nr[sample]

    ranked_starts = []
    for start in generic_starts():
        seeded = phase_seed(start, t_bhpt, h_bhpt, t_fit, h_fit, meta)
        ev = evaluate_model(seeded, t_bhpt, h_bhpt, t_fit, h_fit, meta, min_coverage=0.90)
        if ev["error"] < 0.02:
            ranked_starts.append((float(ev["error"]), seeded))
    ranked_starts.sort(key=lambda item: item[0])
    if not ranked_starts:
        ranked_starts = [
            (25.0, phase_seed(start, t_bhpt, h_bhpt, t_fit, h_fit, meta))
            for start in generic_starts()
        ]

    best_params = None
    best_error = 25.0
    for _start_error, seeded in ranked_starts[:6]:
        result = minimize(
            bounded_objective,
            seeded,
            args=(t_bhpt, h_bhpt, t_fit, h_fit, meta),
            method="Nelder-Mead",
            options={"maxiter": 1200, "xatol": 1e-8, "fatol": 1e-10, "disp": False},
        )
        if float(result.fun) < best_error:
            best_error = float(result.fun)
            best_params = result.x

    return best_params, best_error


def endpoint_values(params: np.ndarray, eval_data: dict) -> dict:
    theta = eval_data["theta"]
    alpha = eval_data["alpha"]
    beta = eval_data["beta"]
    p = unpack(params)
    left = theta <= p["theta_cut"]
    right = theta > p["theta_cut"]
    return {
        "alpha_left_start": float(alpha[left][0]),
        "alpha_left_cut": p["alpha_left_cut"],
        "alpha_right_cut": p["alpha_right_cut"],
        "alpha_right_end": float(alpha[right][-1]),
        "beta_left_start": float(beta[left][0]),
        "beta_left_cut": p["beta_left_cut"],
        "beta_right_cut": p["beta_right_cut"],
        "beta_right_end": float(beta[right][-1]),
    }


def write_markdown(
    params: np.ndarray,
    fit_error: float,
    eval_data: dict,
    t_nr: np.ndarray,
    meta: dict,
) -> None:
    if eval_data["error"] >= TARGET:
        raise RuntimeError(
            f"stricter physical fit error {eval_data['error']:.15g} exceeds {TARGET:g}"
        )

    p = unpack(params)
    common_t = t_nr[eval_data["common"]]
    h_ref = eval_data["h_ref"]
    h_model = eval_data["h_model"]
    theta_cut = p["theta_cut"]
    s_cut = meta["t_bhpt_merger"] + theta_cut / meta["nu"]
    theta_cut_nr = meta["nu"] * (p["t_cut_nr"] - meta["t_nr_merger"])
    left_mask = common_t <= p["t_cut_nr"]
    right_mask = common_t > p["t_cut_nr"]
    left_err = mathcalE_error(h_ref[left_mask], h_model[left_mask])
    right_err = mathcalE_error(h_ref[right_mask], h_model[right_mask])
    endpoints = endpoint_values(params, eval_data)

    lines = [
        "# Stricter physical q=5 (2,2) piecewise-linear scaling fit",
        "",
        "This rerun fits the two-segment stricter ansatz directly in merger-centered",
        "physical time variables. It does not reuse the previous stricter",
        "`s_min`/`s_max` coefficients.",
        "The optimizer ranks generic physical starts and then applies a Nelder-Mead",
        "polish to the best starts.",
        "",
        "The BHPT waveform is the raw surrogate output with `calibrated=False`; the",
        "NR reference is `NRHybSur3dq8`.",
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
        f"t_bhpt_cut = t_bhpt_merger + theta_cut / nu  # {s_cut:.15g}",
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
        f"phi0 = {principal_phase(p['phi0']):.15g}  # radians",
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
        f"- optimizer downsampled `mathcalE`: `{fit_error:.15g}`",
        f"- full-window `mathcalE`: `{eval_data['error']:.15g}`",
        f"- `t <= t_cut_nr` local `mathcalE`: `{left_err:.15g}`",
        f"- `t > t_cut_nr` local `mathcalE`: `{right_err:.15g}`",
        f"- alpha endpoint values `(left_start, left_cut, right_cut, right_end)`: `({endpoints['alpha_left_start']:.15g}, {endpoints['alpha_left_cut']:.15g}, {endpoints['alpha_right_cut']:.15g}, {endpoints['alpha_right_end']:.15g})`",
        f"- beta endpoint values `(left_start, left_cut, right_cut, right_end)`: `({endpoints['beta_left_start']:.15g}, {endpoints['beta_left_cut']:.15g}, {endpoints['beta_right_cut']:.15g}, {endpoints['beta_right_end']:.15g})`",
        "",
        "No extrapolated BHPT samples are used. The evaluated source samples are the",
        "raw BHPT samples whose fitted `tau(t_bhpt)` lies inside the NR comparison",
        "support, and the NR grid is then restricted to that common support.",
        "",
    ]
    (ROOT / "scaling_stricter_physical.md").write_text("\n".join(lines))


def main() -> None:
    t_bhpt, h_bhpt, t_nr, h_nr = load_waveforms()
    meta = physical_metadata(t_bhpt, h_bhpt, t_nr, h_nr)
    params, fit_error = fit_params(t_bhpt, h_bhpt, t_nr, h_nr, meta)
    eval_data = evaluate_model(params, t_bhpt, h_bhpt, t_nr, h_nr, meta)
    write_markdown(params, fit_error, eval_data, t_nr, meta)

    p = unpack(params)
    print(f"mathcalE={eval_data['error']:.15g}")
    print(f"theta_cut={p['theta_cut']:.15g}")
    print(f"t_cut_nr={p['t_cut_nr']:.15g}")
    print(f"phi0={principal_phase(p['phi0']):.15g}")
    print("params=" + np.array2string(params, precision=15))


if __name__ == "__main__":
    main()
