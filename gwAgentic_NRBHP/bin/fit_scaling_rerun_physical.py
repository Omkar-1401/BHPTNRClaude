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
TARGET = 1e-3


def mathcalE_error(h1: np.ndarray, h2: np.ndarray) -> float:
    n1_sqr = np.sum(np.abs(h1) ** 2)
    n2_sqr = np.sum(np.abs(h2) ** 2)
    sdot = np.real(np.sum(h1 * h2.conjugate()))
    return float(((n1_sqr + n2_sqr) - 2.0 * sdot) / (2.0 * n1_sqr))


def interp_complex(
    t_src: np.ndarray, h_src: np.ndarray, t_eval: np.ndarray
) -> np.ndarray:
    return np.interp(t_eval, t_src, h_src.real) + 1j * np.interp(
        t_eval, t_src, h_src.imag
    )


def load_waveforms():
    t_bhpt, h_bhpt = bhptsur.generate_surrogate(
        q=Q, calibrated=False, modes=[MODE], neg_modes=False
    )
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    t_nr, h_nr, _ = nrsur(Q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)
    nr_mask = t_nr >= NR_T_START
    return t_bhpt, h_bhpt[MODE], t_nr[nr_mask], h_nr[MODE][nr_mask]


def physical_metadata(
    t_bhpt: np.ndarray, h_bhpt: np.ndarray, t_nr: np.ndarray, h_nr: np.ndarray
) -> dict:
    return {
        "nu": Q / (1.0 + Q) ** 2,
        "t_bhpt_merger": float(t_bhpt[np.argmax(np.abs(h_bhpt))]),
        "t_nr_merger": float(t_nr[np.argmax(np.abs(h_nr))]),
    }


def model_arrays(
    params: np.ndarray,
    t_source: np.ndarray,
    h_source: np.ndarray,
    meta: dict,
):
    alpha0, alpha1, beta0, beta1, delta_tau, phi0 = params
    dt_bhpt = t_source - meta["t_bhpt_merger"]
    theta = meta["nu"] * dt_bhpt
    alpha = alpha0 + alpha1 * theta
    beta = beta0 + beta1 * theta
    tau = (
        meta["t_nr_merger"]
        + delta_tau
        + beta0 * dt_bhpt
        + 0.5 * beta1 * meta["nu"] * dt_bhpt**2
    )
    h_scaled = alpha * np.exp(1j * phi0) * h_source
    return tau, h_scaled, alpha, beta


def evaluate(
    params: np.ndarray,
    t_source: np.ndarray,
    h_source: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    meta: dict,
) -> dict:
    tau, h_scaled, alpha, beta = model_arrays(params, t_source, h_source, meta)
    if np.any(alpha <= 0.0) or np.any(beta <= 0.0) or np.any(np.diff(tau) <= 0.0):
        return {"error": 99.0}

    common = (t_nr >= tau[0]) & (t_nr <= tau[-1])
    if np.count_nonzero(common) < 1000:
        return {"error": 99.0}

    h_model = interp_complex(tau, h_scaled, t_nr[common])
    return {
        "error": mathcalE_error(h_nr[common], h_model),
        "common": common,
        "h_model": h_model,
        "tau": tau,
        "alpha": alpha,
        "beta": beta,
    }


def phase_seed(params: np.ndarray, t_source, h_source, t_nr, h_nr, meta) -> np.ndarray:
    seeded = params.copy()
    seeded[5] = 0.0
    ev = evaluate(seeded, t_source, h_source, t_nr, h_nr, meta)
    if ev["error"] >= 90.0:
        return params
    tau, h_scaled, _alpha, _beta = model_arrays(seeded, t_source, h_source, meta)
    common = ev["common"]
    h_model = interp_complex(tau, h_scaled, t_nr[common])
    seeded[5] = float(np.angle(np.sum(h_nr[common] * h_model.conjugate())))
    return seeded


def initial_guesses() -> list[np.ndarray]:
    mass_norm = 1.0 / (1.0 + 1.0 / Q)
    return [
        np.array(
            [
                0.798129102,
                -2.04021032e-5,
                0.802847356,
                -1.85292060e-6,
                3.39584934,
                1.03012031,
            ],
            dtype=float,
        ),
        np.array([0.80, -2.0e-5, 0.80, -2.0e-6, 0.0, 0.0], dtype=float),
        np.array([0.80, 0.0, 0.80, 0.0, 0.0, 0.0], dtype=float),
        np.array([mass_norm, 0.0, mass_norm, 0.0, 0.0, 0.0], dtype=float),
        np.array([mass_norm, -2.0e-5, mass_norm, -2.0e-6, 0.0, 0.0], dtype=float),
    ]


def fit_real_scaling(t_source, h_source, t_nr, h_nr, meta) -> tuple[np.ndarray, dict]:
    opt_idx = np.unique(np.r_[np.arange(0, len(t_nr), 3), len(t_nr) - 1])
    t_opt = t_nr[opt_idx]
    h_opt = h_nr[opt_idx]

    def objective(params: np.ndarray) -> float:
        return evaluate(params, t_source, h_source, t_opt, h_opt, meta)["error"]

    best_params = None
    best_error = 99.0
    for guess in initial_guesses():
        start = phase_seed(guess, t_source, h_source, t_opt, h_opt, meta)
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 250, "xatol": 1e-9, "fatol": 1e-11},
        )
        if float(result.fun) < best_error:
            best_error = float(result.fun)
            best_params = result.x
        trial = evaluate(result.x, t_source, h_source, t_nr, h_nr, meta)
        if trial["error"] < TARGET:
            return result.x, trial

    final = evaluate(best_params, t_source, h_source, t_nr, h_nr, meta)
    return best_params, final


def write_markdown(params: np.ndarray, final: dict, t_nr: np.ndarray, meta: dict) -> None:
    common_t = t_nr[final["common"]]
    alpha0, alpha1, beta0, beta1, delta_tau, phi0 = params
    if final["error"] >= TARGET:
        raise RuntimeError(f"physical real fit error {final['error']:.15g} exceeds {TARGET:g}")

    lines = [
        "Using a physically reparameterized real low-order time-dependent fit for",
        "`q=5`, `(2,2)`, I get:",
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
        "```",
        "",
        "The dimensionless physical time variable is merger-centered and scaled by the",
        "symmetric mass ratio:",
        "",
        "```python",
        "theta = nu * (t_bhpt - t_bhpt_merger)",
        "```",
        "",
        "Amplitude scaling:",
        "",
        "```python",
        f"alpha(theta) = {alpha0:.15g} + ({alpha1:.15g}) * theta",
        "```",
        "",
        "Real time-scaling ansatz:",
        "",
        "```python",
        f"beta0 = {beta0:.15g}",
        f"beta1 = {beta1:.15g}",
        "beta(theta) = beta0 + beta1 * theta",
        "```",
        "",
        "The real time map is written relative to the two merger times:",
        "",
        "```python",
        f"Delta_tau = {delta_tau:.15g}",
        "dt_bhpt = t_bhpt - t_bhpt_merger",
        "tau(t_bhpt) = (",
        "    t_nr_merger",
        "    + Delta_tau",
        "    + beta0 * dt_bhpt",
        "    + 0.5 * beta1 * nu * dt_bhpt**2",
        ")",
        "```",
        "",
        "This satisfies:",
        "",
        "```python",
        "d tau / d t_bhpt = beta(theta)",
        "```",
        "",
        "I also included a constant phase rotation:",
        "",
        "```python",
        f"phi0 = {phi0:.15g}  # radians",
        "```",
        "",
        "Applied as:",
        "",
        "```python",
        "h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)",
        "```",
        "",
        "With common-support interpolation, this gave:",
        "",
        "```text",
        f"mathcalE = {final['error']:.15g}",
        "```",
        "",
        f"on NR support approximately `[{common_t[0]:.12g}, {common_t[-1]:.12g}]`.",
        "",
        "The previous arbitrary `t_min`/`t_max` normalization is not used in this",
        "final formula. The only time origin information entering the formula is",
        "through the physical merger times `t_bhpt_merger` and `t_nr_merger`. The",
        "fitted `Delta_tau` is a merger-relative alignment offset, not an absolute",
        "simulation start/end time.",
        "",
    ]
    (ROOT / "scaling_rerun_physical.md").write_text("\n".join(lines))


def main() -> None:
    t_source, h_source, t_nr, h_nr = load_waveforms()
    meta = physical_metadata(t_source, h_source, t_nr, h_nr)
    params, final = fit_real_scaling(t_source, h_source, t_nr, h_nr, meta)
    write_markdown(params, final, t_nr, meta)
    print(f"mathcalE={final['error']:.15g}")
    print("params=" + np.array2string(params, precision=15))


if __name__ == "__main__":
    main()
