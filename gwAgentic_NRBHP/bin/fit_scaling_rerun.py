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
T_START = -5000.1
T_MIN = -6188.39999991155
T_MAX = 114.800000111376
T_LEN = T_MAX - T_MIN
TARGET = 1e-5


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


def x_of_t(t: np.ndarray) -> np.ndarray:
    return 2.0 * (t - T_MIN) / T_LEN - 1.0


def tau_basis(t: np.ndarray) -> np.ndarray:
    u = t - T_MIN
    return u * u / T_LEN - u


def load_waveforms():
    t_bhpt, h_bhpt = bhptsur.generate_surrogate(
        q=Q, calibrated=False, modes=[MODE], neg_modes=False
    )
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    t_nr, h_nr, _ = nrsur(Q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)
    source_mask = (t_bhpt >= T_MIN) & (t_bhpt <= T_MAX)
    nr_mask = t_nr >= T_START
    return (
        t_bhpt[source_mask],
        h_bhpt[MODE][source_mask],
        t_nr[nr_mask],
        h_nr[MODE][nr_mask],
    )


def model_arrays(params: np.ndarray, t_source: np.ndarray, h_source: np.ndarray):
    alpha0, alpha1, beta0, beta1, tau0, phi0 = params
    x = x_of_t(t_source)
    u = t_source - T_MIN
    alpha = alpha0 + alpha1 * x
    beta = beta0 + beta1 * x
    tau = tau0 + beta0 * u + beta1 * tau_basis(t_source)
    h_scaled = alpha * np.exp(1j * phi0) * h_source
    return tau, h_scaled, alpha, beta


def evaluate(
    params: np.ndarray,
    t_source: np.ndarray,
    h_source: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
) -> dict:
    tau, h_scaled, alpha, beta = model_arrays(params, t_source, h_source)
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


def phase_seed(params: np.ndarray, t_source, h_source, t_nr, h_nr) -> np.ndarray:
    seeded = params.copy()
    seeded[5] = 0.0
    ev = evaluate(seeded, t_source, h_source, t_nr, h_nr)
    if ev["error"] >= 90.0:
        return params
    tau, h_scaled, _alpha, _beta = model_arrays(seeded, t_source, h_source)
    common = ev["common"]
    h_model = interp_complex(tau, h_scaled, t_nr[common])
    seeded[5] = float(np.angle(np.sum(h_nr[common] * h_model.conjugate())))
    return seeded


def initial_guesses() -> list[np.ndarray]:
    mass_norm = 1.0 / (1.0 + 1.0 / Q)
    guesses = []
    for alpha0 in (0.80, mass_norm):
        for alpha1 in (-0.02, 0.0):
            for beta0 in (0.80, mass_norm):
                for beta1 in (-0.01, 0.0):
                    for tau_shift in (0.0, 40.0):
                        guesses.append(
                            np.array(
                                [
                                    alpha0,
                                    alpha1,
                                    beta0,
                                    beta1,
                                    T_START + tau_shift,
                                    0.0,
                                ],
                                dtype=float,
                            )
                        )
    return guesses


def fit_real_scaling(t_source, h_source, t_nr, h_nr) -> tuple[np.ndarray, dict]:
    opt_idx = np.unique(np.r_[np.arange(0, len(t_nr), 3), len(t_nr) - 1])
    t_opt = t_nr[opt_idx]
    h_opt = h_nr[opt_idx]

    def objective(params: np.ndarray) -> float:
        return evaluate(params, t_source, h_source, t_opt, h_opt)["error"]

    best_params = None
    best_error = 99.0
    for guess in initial_guesses():
        start = phase_seed(guess, t_source, h_source, t_opt, h_opt)
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 800, "xatol": 1e-9, "fatol": 1e-11},
        )
        if float(result.fun) < best_error:
            best_error = float(result.fun)
            best_params = result.x

    final = evaluate(best_params, t_source, h_source, t_nr, h_nr)
    return best_params, final


def write_markdown(params: np.ndarray, final: dict, t_nr: np.ndarray) -> None:
    common_t = t_nr[final["common"]]
    alpha0, alpha1, beta0, beta1, tau0, phi0 = params
    target_status = "reached" if final["error"] < TARGET else "not reached"
    lines = [
        "Using a fresh rerun of the real low-order time-dependent fit for `q=5`, `(2,2)`, I get:",
        "",
        "```python",
        "x = 2 * (t - t_min) / (t_max - t_min) - 1",
        f"t_min = {T_MIN:.15g}",
        f"t_max = {T_MAX:.15g}",
        "```",
        "",
        "Amplitude scaling:",
        "",
        "```python",
        f"alpha(t) = {alpha0:.15g} + ({alpha1:.15g}) * x",
        "```",
        "",
        "Real time-scaling ansatz:",
        "",
        "```python",
        f"beta0 = {beta0:.15g}",
        f"beta1 = {beta1:.15g}",
        "beta(t) = beta0 + beta1 * x",
        "```",
        "",
        "The real time map is the integral of `beta(t) = d tau / dt`:",
        "",
        "```python",
        f"tau0 = {tau0:.15g}",
        "u = t - t_min",
        "tau(t) = tau0 + beta0 * u + beta1 * (u**2 / (t_max - t_min) - u)",
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
        "h_model(tau(t)) = alpha(t) * exp(1j * phi0) * h_BHPT(t)",
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
        f"The requested `mathcalE < {TARGET:g}` target was {target_status}. This is the best result found in this rerun with real linear `alpha(t)` and real linear `beta(t)`, using generic starts rather than previous fitted coefficients.",
        "",
    ]
    (ROOT / "scaling_rerun.md").write_text("\n".join(lines))


def main() -> None:
    t_source, h_source, t_nr, h_nr = load_waveforms()
    params, final = fit_real_scaling(t_source, h_source, t_nr, h_nr)
    write_markdown(params, final, t_nr)
    print(f"mathcalE={final['error']:.15g}")
    print("params=" + np.array2string(params, precision=15))


if __name__ == "__main__":
    main()
