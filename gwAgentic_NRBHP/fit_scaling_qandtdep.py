from __future__ import annotations

import sys
import warnings
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


MODE = (2, 2)
Q_VALUES = np.linspace(3.0, 8.0, 40)
T_START = -5000.1
S_MIN = -6188.39999991155
S_MAX = 114.800000111376
S_LEN = S_MAX - S_MIN
MASTER_DEGREE = 4

# q-dependent constant-scaling fit from scaling_qdep.md, used only as a seed.
ALPHA_CONST_QDEP = np.array(
    [-1.34005326034, 2.88788867946, -6.79063043803, 7.02639406495]
)
BETA_CONST_QDEP = np.array(
    [-1.24116450042, 1.63823509893, -1.97154177912, 1.33749010735]
)

Q5_SCALING_MD_SEED = np.array(
    [
        0.777159886604501,
        0.0383836831997672,
        0.807589809897401,
        0.0043352703577125,
        -4998.577093277285,
        1.97530886542818,
    ],
    dtype=float,
)


def mathcalE_error(h_ref: np.ndarray, h_model: np.ndarray) -> float:
    n_ref = np.sum(np.abs(h_ref) ** 2)
    n_model = np.sum(np.abs(h_model) ** 2)
    overlap = np.real(np.sum(h_ref * h_model.conjugate()))
    return float(((n_ref + n_model) - 2.0 * overlap) / (2.0 * n_ref))


def interp_complex(
    t_src: np.ndarray, h_src: np.ndarray, t_eval: np.ndarray
) -> np.ndarray:
    return np.interp(t_eval, t_src, h_src.real) + 1j * np.interp(
        t_eval, t_src, h_src.imag
    )


def const_qdep(q: float, coeffs: np.ndarray) -> float:
    return float(
        1.0
        + coeffs[0] / q
        + coeffs[1] / q**2
        + coeffs[2] / q**3
        + coeffs[3] / q**4
    )


def x_of_s(source_t: np.ndarray) -> np.ndarray:
    return 2.0 * (source_t - S_MIN) / S_LEN - 1.0


def tau_basis(source_t: np.ndarray) -> np.ndarray:
    u = source_t - S_MIN
    return u * u / S_LEN - u


def source_model(
    params: np.ndarray, source_t: np.ndarray, h_bhpt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a0, a1, b0, b1, tau0, phi0 = params
    x = x_of_s(source_t)
    u = source_t - S_MIN
    alpha = a0 + a1 * x
    beta = b0 + b1 * x
    tau = tau0 + b0 * u + b1 * tau_basis(source_t)
    h_scaled = alpha * np.exp(1j * phi0) * h_bhpt
    return tau, h_scaled, alpha, beta


def load_case(nrsur, q: float) -> dict:
    t_bhpt, h_bhpt = bhptsur.generate_surrogate(
        q=q, calibrated=False, modes=[MODE], neg_modes=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_nr, h_nr, _ = nrsur(q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)
    nr_mask = t_nr >= T_START
    opt_nr_idx = np.unique(np.r_[np.arange(0, np.count_nonzero(nr_mask), 8), np.count_nonzero(nr_mask) - 1])
    return {
        "q": q,
        "t_bhpt": t_bhpt,
        "h_bhpt": h_bhpt[MODE],
        "t_source_opt": t_bhpt[::4],
        "h_source_opt": h_bhpt[MODE][::4],
        "t_nr": t_nr[nr_mask],
        "h_nr": h_nr[MODE][nr_mask],
        "t_nr_opt": t_nr[nr_mask][opt_nr_idx],
        "h_nr_opt": h_nr[MODE][nr_mask][opt_nr_idx],
    }


def evaluate(
    params: np.ndarray,
    source_t: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    min_samples: int = 1000,
) -> dict:
    tau, h_scaled, alpha, beta = source_model(params, source_t, h_bhpt)
    if np.any(beta <= 0.0) or np.any(np.diff(tau) <= 0.0):
        return {"error": 99.0}

    common = (t_nr >= tau[0]) & (t_nr <= tau[-1])
    if np.count_nonzero(common) < min_samples:
        return {"error": 99.0}

    source_lo = float(np.interp(t_nr[common][0], tau, source_t))
    source_hi = float(np.interp(t_nr[common][-1], tau, source_t))
    used = (source_t >= source_lo) & (source_t <= source_hi)
    if np.any(alpha[used] <= 0.0):
        return {"error": 99.0}

    h_model = interp_complex(tau, h_scaled, t_nr[common])
    return {
        "error": mathcalE_error(h_nr[common], h_model),
        "common": common,
        "source_lo": source_lo,
        "source_hi": source_hi,
        "h_model": h_model,
    }


def phase_seed(params: np.ndarray, case: dict) -> np.ndarray:
    seeded = params.copy()
    no_phase = seeded.copy()
    no_phase[5] = 0.0
    ev = evaluate(
        no_phase,
        case["t_source_opt"],
        case["h_source_opt"],
        case["t_nr_opt"],
        case["h_nr_opt"],
    )
    if ev["error"] >= 90.0:
        return seeded
    tau, h_scaled, _alpha, _beta = source_model(
        no_phase, case["t_source_opt"], case["h_source_opt"]
    )
    common = ev["common"]
    h_model = interp_complex(tau, h_scaled, case["t_nr_opt"][common])
    seeded[5] = float(np.angle(np.sum(case["h_nr_opt"][common] * h_model.conjugate())))
    return seeded


def seed_params(q: float, shift: float = 0.0) -> np.ndarray:
    alpha = const_qdep(q, ALPHA_CONST_QDEP)
    beta = const_qdep(q, BETA_CONST_QDEP)
    return np.array([alpha, 0.0, beta, 0.0, beta * S_MIN + shift, 0.0], dtype=float)


def optimize_case(case: dict, previous: np.ndarray | None) -> dict:
    starts = [seed_params(case["q"], shift) for shift in (0.0, 3.0, -3.0)]
    if previous is not None:
        starts.insert(0, previous.copy())
    if abs(case["q"] - 5.0) < 1.0:
        starts.append(Q5_SCALING_MD_SEED.copy())

    starts = [phase_seed(start, case) for start in starts]

    def objective(params: np.ndarray) -> float:
        return evaluate(
            params,
            case["t_source_opt"],
            case["h_source_opt"],
            case["t_nr_opt"],
            case["h_nr_opt"],
        )["error"]

    best = {"error": 99.0, "params": starts[0]}
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 550, "xatol": 1e-8, "fatol": 1e-10},
        )
        if float(result.fun) < best["error"]:
            best = {"error": float(result.fun), "params": result.x}

    full = evaluate(
        best["params"],
        case["t_bhpt"],
        case["h_bhpt"],
        case["t_nr"],
        case["h_nr"],
    )
    return {
        "q": case["q"],
        "params": best["params"],
        "fit_error": best["error"],
        "error": full["error"],
        "source_lo": full["source_lo"],
        "source_hi": full["source_hi"],
        "support_lo": float(case["t_nr"][full["common"]][0]),
        "support_hi": float(case["t_nr"][full["common"]][-1]),
    }


def master_design(q_values: np.ndarray) -> np.ndarray:
    y = 1.0 / q_values
    return np.vstack([y**power for power in range(MASTER_DEGREE + 1)]).T


def fit_master(q_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(master_design(q_values), values, rcond=None)[0]


def eval_master(q: float, coeffs: np.ndarray) -> float:
    y = 1.0 / q
    return float(sum(coeffs[i] * y**i for i in range(len(coeffs))))


def optimize_tau0_phi(case: dict, coeffs_by_name: dict[str, np.ndarray]) -> dict:
    q = case["q"]
    base = np.array(
        [
            eval_master(q, coeffs_by_name["A_alpha"]),
            eval_master(q, coeffs_by_name["B_alpha"]),
            eval_master(q, coeffs_by_name["A_beta"]),
            eval_master(q, coeffs_by_name["B_beta"]),
            eval_master(q, coeffs_by_name["A_beta"]) * S_MIN,
            0.0,
        ],
        dtype=float,
    )
    base = phase_seed(base, case)

    def objective(offset_phase: np.ndarray) -> float:
        params = base.copy()
        params[4] = offset_phase[0]
        params[5] = offset_phase[1]
        return evaluate(
            params,
            case["t_source_opt"],
            case["h_source_opt"],
            case["t_nr_opt"],
            case["h_nr_opt"],
        )["error"]

    starts = [base[[4, 5]], np.array([base[4] + 3.0, base[5]]), np.array([base[4] - 3.0, base[5]])]
    best_fun = 99.0
    best = starts[0]
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 250, "xatol": 1e-8, "fatol": 1e-10},
        )
        if float(result.fun) < best_fun:
            best_fun = float(result.fun)
            best = result.x

    params = base.copy()
    params[4] = best[0]
    params[5] = best[1]
    full = evaluate(
        params,
        case["t_bhpt"],
        case["h_bhpt"],
        case["t_nr"],
        case["h_nr"],
    )
    return {
        "q": q,
        "params": params,
        "fit_error": best_fun,
        "error": full["error"],
        "source_lo": full["source_lo"],
        "source_hi": full["source_hi"],
        "support_lo": float(case["t_nr"][full["common"]][0]),
        "support_hi": float(case["t_nr"][full["common"]][-1]),
    }


def format_poly(name: str, coeffs: np.ndarray) -> list[str]:
    terms = [f"{coeffs[0]:.12g}"]
    for power in range(1, len(coeffs)):
        terms.append(f"({coeffs[power]:.12g}) / q**{power}")
    return [f"{name}(q) = " + " + ".join(terms)]


def write_markdown(per_q: list[dict], master: dict[str, np.ndarray], final: list[dict]) -> None:
    per_q_arr = np.array([[row["q"], *row["params"][:4], row["error"]] for row in per_q])
    final_errors = np.array([row["error"] for row in final])
    per_q_errors = np.array([row["error"] for row in per_q])
    max_idx = int(np.argmax(final_errors))

    lines = [
        "# q- and time-dependent scaling fit",
        "",
        "Fit form for the raw BHPT-to-NRHybSur3dq8 `(2,2)` comparison:",
        "",
        "```python",
        f"x = 2 * (t - t_min) / (t_max - t_min) - 1",
        f"t_min = {S_MIN:.15g}",
        f"t_max = {S_MAX:.15g}",
        "",
        "alpha(q, t) = A_alpha(q) + B_alpha(q) * x",
        "beta(q, t) = A_beta(q) + B_beta(q) * x",
        "```",
        "",
        "The time map is the integral of `beta(q,t)` over raw BHPT source time:",
        "",
        "```python",
        "u = t - t_min",
        "tau(q, t) = tau0(q) + A_beta(q) * u + B_beta(q) * (u**2 / (t_max - t_min) - u)",
        "h_model(tau(q,t)) = alpha(q,t) * exp(1j * phi0(q)) * h_BHPT(t)",
        "```",
        "",
        f"The per-q fit used {len(Q_VALUES)} uniformly spaced q values from 3 to 8. BHPT waveforms used `calibrated=False`; NR waveforms used `NRHybSur3dq8`, nonspinning, `dt=0.1`, `f_low=5e-3`, and `t_start=-5000.1`. The error is the `mathcalE_error` definition in `NRBHP_ansatz.py`, with NR as the reference waveform.",
        "",
        "For the master q-fit, each coefficient is represented as a quartic polynomial in `1/q`:",
        "",
        "```python",
        "C(q) = c0 + c1/q + c2/q**2 + c3/q**3 + c4/q**4",
        "```",
        "",
        "## master coefficients",
        "",
        "| coefficient | c0 | c1 | c2 | c3 | c4 |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for name in ["A_alpha", "B_alpha", "A_beta", "B_beta"]:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(f"{value:.12g}" for value in master[name])
            + " |"
        )

    lines.extend(
        [
            "",
            "Equivalent Python definitions:",
            "",
            "```python",
        ]
    )
    for name in ["A_alpha", "B_alpha", "A_beta", "B_beta"]:
        lines.extend(format_poly(name, master[name]))
    lines.extend(
        [
            "```",
            "",
            "## error summary",
            "",
            "| model | min mathcalE | median mathcalE | max mathcalE | q at max |",
            "|:---|---:|---:|---:|---:|",
            f"| independent per-q coefficients | {per_q_errors.min():.6g} | {np.median(per_q_errors):.6g} | {per_q_errors.max():.6g} | {per_q[int(np.argmax(per_q_errors))]['q']:.6g} |",
            f"| master q-fit coefficients, optimized tau0/phi0 | {final_errors.min():.6g} | {np.median(final_errors):.6g} | {final_errors.max():.6g} | {final[max_idx]['q']:.6g} |",
            "",
            "The master-fit error row uses only the q-dependent `A/B` coefficient formulae above, while re-optimizing the nuisance integration constant `tau0(q)` and phase rotation `phi0(q)` for each q. These nuisance quantities are not included in the master coefficient ansatz.",
            "",
            "## per-q fitted coefficients",
            "",
            "| q | A_alpha | B_alpha | A_beta | B_beta | per-q mathcalE | master mathcalE | tau0 master | phi0 master |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    final_by_q = {row["q"]: row for row in final}
    for row in per_q:
        q = row["q"]
        params = row["params"]
        final_row = final_by_q[q]
        final_params = final_row["params"]
        lines.append(
            f"| {q:.8g} | {params[0]:.10g} | {params[1]:.10g} | {params[2]:.10g} | {params[3]:.10g} | {row['error']:.6g} | {final_row['error']:.6g} | {final_params[4]:.10g} | {final_params[5]:.10g} |"
        )

    lines.extend(
        [
            "",
            "No extrapolated BHPT samples are used in the error calculation; each row masks the NR waveform to the common support of the transformed BHPT time array before interpolation.",
            "",
        ]
    )
    (ROOT / "scaling_qandtdep.md").write_text("\n".join(lines))


def main() -> None:
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    cases = [load_case(nrsur, float(q)) for q in Q_VALUES]

    per_q = []
    previous = None
    for case in cases:
        row = optimize_case(case, previous)
        per_q.append(row)
        previous = row["params"]
        print(
            f"q={case['q']:.6g} per-q mathcalE={row['error']:.6g} "
            f"A_alpha={row['params'][0]:.6g} B_alpha={row['params'][1]:.6g} "
            f"A_beta={row['params'][2]:.6g} B_beta={row['params'][3]:.6g}",
            flush=True,
        )

    q_values = np.array([row["q"] for row in per_q])
    params = np.array([row["params"][:4] for row in per_q])
    master = {
        "A_alpha": fit_master(q_values, params[:, 0]),
        "B_alpha": fit_master(q_values, params[:, 1]),
        "A_beta": fit_master(q_values, params[:, 2]),
        "B_beta": fit_master(q_values, params[:, 3]),
    }

    final = []
    for case in cases:
        row = optimize_tau0_phi(case, master)
        final.append(row)
        print(
            f"q={case['q']:.6g} master mathcalE={row['error']:.6g}",
            flush=True,
        )

    write_markdown(per_q, master, final)


if __name__ == "__main__":
    main()
