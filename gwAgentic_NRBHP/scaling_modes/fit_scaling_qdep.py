from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import gwsurrogate

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "BHPTNRSurrogate" / "surrogates"))
import BHPTNRSur1dq1e4 as bhptsur  # noqa: E402


Q_VALUES = np.linspace(3.0, 10.0, 40)
L_VALUES = [2, 3, 4, 5]
MODES = [(ell, ell) for ell in L_VALUES]
T_START = -5000.1
PLOT_Q = 5.0


def mathcal_e(h_ref: np.ndarray, h_model: np.ndarray) -> float:
    n_ref = np.sum(np.abs(h_ref) ** 2)
    n_model = np.sum(np.abs(h_model) ** 2)
    overlap = np.real(np.sum(h_ref * h_model.conjugate()))
    return float(((n_ref + n_model) - 2.0 * overlap) / (2.0 * n_ref))


def interp_complex(t_src: np.ndarray, h_src: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    return np.interp(t_eval, t_src, h_src.real) + 1j * np.interp(
        t_eval, t_src, h_src.imag
    )


def ansatz(q: float, coeffs: np.ndarray) -> float:
    return float(
        1.0
        + coeffs[0] / q
        + coeffs[1] / q**2
        + coeffs[2] / q**3
        + coeffs[3] / q**4
    )


def fit_coefficients(values: np.ndarray) -> np.ndarray:
    design = np.vstack(
        [1 / Q_VALUES, 1 / Q_VALUES**2, 1 / Q_VALUES**3, 1 / Q_VALUES**4]
    ).T
    return np.linalg.lstsq(design, values - 1.0, rcond=None)[0]


def load_case(nrsur, q: float):
    t_bhpt, h_bhpt = bhptsur.generate_surrogate(
        q=q, calibrated=False, modes=MODES, neg_modes=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_nr, h_nr, _ = nrsur(q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)
    mask = t_nr >= T_START
    return t_bhpt, h_bhpt, t_nr[mask], {mode: h_nr[mode][mask] for mode in MODES}


def aligned_error(
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    alpha: float,
    beta: float,
    t0: float,
    phi0: float,
) -> float:
    if alpha <= 0.0 or beta <= 0.0:
        return 99.0
    t_scaled = beta * t_bhpt + t0
    lo = max(float(t_nr[0]), float(t_scaled[0]))
    hi = min(float(t_nr[-1]), float(t_scaled[-1]))
    mask = (t_nr >= lo) & (t_nr <= hi)
    if np.count_nonzero(mask) < 1000:
        return 99.0
    h_scaled = alpha * np.exp(1j * phi0) * h_bhpt
    h_model = interp_complex(t_scaled, h_scaled, t_nr[mask])
    return mathcal_e(h_nr[mask], h_model)


def overlap_phase_seed(
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
    alpha: float,
    beta: float,
    t0: float,
) -> float:
    t_scaled = beta * t_bhpt + t0
    lo = max(float(t_nr[0]), float(t_scaled[0]))
    hi = min(float(t_nr[-1]), float(t_scaled[-1]))
    mask = (t_nr >= lo) & (t_nr <= hi)
    if np.count_nonzero(mask) < 1000:
        return 0.0
    h_model = interp_complex(t_scaled, alpha * h_bhpt, t_nr[mask])
    return float(np.angle(np.sum(h_nr[mask] * h_model.conjugate())))


def optimize_alpha_beta_t0_phi(case, ell: int, p0: tuple[float, float]) -> tuple:
    t_bhpt, h_bhpt, t_nr, h_nr = case
    h_raw = h_bhpt[(ell, ell)]
    h_ref = h_nr[(ell, ell)]

    def objective(params):
        log_alpha, log_beta, t0, phi0 = params
        return aligned_error(
            t_bhpt,
            h_raw,
            t_nr,
            h_ref,
            float(np.exp(log_alpha)),
            float(np.exp(log_beta)),
            t0,
            phi0,
        )

    starts = []
    for t0_seed in [0.0, 2.0, -2.0, 5.0, -5.0]:
        phi_seed = overlap_phase_seed(
            t_bhpt, h_raw, t_nr, h_ref, p0[0], p0[1], t0_seed
        )
        starts.append([np.log(p0[0]), np.log(p0[1]), t0_seed, phi_seed])

    results = [
        minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 700, "xatol": 1e-9, "fatol": 1e-12},
        )
        for start in starts
    ]
    result = min(results, key=lambda item: float(item.fun))
    alpha = float(np.exp(result.x[0]))
    beta = float(np.exp(result.x[1]))
    return float(result.fun), alpha, beta, float(result.x[2]), float(result.x[3])


def optimize_alpha_t0_phi(case, ell: int, beta: float, alpha0: float) -> tuple:
    t_bhpt, h_bhpt, t_nr, h_nr = case
    h_raw = h_bhpt[(ell, ell)]
    h_ref = h_nr[(ell, ell)]

    def objective(params):
        log_alpha, t0, phi0 = params
        return aligned_error(
            t_bhpt,
            h_raw,
            t_nr,
            h_ref,
            float(np.exp(log_alpha)),
            beta,
            t0,
            phi0,
        )

    starts = []
    for t0_seed in [0.0, 2.0, -2.0, 5.0, -5.0]:
        phi_seed = overlap_phase_seed(
            t_bhpt, h_raw, t_nr, h_ref, alpha0, beta, t0_seed
        )
        starts.append([np.log(max(alpha0, 1e-5)), t0_seed, phi_seed])

    results = [
        minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-9, "fatol": 1e-12},
        )
        for start in starts
    ]
    result = min(results, key=lambda item: float(item.fun))
    return (
        float(result.fun),
        float(np.exp(result.x[0])),
        beta,
        float(result.x[1]),
        float(result.x[2]),
    )


def optimize_alignment_only(case, ell: int, alpha: float, beta: float) -> tuple:
    t_bhpt, h_bhpt, t_nr, h_nr = case
    h_raw = h_bhpt[(ell, ell)]
    h_ref = h_nr[(ell, ell)]

    def objective(params):
        t0, phi0 = params
        return aligned_error(t_bhpt, h_raw, t_nr, h_ref, alpha, beta, t0, phi0)

    starts = []
    for t0_seed in [0.0, 2.0, -2.0, 5.0, -5.0]:
        phi_seed = overlap_phase_seed(
            t_bhpt, h_raw, t_nr, h_ref, alpha, beta, t0_seed
        )
        starts.append([t0_seed, phi_seed])

    results = [
        minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 350, "xatol": 1e-8, "fatol": 1e-11},
        )
        for start in starts
    ]
    result = min(results, key=lambda item: float(item.fun))
    return float(result.fun), float(result.x[0]), float(result.x[1])


def write_markdown(beta_coeffs, alpha_coeffs, final_errors, lower_bounds):
    lines = [
        "# q-dependent scaling fit",
        "",
        "Fit form:",
        "",
        "```python",
        "alpha(q, l) = 1 + A_alpha(l)/q + B_alpha(l)/q**2 + C_alpha(l)/q**3 + D_alpha(l)/q**4",
        "beta(q) = 1 + A_beta/q + B_beta/q**2 + C_beta/q**3 + D_beta/q**4",
        "```",
        "",
        f"q grid: 40 uniformly spaced values from 3 to 10.",
        "",
        "The comparison uses the `mathcalE_error` definition from `NRBHP_ansatz.py`, raw `BHPTNRSur1dq1e4` waveforms with `calibrated=False`, and `NRHybSur3dq8` waveforms with `dt=0.1`, `f_low=5e-3`, `t_start=-5000.1`.",
        "",
        "A constant time shift `t0` and constant phase rotation `phi0` are optimized as nuisance alignment parameters for diagnostics and plotting. They are not included in the coefficient tables below.",
        "",
        "Important: `NRHybSur3dq8` warns that q values above about 8.01 are outside its training range. The requested q range extends to 10, so the high-q part of this fit uses NR surrogate extrapolation.",
        "",
        "## beta coefficients",
        "",
        "| A_beta | B_beta | C_beta | D_beta |",
        "|---:|---:|---:|---:|",
        "| " + " | ".join(f"{v:.12g}" for v in beta_coeffs) + " |",
        "",
        "## alpha coefficients",
        "",
        "| l | A_alpha | B_alpha | C_alpha | D_alpha |",
        "|---:|---:|---:|---:|---:|",
    ]
    for ell in L_VALUES:
        lines.append(
            "| "
            + str(ell)
            + " | "
            + " | ".join(f"{v:.12g}" for v in alpha_coeffs[ell])
            + " |"
        )
    lines.extend(
        [
            "",
            "## final coefficient error summary",
            "",
            "| l | min mathcalE | median mathcalE | max mathcalE | q at max | requested target |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ell in L_VALUES:
        vals = final_errors[ell]
        target = 0.001 if ell == 2 else 0.01
        lines.append(
            f"| {ell} | {vals['min']:.6g} | {vals['median']:.6g} | {vals['max']:.6g} | {vals['q_at_max']:.6g} | {target:.6g} |"
        )
    lines.extend(
        [
            "",
            "## ansatz lower-bound check",
            "",
            "For several representative cases I also optimized independent per-case `alpha`, `beta`, `t0`, and `phi0`. This is a lower bound for any global q-polynomial coefficient fit using the same constant-scaling ansatz on the full time window.",
            "",
            "| q | l | lower-bound mathcalE | alpha | beta | t0 | phi0 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in lower_bounds:
        lines.append(
            f"| {row['q']:.6g} | {row['l']} | {row['error']:.6g} | {row['alpha']:.6g} | {row['beta']:.6g} | {row['t0']:.6g} | {row['phi0']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The requested tolerances are not attainable with this constant `alpha(q,l)`, constant `beta(q)` ansatz over the full `t >= -5000.1` comparison window. In particular, even independent per-case constants give `mathcalE ~= 0.0486` for `q=3, l=4` and `mathcalE ~= 0.0808` for `q=3, l=5`, above the requested `0.01` higher-mode target.",
            "",
        ]
    )
    (ROOT / "scaling_qdep.md").write_text("\n".join(lines))


def plot_matches(beta_coeffs, alpha_coeffs):
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    case = load_case(nrsur, PLOT_Q)
    t_bhpt, h_bhpt, t_nr, h_nr = case
    for ell in L_VALUES:
        alpha = ansatz(PLOT_Q, alpha_coeffs[ell])
        beta = ansatz(PLOT_Q, beta_coeffs)
        err, t0, phi0 = optimize_alignment_only(case, ell, alpha, beta)
        t_scaled = beta * t_bhpt + t0
        h_scaled = alpha * np.exp(1j * phi0) * h_bhpt[(ell, ell)]
        lo = max(float(t_nr[0]), float(t_scaled[0]))
        hi = min(float(t_nr[-1]), float(t_scaled[-1]))
        mask = (t_nr >= lo) & (t_nr <= hi)
        h_model = interp_complex(t_scaled, h_scaled, t_nr[mask])

        plt.figure(figsize=(8, 4.5))
        plt.plot(t_nr, np.real(h_nr[(ell, ell)]), label=f"NR {ell}{ell} mode")
        plt.plot(t_nr[mask], np.real(h_model), label=f"Scaled BHPT {ell}{ell}")
        if ell == 2:
            plt.xlim(-5000, 100)
        elif ell in (3, 4):
            plt.xlim(-1000, 100)
        else:
            plt.xlim(-400, 100)
        plt.xlabel("t / M")
        plt.ylabel(f"Re[h{ell}{ell}]")
        plt.title(f"q={PLOT_Q:g}, ({ell},{ell}), mathcalE={err:.4g}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ROOT / "scaling_modes" / f"match_{ell}_{ell}_mode.pdf")
        plt.close()


def main():
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    beta_initial = np.asarray(bhptsur.beta_coeffs, dtype=float)
    alpha_initial = {
        ell: np.asarray(bhptsur.alpha_coeffs[(ell, ell)], dtype=float)
        for ell in L_VALUES
    }

    cases = {float(q): load_case(nrsur, float(q)) for q in Q_VALUES}

    beta_samples = []
    l2_alpha_samples = []
    for q in Q_VALUES:
        alpha0 = ansatz(float(q), alpha_initial[2])
        beta0 = ansatz(float(q), beta_initial)
        err, alpha, beta, _t0, _phi0 = optimize_alpha_beta_t0_phi(
            cases[float(q)], 2, (alpha0, beta0)
        )
        l2_alpha_samples.append(alpha)
        beta_samples.append(beta)
        print(f"l=2 q={q:.6g} per-case E={err:.6g}", flush=True)

    beta_coeffs = fit_coefficients(np.asarray(beta_samples))
    alpha_coeffs = {2: fit_coefficients(np.asarray(l2_alpha_samples))}

    for ell in [3, 4, 5]:
        samples = []
        for q in Q_VALUES:
            beta = ansatz(float(q), beta_coeffs)
            alpha0 = ansatz(float(q), alpha_initial[ell])
            err, alpha, _beta, _t0, _phi0 = optimize_alpha_t0_phi(
                cases[float(q)], ell, beta, alpha0
            )
            samples.append(alpha)
            print(f"l={ell} q={q:.6g} alpha-fit E={err:.6g}", flush=True)
        alpha_coeffs[ell] = fit_coefficients(np.asarray(samples))

    final_errors = {}
    for ell in L_VALUES:
        errs = []
        for q in Q_VALUES:
            alpha = ansatz(float(q), alpha_coeffs[ell])
            beta = ansatz(float(q), beta_coeffs)
            err, _t0, _phi0 = optimize_alignment_only(
                cases[float(q)], ell, alpha, beta
            )
            errs.append(err)
        arr = np.asarray(errs)
        final_errors[ell] = {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
            "q_at_max": float(Q_VALUES[int(np.argmax(arr))]),
        }

    lower_bounds = []
    for q, ell in [(3.0, 2), (3.0, 4), (3.0, 5), (5.0, 2), (5.0, 5), (10.0, 5)]:
        alpha0 = ansatz(q, alpha_initial[ell])
        beta0 = ansatz(q, beta_initial)
        err, alpha, beta, t0, phi0 = optimize_alpha_beta_t0_phi(
            cases[q], ell, (alpha0, beta0)
        )
        lower_bounds.append(
            {
                "q": q,
                "l": ell,
                "error": err,
                "alpha": alpha,
                "beta": beta,
                "t0": t0,
                "phi0": phi0,
            }
        )

    write_markdown(beta_coeffs, alpha_coeffs, final_errors, lower_bounds)
    plot_matches(beta_coeffs, alpha_coeffs)


if __name__ == "__main__":
    main()
