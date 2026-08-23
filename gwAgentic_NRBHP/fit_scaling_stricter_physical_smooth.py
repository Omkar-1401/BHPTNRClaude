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
TARGET = 1.5e-4
MIN_TRANSITION_WIDTH = 2.0
MAX_TRANSITION_WIDTH = 30.0


def mathcalE_error(h_ref: np.ndarray, h_model: np.ndarray) -> float:
    return float(
        np.sum(np.abs(h_ref - h_model) ** 2)
        / (2.0 * np.sum(np.abs(h_ref) ** 2))
    )


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
        theta_transition_width,
        beta_left,
        beta_right_edge,
        beta_right_slope,
        alpha_left,
        alpha_right_edge,
        alpha_right_slope,
        phi0,
    ) = [float(v) for v in params]
    return {
        "theta_cut": theta_cut,
        "t_cut_nr": t_cut_nr,
        "theta_transition_width": theta_transition_width,
        "beta_left": beta_left,
        "beta_right_edge": beta_right_edge,
        "beta_right_slope": beta_right_slope,
        "alpha_left": alpha_left,
        "alpha_right_edge": alpha_right_edge,
        "alpha_right_slope": alpha_right_slope,
        "phi0": phi0,
    }


def smooth_constant_to_line(
    theta: np.ndarray,
    theta_cut: float,
    width: float,
    left_value: float,
    right_edge_value: float,
    right_slope: float,
) -> np.ndarray:
    dtheta = theta - theta_cut
    out = np.empty_like(theta, dtype=float)

    left = dtheta <= 0.0
    right = dtheta >= width
    middle = ~(left | right)

    out[left] = left_value
    out[right] = right_edge_value + right_slope * (dtheta[right] - width)

    if np.any(middle):
        z = dtheta[middle] / width
        h00 = 2.0 * z**3 - 3.0 * z**2 + 1.0
        h01 = -2.0 * z**3 + 3.0 * z**2
        h11 = z**3 - z**2
        out[middle] = (
            h00 * left_value
            + h01 * right_edge_value
            + h11 * width * right_slope
        )

    return out


def cumulative_trapezoid_integral(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    return np.r_[0.0, np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)]


def model_arrays(
    params: np.ndarray, t_bhpt: np.ndarray, h_bhpt: np.ndarray, meta: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = unpack(params)
    nu = meta["nu"]
    theta = nu * (t_bhpt - meta["t_bhpt_merger"])

    beta = smooth_constant_to_line(
        theta,
        p["theta_cut"],
        p["theta_transition_width"],
        p["beta_left"],
        p["beta_right_edge"],
        p["beta_right_slope"],
    )
    alpha = smooth_constant_to_line(
        theta,
        p["theta_cut"],
        p["theta_transition_width"],
        p["alpha_left"],
        p["alpha_right_edge"],
        p["alpha_right_slope"],
    )

    s_cut = meta["t_bhpt_merger"] + p["theta_cut"] / nu
    beta_integral = cumulative_trapezoid_integral(t_bhpt, beta)
    beta_integral_cut = float(np.interp(s_cut, t_bhpt, beta_integral))
    tau = p["t_cut_nr"] + beta_integral - beta_integral_cut

    h_scaled = alpha * np.exp(1j * p["phi0"]) * h_bhpt
    return tau, h_scaled, alpha, beta, theta


def validate_basic(params: np.ndarray, meta: dict) -> bool:
    p = unpack(params)
    s_cut = meta["t_bhpt_merger"] + p["theta_cut"] / meta["nu"]
    theta_end = p["theta_cut"] + p["theta_transition_width"]
    if not (-120.0 <= p["theta_cut"] <= -10.0):
        return False
    if not (-800.0 <= p["t_cut_nr"] <= -50.0):
        return False
    if not (-900.0 <= s_cut <= -70.0):
        return False
    if not (MIN_TRANSITION_WIDTH <= p["theta_transition_width"] <= MAX_TRANSITION_WIDTH):
        return False
    if not (-80.0 <= theta_end <= 25.0):
        return False
    if not (0.25 <= p["beta_left"] <= 1.5):
        return False
    if not (0.25 <= p["beta_right_edge"] <= 1.5):
        return False
    if not (-1.0e-2 <= p["beta_right_slope"] <= 1.0e-2):
        return False
    if not (0.05 <= p["alpha_left"] <= 2.5):
        return False
    if not (0.05 <= p["alpha_right_edge"] <= 2.5):
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
    seeded[9] = 0.0
    ev = evaluate_model(seeded, t_bhpt, h_bhpt, t_nr, h_nr, meta, min_coverage=0.90)
    if ev["error"] >= 20.0:
        return params
    h_model = ev["h_model"]
    seeded[9] = principal_phase(
        float(np.angle(np.sum(ev["h_ref"] * h_model.conjugate())))
    )
    return seeded


def previous_smooth_reduced_start(width: float | None = None) -> np.ndarray:
    transition_width = 23.999984390868 if width is None else width
    theta_cut = -38.915741728485
    t_cut_nr = -226.486911272952
    theta_end = theta_cut + transition_width

    beta_right_slope = 0.00186223006453996
    alpha_right_slope = -0.00605617268087102
    beta_right_edge = 0.789790272925979 + beta_right_slope * (
        theta_end - -26.915749533051
    )
    alpha_right_edge = 0.889358733860003 + alpha_right_slope * (
        theta_end - -26.915749533051
    )

    return np.array(
        [
            theta_cut,
            t_cut_nr,
            transition_width,
            0.805438644214298,
            beta_right_edge,
            beta_right_slope,
            0.807796140498049,
            alpha_right_edge,
            alpha_right_slope,
            1.43573268651453,
        ],
        dtype=float,
    )


def generic_starts() -> list[np.ndarray]:
    starts = [previous_smooth_reduced_start()]
    for width in (8.0, 12.0, 16.0, 20.0, 24.0):
        starts.append(previous_smooth_reduced_start(width))

    for theta_cut, t_cut_nr in (
        (-65.0, -420.0),
        (-50.0, -310.0),
        (-40.0, -230.0),
        (-32.0, -190.0),
        (-28.0, -170.0),
        (-22.0, -140.0),
    ):
        for width in (6.0, 12.0, 20.0, 28.0):
            for beta_right_slope in (5.0e-4, 1.0e-3, 2.0e-3, 3.0e-3):
                starts.append(
                    np.array(
                        [
                            theta_cut,
                            t_cut_nr,
                            width,
                            0.805,
                            0.812,
                            beta_right_slope,
                            0.808,
                            0.82,
                            -4.0e-3,
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
    for _start_error, seeded in ranked_starts[:10]:
        result = minimize(
            bounded_objective,
            seeded,
            args=(t_bhpt, h_bhpt, t_fit, h_fit, meta),
            method="Nelder-Mead",
            options={"maxiter": 1800, "xatol": 1e-8, "fatol": 1e-10, "disp": False},
        )
        if float(result.fun) < best_error:
            best_error = float(result.fun)
            best_params = result.x

    full_result = minimize(
        bounded_objective,
        best_params,
        args=(t_bhpt, h_bhpt, t_nr, h_nr, meta),
        method="Nelder-Mead",
        options={"maxiter": 900, "xatol": 1e-8, "fatol": 1e-10, "disp": False},
    )
    if float(full_result.fun) < best_error:
        best_error = float(full_result.fun)
        best_params = full_result.x

    return best_params, best_error


def scalar_model_values(params: np.ndarray, meta: dict, theta_values: np.ndarray) -> dict:
    p = unpack(params)
    alpha = smooth_constant_to_line(
        theta_values,
        p["theta_cut"],
        p["theta_transition_width"],
        p["alpha_left"],
        p["alpha_right_edge"],
        p["alpha_right_slope"],
    )
    beta = smooth_constant_to_line(
        theta_values,
        p["theta_cut"],
        p["theta_transition_width"],
        p["beta_left"],
        p["beta_right_edge"],
        p["beta_right_slope"],
    )
    t_bhpt_values = meta["t_bhpt_merger"] + theta_values / meta["nu"]
    return {"theta": theta_values, "t_bhpt": t_bhpt_values, "alpha": alpha, "beta": beta}


def local_error(h_ref: np.ndarray, h_model: np.ndarray, mask: np.ndarray) -> float:
    if np.count_nonzero(mask) == 0:
        return float("nan")
    return mathcalE_error(h_ref[mask], h_model[mask])


def write_markdown(
    params: np.ndarray,
    fit_error: float,
    eval_data: dict,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    meta: dict,
) -> None:
    if eval_data["error"] >= TARGET:
        raise RuntimeError(
            "reduced stricter physical smooth fit error "
            f"{eval_data['error']:.15g} exceeds {TARGET:g}"
        )

    p = unpack(params)
    common_t = t_nr[eval_data["common"]]
    h_ref = eval_data["h_ref"]
    h_model = eval_data["h_model"]
    theta_cut = p["theta_cut"]
    width = p["theta_transition_width"]
    theta_end = theta_cut + width
    s_cut = meta["t_bhpt_merger"] + theta_cut / meta["nu"]
    s_end = meta["t_bhpt_merger"] + theta_end / meta["nu"]
    theta_cut_nr = meta["nu"] * (p["t_cut_nr"] - meta["t_nr_merger"])

    tau_full, _h_scaled, _alpha_full, _beta_full, _theta_full = model_arrays(
        params, t_bhpt, h_bhpt, meta
    )
    transition_theta = np.array([theta_cut, theta_end], dtype=float)
    transition_t_bhpt = meta["t_bhpt_merger"] + transition_theta / meta["nu"]
    transition_t_nr = np.interp(transition_t_bhpt, t_bhpt, tau_full)
    transition_values = scalar_model_values(params, meta, transition_theta)

    before_mask = common_t <= transition_t_nr[0]
    transition_mask = (common_t > transition_t_nr[0]) & (common_t <= transition_t_nr[1])
    after_mask = common_t > transition_t_nr[1]

    before_err = local_error(h_ref, h_model, before_mask)
    transition_err = local_error(h_ref, h_model, transition_mask)
    after_err = local_error(h_ref, h_model, after_mask)

    theta_used = eval_data["theta"]
    endpoint_theta = np.array([theta_used[0], theta_used[-1]], dtype=float)
    endpoint_values = scalar_model_values(params, meta, endpoint_theta)

    lines = [
        "# Reduced stricter physical smooth q=5 (2,2) scaling fit",
        "",
        "This run replaces the previous smooth ansatz with a lower-parameter",
        "version motivated by the nearly constant pre-cutoff behavior. Before",
        "`theta_cut`, both `alpha` and `beta` are constants. After a fitted smooth",
        "transition interval, the post-cutoff branches are linear in `theta`.",
        "",
        "The BHPT waveform is the raw surrogate output with `calibrated=False`; the",
        "NR reference is `NRHybSur3dq8`.",
        "",
        "```python",
        f"q = {Q}",
        "mode = (2, 2)",
        "nu = q / (1 + q)**2",
        f"nu = {meta['nu']:.15g}",
        "",
        f"t_bhpt_merger = {meta['t_bhpt_merger']:.15g}",
        f"t_nr_merger = {meta['t_nr_merger']:.15g}",
        "theta = nu * (t_bhpt - t_bhpt_merger)",
        "Theta_nr = nu * (t_nr - t_nr_merger)",
        "```",
        "",
        "The 10 fitted parameters are:",
        "",
        "```python",
        f"theta_cut = {theta_cut:.15g}",
        f"t_cut_nr = {p['t_cut_nr']:.15g}",
        f"theta_transition_width = {width:.15g}",
        f"beta_left = {p['beta_left']:.15g}",
        f"beta_right_edge = {p['beta_right_edge']:.15g}",
        f"beta_right_slope = {p['beta_right_slope']:.15g}",
        f"alpha_left = {p['alpha_left']:.15g}",
        f"alpha_right_edge = {p['alpha_right_edge']:.15g}",
        f"alpha_right_slope = {p['alpha_right_slope']:.15g}",
        f"phi0 = {principal_phase(p['phi0']):.15g}",
        "```",
        "",
        "The optimized source cutoff and transition are:",
        "",
        "```python",
        f"t_bhpt_cut = t_bhpt_merger + theta_cut / nu  # {s_cut:.15g}",
        f"theta_transition_end = theta_cut + theta_transition_width  # {theta_end:.15g}",
        f"t_bhpt_transition_end = t_bhpt_merger + theta_transition_end / nu  # {s_end:.15g}",
        f"Theta_cut_nr = {theta_cut_nr:.15g}",
        f"t_cut_nr = t_nr_merger + Theta_cut_nr / nu  # {p['t_cut_nr']:.15g}",
        "```",
        "",
        "Functional form:",
        "",
        "```python",
        "# theta <= theta_cut",
        "alpha(theta) = alpha_left",
        "beta(theta) = beta_left",
        "",
        "# theta_cut < theta < theta_cut + theta_transition_width",
        "z = (theta - theta_cut) / theta_transition_width",
        "H00 = 2*z**3 - 3*z**2 + 1",
        "H01 = -2*z**3 + 3*z**2",
        "H11 = z**3 - z**2",
        "y_smooth = H00*y_left + H01*y_right_edge + H11*theta_transition_width*y_right_slope",
        "",
        "# theta >= theta_cut + theta_transition_width",
        "alpha(theta) = alpha_right_edge + alpha_right_slope * (theta - theta_transition_end)",
        "beta(theta) = beta_right_edge + beta_right_slope * (theta - theta_transition_end)",
        "```",
        "",
        "The time map is the numerical integral of the smooth positive `beta(theta)`,",
        "shifted so that `tau(t_bhpt_cut) = t_cut_nr`:",
        "",
        "```python",
        "d tau / d t_bhpt = beta(theta)",
        "tau(t_bhpt_cut) = t_cut_nr",
        "h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)",
        "```",
        "",
        "Transition values:",
        "",
        "| location | theta | t_bhpt | mapped tau | alpha | beta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| cutoff/start | {transition_theta[0]:.15g} | {transition_t_bhpt[0]:.15g} | {transition_t_nr[0]:.15g} | {transition_values['alpha'][0]:.15g} | {transition_values['beta'][0]:.15g} |",
        f"| transition end | {transition_theta[1]:.15g} | {transition_t_bhpt[1]:.15g} | {transition_t_nr[1]:.15g} | {transition_values['alpha'][1]:.15g} | {transition_values['beta'][1]:.15g} |",
        "",
        "Diagnostics:",
        "",
        f"- q: `{Q}`",
        f"- mode: `{MODE}`",
        f"- time window: NR restricted to `{NR_T_START}` through `{NR_T_END}`, then common support only",
        "- BHPT `calibrated`: `False`",
        "- optimized phase rotation: `True`",
        f"- common NR support: `[{common_t[0]:.15g}, {common_t[-1]:.15g}]`",
        f"- optimizer downsampled `mathcalE`: `{fit_error:.15g}`",
        f"- full-window `mathcalE`: `{eval_data['error']:.15g}`",
        f"- pre-transition local `mathcalE`: `{before_err:.15g}`",
        f"- smooth-transition local `mathcalE`: `{transition_err:.15g}`",
        f"- post-transition local `mathcalE`: `{after_err:.15g}`",
        f"- alpha endpoint values `(used_start, used_end)`: `({endpoint_values['alpha'][0]:.15g}, {endpoint_values['alpha'][1]:.15g})`",
        f"- beta endpoint values `(used_start, used_end)`: `({endpoint_values['beta'][0]:.15g}, {endpoint_values['beta'][1]:.15g})`",
        f"- minimum alpha on used source support: `{np.min(eval_data['alpha']):.15g}`",
        f"- minimum beta on used source support: `{np.min(eval_data['beta']):.15g}`",
        "- pre-cutoff alpha and beta slopes: `0` by construction",
        "- transition edge value jumps: `0` by construction for both alpha and beta",
        "",
        "No extrapolated BHPT samples are used. The evaluated source samples are the",
        "raw BHPT samples whose fitted `tau(t_bhpt)` lies inside the NR comparison",
        "support, and the NR grid is then restricted to that common support.",
        "",
    ]
    (ROOT / "scaling_stricter_physical_smooth.md").write_text("\n".join(lines))


def main() -> None:
    t_bhpt, h_bhpt, t_nr, h_nr = load_waveforms()
    meta = physical_metadata(t_bhpt, h_bhpt, t_nr, h_nr)
    params, fit_error = fit_params(t_bhpt, h_bhpt, t_nr, h_nr, meta)
    eval_data = evaluate_model(params, t_bhpt, h_bhpt, t_nr, h_nr, meta)
    write_markdown(params, fit_error, eval_data, t_bhpt, h_bhpt, t_nr, meta)

    p = unpack(params)
    print(f"mathcalE={eval_data['error']:.15g}")
    print(f"theta_cut={p['theta_cut']:.15g}")
    print(f"theta_transition_width={p['theta_transition_width']:.15g}")
    print(f"t_cut_nr={p['t_cut_nr']:.15g}")
    print(f"phi0={principal_phase(p['phi0']):.15g}")
    print("params=" + np.array2string(params, precision=15))


if __name__ == "__main__":
    main()
