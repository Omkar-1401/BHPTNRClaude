
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import gwsurrogate


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.append(str(REPO_ROOT / "BHPTNRSurrogate" / "surrogates"))
import BHPTNRSur1dq1e4 as bhptsur  # noqa: E402


Q = 5
MODE = (2, 2)
T_START_REQUESTED = -5000.1
T_END = 100.0
CUTOFF_MAX = -100.001

# Parameters are refined by a short Nelder-Mead polish in fit_params().
# Vector entries:
# [t_cut, s_min, s_cut, s_max, k_left, k_right,
#  alpha_left_min, alpha_left_cut, alpha_right_cut, alpha_right_max, phi]
INITIAL_PARAMS = np.array(
    [
        -179.06147911741959,
        -6211.6636152902138,
        -221.39978461664884,
        114.80000009274141,
        4.1948613171004464e-08,
        0.00012760251505617359,
        0.80743219206339845,
        0.80846982337499451,
        0.85852691114074275,
        0.66940798349834685,
        510.35966328500285,
    ],
    dtype=float,
)


def mathcalE_error(h1: np.ndarray, h2: np.ndarray) -> float:
    """Definition used in NRBHP_ansatz.py, with h1 as the reference waveform."""
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


def principal_phase(phi: float) -> float:
    return float((phi + np.pi) % (2.0 * np.pi) - np.pi)


def load_waveforms():
    t_bhpt, h_bhpt = bhptsur.generate_surrogate(q=Q, calibrated=False)
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    t_nr, h_nr, _ = nrsur(Q, [0, 0, 0.0], [0, 0, 0.0], dt=0.1, f_low=5e-3)
    mask = (t_nr >= T_START_REQUESTED) & (t_nr <= T_END)
    return t_bhpt, h_bhpt[MODE], t_nr[mask], h_nr[MODE][mask]


def unpack(params: np.ndarray) -> dict[str, float]:
    (
        t_cut,
        s_min,
        s_cut,
        s_max,
        k_left,
        k_right,
        alpha_left_min,
        alpha_left_cut,
        alpha_right_cut,
        alpha_right_max,
        phi,
    ) = [float(v) for v in params]
    return {
        "t_cut": t_cut,
        "s_min": s_min,
        "s_cut": s_cut,
        "s_max": s_max,
        "k_left": k_left,
        "k_right": k_right,
        "alpha_left_min": alpha_left_min,
        "alpha_left_cut": alpha_left_cut,
        "alpha_right_cut": alpha_right_cut,
        "alpha_right_max": alpha_right_max,
        "phi": phi,
    }


def beta_averages(params: np.ndarray, t_start: float) -> tuple[float, float]:
    p = unpack(params)
    beta_bar_left = (p["t_cut"] - t_start) / (p["s_cut"] - p["s_min"])
    beta_bar_right = (T_END - p["t_cut"]) / (p["s_max"] - p["s_cut"])
    return beta_bar_left, beta_bar_right


def beta_endpoints(params: np.ndarray, t_start: float) -> tuple[float, float, float, float]:
    p = unpack(params)
    beta_bar_left, beta_bar_right = beta_averages(params, t_start)
    beta_left_min = beta_bar_left + p["k_left"] * (p["s_min"] - p["s_cut"])
    beta_left_cut = beta_bar_left + p["k_left"] * (p["s_cut"] - p["s_min"])
    beta_right_cut = beta_bar_right + p["k_right"] * (p["s_cut"] - p["s_max"])
    beta_right_max = beta_bar_right + p["k_right"] * (p["s_max"] - p["s_cut"])
    return beta_left_min, beta_left_cut, beta_right_cut, beta_right_max


def validate_params(params: np.ndarray, t_bhpt: np.ndarray, t_start: float) -> bool:
    p = unpack(params)
    if not (-800.0 <= p["t_cut"] <= CUTOFF_MAX):
        return False
    if not (-7200.0 <= p["s_min"] <= -5600.0):
        return False
    if not (-800.0 <= p["s_cut"] <= -80.0):
        return False
    if not (80.0 <= p["s_max"] <= float(t_bhpt[-1]) + 1e-9):
        return False
    if not (p["s_min"] < p["s_cut"] < p["s_max"]):
        return False
    alphas = [
        p["alpha_left_min"],
        p["alpha_left_cut"],
        p["alpha_right_cut"],
        p["alpha_right_max"],
    ]
    if min(alphas) <= 0.05 or max(alphas) > 5.0:
        return False
    betas = beta_endpoints(params, t_start)
    if min(betas) <= 0.05 or max(betas) > 2.0:
        return False
    return True


def transform_source_grid(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_start: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = unpack(params)
    source_inner = t_bhpt[(t_bhpt > p["s_min"]) & (t_bhpt < p["s_max"])]
    source_t = np.unique(np.r_[p["s_min"], p["s_cut"], source_inner, p["s_max"]])
    h_source = interp_complex(t_bhpt, h_bhpt, source_t)

    left = source_t <= p["s_cut"]
    tau = np.empty_like(source_t, dtype=float)
    alpha = np.empty_like(source_t, dtype=float)
    beta_bar_left, beta_bar_right = beta_averages(params, t_start)

    tau[left] = (
        t_start
        + beta_bar_left * (source_t[left] - p["s_min"])
        + p["k_left"] * (source_t[left] - p["s_min"]) * (source_t[left] - p["s_cut"])
    )
    tau[~left] = (
        p["t_cut"]
        + beta_bar_right * (source_t[~left] - p["s_cut"])
        + p["k_right"] * (source_t[~left] - p["s_cut"]) * (source_t[~left] - p["s_max"])
    )

    x_left = (source_t[left] - p["s_min"]) / (p["s_cut"] - p["s_min"])
    x_right = (source_t[~left] - p["s_cut"]) / (p["s_max"] - p["s_cut"])
    alpha[left] = p["alpha_left_min"] + (
        p["alpha_left_cut"] - p["alpha_left_min"]
    ) * x_left
    alpha[~left] = p["alpha_right_cut"] + (
        p["alpha_right_max"] - p["alpha_right_cut"]
    ) * x_right

    return source_t, tau, alpha, h_source


def evaluate_model(
    params: np.ndarray,
    t_bhpt: np.ndarray,
    h_bhpt: np.ndarray,
    t_nr: np.ndarray,
    h_nr: np.ndarray,
) -> dict:
    t_start = float(t_nr[0])
    if not validate_params(params, t_bhpt, t_start):
        return {"error": 50.0}

    source_t, tau, alpha, h_source = transform_source_grid(
        params, t_bhpt, h_bhpt, t_start
    )
    if np.any(np.diff(tau) <= 0.0) or np.any(alpha <= 0.0):
        return {"error": 50.0}

    common = (t_nr >= tau[0]) & (t_nr <= tau[-1])
    if np.count_nonzero(common) < int(0.85 * len(t_nr)):
        return {"error": 50.0}

    h_scaled = alpha * np.exp(1j * unpack(params)["phi"]) * h_source
    h_model = interp_complex(tau, h_scaled, t_nr[common])
    err = mathcalE_error(h_nr[common], h_model)
    return {
        "error": err,
        "source_t": source_t,
        "tau": tau,
        "alpha": alpha,
        "common": common,
        "h_model": h_model,
        "h_ref": h_nr[common],
    }


def fit_params(t_bhpt: np.ndarray, h_bhpt: np.ndarray, t_nr: np.ndarray, h_nr: np.ndarray):
    sample = np.unique(np.r_[np.arange(0, len(t_nr), 5), len(t_nr) - 1])
    t_fit = t_nr[sample]
    h_fit = h_nr[sample]

    def objective(params: np.ndarray) -> float:
        return evaluate_model(params, t_bhpt, h_bhpt, t_fit, h_fit)["error"]

    result = minimize(
        objective,
        INITIAL_PARAMS.copy(),
        method="Nelder-Mead",
        options={"maxiter": 1200, "xatol": 1e-10, "fatol": 1e-12},
    )
    return result.x, float(result.fun)


def alpha_on_source(params: np.ndarray, source_t: np.ndarray) -> np.ndarray:
    p = unpack(params)
    alpha = np.empty_like(source_t, dtype=float)
    left = source_t <= p["s_cut"]
    x_left = (source_t[left] - p["s_min"]) / (p["s_cut"] - p["s_min"])
    x_right = (source_t[~left] - p["s_cut"]) / (p["s_max"] - p["s_cut"])
    alpha[left] = p["alpha_left_min"] + (
        p["alpha_left_cut"] - p["alpha_left_min"]
    ) * x_left
    alpha[~left] = p["alpha_right_cut"] + (
        p["alpha_right_max"] - p["alpha_right_cut"]
    ) * x_right
    return alpha


def beta_on_source(params: np.ndarray, source_t: np.ndarray, t_start: float) -> np.ndarray:
    p = unpack(params)
    beta_bar_left, beta_bar_right = beta_averages(params, t_start)
    beta = np.empty_like(source_t, dtype=float)
    left = source_t <= p["s_cut"]
    beta[left] = beta_bar_left + p["k_left"] * (
        2.0 * source_t[left] - p["s_min"] - p["s_cut"]
    )
    beta[~left] = beta_bar_right + p["k_right"] * (
        2.0 * source_t[~left] - p["s_cut"] - p["s_max"]
    )
    return beta


def write_markdown(params: np.ndarray, fit_error: float, eval_data: dict, t_nr: np.ndarray):
    p = unpack(params)
    beta_bar_left, beta_bar_right = beta_averages(params, float(t_nr[0]))
    betas = beta_endpoints(params, float(t_nr[0]))
    common_t = t_nr[eval_data["common"]]
    h_ref = eval_data["h_ref"]
    h_model = eval_data["h_model"]
    left_mask = common_t <= p["t_cut"]
    right_mask = common_t > p["t_cut"]
    left_err = mathcalE_error(h_ref[left_mask], h_model[left_mask])
    right_err = mathcalE_error(h_ref[right_mask], h_model[right_mask])

    lines = [
        "# Stricter q=5 (2,2) piecewise-linear scaling fit",
        "",
        "Using a two-segment linear fit for `alpha(s)` and `beta(s)` in raw BHPT source time `s`, I get:",
        "",
        "```python",
        f"t_start = {common_t[0]:.15g}",
        f"t_end = {T_END:.15g}",
        f"t_cut = {p['t_cut']:.15g}  # optimized cutoff, constrained to be < -100M",
        f"s_min = {p['s_min']:.15g}",
        f"s_cut = {p['s_cut']:.15g}",
        f"s_max = {p['s_max']:.15g}",
        "```",
        "",
        "For `s <= s_cut`:",
        "",
        "```python",
        "x_left = (s - s_min) / (s_cut - s_min)",
        f"alpha_left(s) = {p['alpha_left_min']:.15g} + ({p['alpha_left_cut'] - p['alpha_left_min']:.15g}) * x_left",
        f"tau_left(s) = t_start + {beta_bar_left:.15g} * (s - s_min) + ({p['k_left']:.15g}) * (s - s_min) * (s - s_cut)",
        f"beta_left(s) = {beta_bar_left:.15g} + ({p['k_left']:.15g}) * (2*s - s_min - s_cut)",
        "```",
        "",
        "For `s > s_cut`:",
        "",
        "```python",
        "x_right = (s - s_cut) / (s_max - s_cut)",
        f"alpha_right(s) = {p['alpha_right_cut']:.15g} + ({p['alpha_right_max'] - p['alpha_right_cut']:.15g}) * x_right",
        f"tau_right(s) = t_cut + {beta_bar_right:.15g} * (s - s_cut) + ({p['k_right']:.15g}) * (s - s_cut) * (s - s_max)",
        f"beta_right(s) = {beta_bar_right:.15g} + ({p['k_right']:.15g}) * (2*s - s_cut - s_max)",
        "```",
        "",
        "Phase rotation:",
        "",
        "```python",
        f"phi0 = {principal_phase(p['phi']):.15g}  # radians, principal value",
        f"phi0_unwrapped = {p['phi']:.15g}",
        "```",
        "",
        "Applied as:",
        "",
        "```python",
        "h_model(tau(s)) = alpha(s) * exp(1j * phi0) * h_BHPT(s)",
        "```",
        "",
        "The fit uses raw `BHPTNRSur1dq1e4` with `calibrated=False`, `NRHybSur3dq8` with `dt=0.1`, `f_low=5e-3`, and the `mathcalE_error` definition in `NRBHP_ansatz.py` with NR as the reference waveform.",
        "",
        "Diagnostics:",
        "",
        f"- q: `{Q}`",
        f"- mode: `{MODE}`",
        f"- optimized cutoff in NR time: `{p['t_cut']:.15g}`",
        f"- corresponding BHPT source split: `{p['s_cut']:.15g}`",
        f"- common NR support: `[{common_t[0]:.15g}, {common_t[-1]:.15g}]`",
        f"- beta endpoint values `(left_min, left_cut, right_cut, right_max)`: `({betas[0]:.15g}, {betas[1]:.15g}, {betas[2]:.15g}, {betas[3]:.15g})`",
        f"- optimizer downsampled `mathcalE`: `{fit_error:.15g}`",
        f"- full-window `mathcalE`: `{eval_data['error']:.15g}`",
        f"- `t <= t_cut` local `mathcalE`: `{left_err:.15g}`",
        f"- `t > t_cut` local `mathcalE`: `{right_err:.15g}`",
        "",
        "No extrapolated BHPT samples are used. The source interval is inside the raw BHPT support and endpoint BHPT values are obtained by interpolation within that support.",
        "",
    ]
    (ROOT / "scaling_stricter.md").write_text("\n".join(lines))


def plot_waveform(params: np.ndarray, eval_data: dict, t_nr: np.ndarray):
    p = unpack(params)
    common_t = t_nr[eval_data["common"]]
    h_ref = eval_data["h_ref"]
    h_model = eval_data["h_model"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.2),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 2]},
    )
    windows = [(-1000.0, p["t_cut"]), (p["t_cut"], T_END)]
    for ax, (lo, hi) in zip(axes, windows):
        mask = (common_t >= lo) & (common_t <= hi)
        ax.plot(common_t[mask], np.real(h_ref[mask]), label="NRHybSur3dq8", lw=1.7)
        ax.plot(
            common_t[mask],
            np.real(h_model[mask]),
            label="scaled raw BHPT",
            lw=1.2,
            ls="--",
        )
        ax.set_xlim(lo, hi)
        ax.set_xlabel("t / M")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Re[h22]")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Stricter agentic fit")
    fig.text(
        0.5,
        0.925,
        f"q=5, (2,2), cutoff={p['t_cut']:.2f}M, mathcalE={eval_data['error']:.3e}",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(ROOT / "scaled_bhpt_vs_nrhybsur_q5_22_stricter.pdf")
    plt.close(fig)


def plot_scaling(params: np.ndarray, t_start: float):
    p = unpack(params)
    left_s = np.linspace(p["s_min"], p["s_cut"], 400)
    right_s = np.linspace(p["s_cut"], p["s_max"], 400)
    source_t = np.r_[left_s, right_s[1:]]
    alpha = alpha_on_source(params, source_t)
    beta = beta_on_source(params, source_t, t_start)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    axes[0].plot(source_t, alpha, color="C0", lw=1.6)
    axes[0].axvline(p["s_cut"], color="0.25", lw=1.0, ls=":")
    axes[0].set_ylabel("alpha(s)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(source_t, beta, color="C1", lw=1.6)
    axes[1].axvline(p["s_cut"], color="0.25", lw=1.0, ls=":")
    axes[1].set_xlabel("raw BHPT source time s / M")
    axes[1].set_ylabel("beta(s)")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Piecewise-linear scaling, tau(s_cut)={p['t_cut']:.2f}M")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ROOT / "scaling_stricter.pdf")
    plt.close(fig)


def main() -> None:
    t_bhpt, h_bhpt, t_nr, h_nr = load_waveforms()
    params, fit_error = fit_params(t_bhpt, h_bhpt, t_nr, h_nr)
    eval_data = evaluate_model(params, t_bhpt, h_bhpt, t_nr, h_nr)
    if eval_data["error"] >= 1.0:
        raise RuntimeError("Piecewise-linear fit failed.")

    write_markdown(params, fit_error, eval_data, t_nr)
    plot_waveform(params, eval_data, t_nr)
    plot_scaling(params, float(t_nr[0]))

    p = unpack(params)
    print(f"mathcalE={eval_data['error']:.15g}")
    print(f"t_cut={p['t_cut']:.15g}")
    print(f"s_cut={p['s_cut']:.15g}")
    print(f"phi0={principal_phase(p['phi']):.15g}")


if __name__ == "__main__":
    main()
