from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PLOT_DIR = ROOT / "Notebook_plots" / "wf_peak_ratios"
RESULTS_PATH = ROOT / "wf_peak_ratios_results" / "fit_results.json"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "BHPTNRSurrogate" / "surrogates"))

import fit_scaling_wf_peak_ratios as wfpeak  # noqa: E402
import gwsurrogate  # noqa: E402


mpl.rcdefaults()


def q_label(q: float) -> str:
    return f"{q:g}"


def q_tag(q: float) -> str:
    return q_label(q).replace(".", "p").replace("-", "m")


def load_model() -> dict:
    return json.loads(RESULTS_PATH.read_text())["model"]


def plot_args() -> SimpleNamespace:
    return SimpleNamespace(
        min_peak_time=-5000.0,
        max_peak_time=-20.0,
        prominence_fraction=0.01,
        min_model_peaks=6,
    )


def build_plot_data(q_input: float) -> dict:
    model = load_model()
    args = plot_args()
    nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    case = wfpeak.load_case(nrsur, q_input)
    eval_data = wfpeak.evaluate_case(
        case, model, args, min_coverage=0.0, include_arrays=True
    )
    if "tau" not in eval_data:
        raise RuntimeError(
            f"Could not build q={q_input:g} waveform plot: "
            f"{eval_data.get('failure', 'unknown failure')}"
        )

    tau = eval_data["tau"]
    tau_full, h_scaled_full, alpha_full, beta_full, theta_full, nodes = wfpeak.model_arrays(
        case, model, args
    )
    use = (tau_full >= case["t_nr"][0]) & (tau_full <= case["t_nr"][-1])
    tau_use = tau_full[use]
    h_scaled_use = h_scaled_full[use]
    common = (case["t_nr"] >= tau_use[0]) & (case["t_nr"] <= tau_use[-1])
    h_model0 = wfpeak.interp_complex(tau_use, h_scaled_use, case["t_nr"][common])
    h_model = h_model0 * np.exp(1j * eval_data["phi0"])

    node_s_bhpt = nodes["theta"] / case["meta"]["nu"]
    node_tau = case["meta"]["t_nr_merger"] + nodes["beta"] * node_s_bhpt

    return {
        "case": case,
        "model": model,
        "eval": eval_data,
        "common_t": case["t_nr"][common],
        "h_nr": case["h_nr"][common],
        "h_model": h_model,
        "tau": tau_use,
        "alpha": alpha_full[use],
        "beta": beta_full[use],
        "theta": theta_full[use],
        "node_tau": node_tau,
        "node_alpha": nodes["alpha"],
        "node_beta": nodes["beta"],
    }


def save_waveform_plot(data: dict, q_input: float) -> Path:
    title = f"Waveform peak-ratio fit (q = {q_label(q_input)})"
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    for ax in (ax1, ax2):
        ax.plot(data["common_t"], np.real(data["h_nr"]), label="NR 22 mode")
        ax.plot(data["common_t"], np.real(data["h_model"]), label="BHPT 22")
        ax.grid(True)
        ax.set_xlabel("t/M")

    ax1.set_xlim(-1000, -250)
    ax1.legend()
    ax2.set_xlim(-250, 100)
    fig.suptitle(f"{title}; mathcalE = {data['eval']['error']:.3g}")
    fig.tight_layout()
    path = PLOT_DIR / f"wf_peak_ratios_q{q_tag(q_input)}_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_plot(data: dict, q_input: float) -> Path:
    title = f"Waveform peak-ratio fit (q = {q_label(q_input)})"
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax1.plot(data["tau"], data["alpha"], label="continuous")
    ax1.scatter(
        data["node_tau"],
        data["node_alpha"],
        s=14,
        label="peak nodes",
        zorder=3,
    )
    ax1.set_xlim(-1000, 100)
    ax1.grid(True)
    ax1.set_xlabel("t/M")
    ax1.set_ylabel(r"$\alpha(t)$")
    ax1.legend()

    ax2.plot(data["tau"], data["beta"], label="continuous")
    ax2.scatter(
        data["node_tau"],
        data["node_beta"],
        s=14,
        label="peak nodes",
        zorder=3,
    )
    ax2.set_xlim(-1000, 100)
    ax2.grid(True)
    ax2.set_xlabel("t/M")
    ax2.set_ylabel(r"$\beta(t)$")
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    path = PLOT_DIR / f"wf_peak_ratios_q{q_tag(q_input)}_params.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the waveform peak-ratio q-dependent scaling model."
    )
    parser.add_argument("--q-input", type=float, default=2.0)
    args = parser.parse_args()

    data = build_plot_data(args.q_input)
    waveform_path = save_waveform_plot(data, args.q_input)
    params_path = save_parameter_plot(data, args.q_input)
    print(f"mathcalE={data['eval']['error']:.8g}")
    print(f"coverage={data['eval']['coverage']:.8g}")
    print(f"wrote {waveform_path}")
    print(f"wrote {params_path}")


if __name__ == "__main__":
    main()
