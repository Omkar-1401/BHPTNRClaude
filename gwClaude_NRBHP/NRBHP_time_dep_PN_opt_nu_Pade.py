"""
Plotting script for the PP-anchored nu/Pade bake-off model (PN_opt_nu_Pade).

Reads the selected fit from PN_opt_nu_Pade_results/selected_fit.json (written by
fit_scaling_PN_opt_nu_Pade.py) and generates diagnostic plots for
q = 2, 3, 5, 8 in Agentic_plots/Pade_nu/.  q = 2 is an extrapolation test
(training range is q in [3, 8]).

Change Q_INPUTS to plot different values; no other edits needed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTutils"))

import bhpt_utils
import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_PN_opt_nu_Pade as npade

mpl.rcdefaults()

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
PLOT_DIR = ROOT / "Agentic_plots" / "Pade_nu"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_JSON = npade.SELECTED_JSON


def load_reps() -> tuple[str, dict]:
    d = json.loads(SELECTED_JSON.read_text())
    return d["form"], d["reps"]


def _extrap(q: float) -> bool:
    return q < 3.0 or q > 8.0


def _tag(q: float) -> str:
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if _extrap(q) else "")


def _title(q: float, form: str, error: float = float("nan")) -> str:
    extrap = " [EXTRAPOLATION]" if _extrap(q) else ""
    err_str = f";  mathcalE = {error:.3g}" if np.isfinite(error) else ""
    return f"PN_opt_nu_Pade ({form})  q={q}{extrap};  nu={npade.nu_of(q):.4f}{err_str}"


def build_model(q: float, reps: dict) -> dict:
    qdep.generate_and_cache_waveform(q)
    case = qdep.load_case(q, source_stride=3, nr_stride=5)
    params = npade.constrained_master_params(q, reps)
    params = qdep.anchored_start(params, case["meta"])
    params = qdep.phase_seed(params, case)
    params, _ = qdep.polish_nuisance(params, case)
    ev = creative.evaluate_model(
        params, case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"], case["losses"], min_coverage=qdep.MIN_COVERAGE,
    )
    arrays = creative.model_arrays(params, case["t_bhpt"], case["h_bhpt"], case["losses"])
    return {"eval": ev, "params": params, "arrays": arrays, "case": case}


def save_waveform_plot(q, form, common_t, h_nr, h_model, t_switch_nr, error) -> Path:
    split = float(np.clip(t_switch_nr, -900.0, -50.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                   gridspec_kw={"width_ratios": [3, 2]})
    for ax, xlim in [(ax1, (-1000, split)), (ax2, (split, 100))]:
        ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
        ax.plot(common_t, np.real(h_model), label="BHPT calibrated", alpha=0.8)
        ax.set_xlim(*xlim); ax.grid(True); ax.set_xlabel("t/M")
    ax1.legend(fontsize=8)
    fig.suptitle(_title(q, form, error)); fig.tight_layout()
    path = PLOT_DIR / f"nu_Pade_{_tag(q)}_waveform.pdf"
    fig.savefig(path); plt.close(fig)
    return path


def save_parameter_plot(q, form, tau, alpha, beta) -> Path:
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(q, [creative.MODE])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(tau, alpha, label="PN_opt_nu_Pade")
    ax1.axhline(ab["alpha_l2m2"], color="k", ls="--", label="BHPTNRSurrogate")
    ax1.set_xlim(-1000, 100); ax1.grid(True)
    ax1.set_xlabel("t/M"); ax1.set_ylabel(r"$\alpha(t)$"); ax1.legend(fontsize=7)
    ax2.plot(tau, beta, label="PN_opt_nu_Pade")
    ax2.axhline(ab["beta"], color="k", ls="--", label="BHPTNRSurrogate")
    ax2.set_xlim(-1000, 100); ax2.grid(True)
    ax2.set_xlabel("t/M"); ax2.set_ylabel(r"$\beta(t)$")
    fig.suptitle(_title(q, form)); fig.tight_layout()
    path = PLOT_DIR / f"nu_Pade_{_tag(q)}_params.pdf"
    fig.savefig(path); plt.close(fig)
    return path


def save_zoomed_plot(q, form, common_t, h_nr, h_model) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax.plot(common_t, np.real(h_model), label="BHPT calibrated", alpha=0.8)
    ax.set_xlim(-100, 100); ax.grid(True)
    ax.set_xlabel("t/M"); ax.legend(fontsize=8); ax.set_title(_title(q, form))
    fig.tight_layout()
    path = PLOT_DIR / f"nu_Pade_{_tag(q)}_zoomed.pdf"
    fig.savefig(path); plt.close(fig)
    return path


def save_param_vs_q(reps: dict, form: str) -> Path:
    """Show how each fitted parameter behaves vs q, incl. PP anchor and extrapolation."""
    cache = json.loads(npade.QDEP_CACHE.read_text())
    rows = [cache[k] for k in sorted(cache, key=float)]
    q_perq = np.array([r["q"] for r in rows])
    P = np.array([r["params"] for r in rows])
    q_fine = np.linspace(1.5, 9.0, 300)
    show = ["p0", "w", "alpha_i", "beta_i", "beta_r", "beta_L"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, name in zip(axes.ravel(), show):
        i = npade.PARAM_NAMES.index(name)
        ax.scatter(q_perq, P[:, i], s=8, c="k", zorder=5, label="per-q fit")
        ax.plot(q_fine, [npade.eval_param(q, reps[name]) for q in q_fine],
                "g-", label=f"{form}")
        ax.axvline(3.0, color="gray", ls="--", lw=0.8)
        ax.axvline(8.0, color="gray", ls="--", lw=0.8)
        anch = npade.PP_ANCHOR[name]
        if anch is not None:
            ax.axhline(anch, color="b", ls=":", lw=0.8, label=f"PP limit={anch:g}")
        ax.set_xlabel("q"); ax.set_ylabel(name); ax.grid(True); ax.legend(fontsize=7)
    fig.suptitle(f"PN_opt_nu_Pade ({form}): parameter q-dependence "
                 "(dashed = training edges; q<3 is extrapolation)")
    fig.tight_layout()
    path = PLOT_DIR / "nu_Pade_param_vs_q.pdf"
    fig.savefig(path); plt.close(fig)
    return path


def main() -> None:
    form, reps = load_reps()
    print(f"Selected form: {form}")
    all_paths, errors = [], {}
    for q in Q_INPUTS:
        print(f"\n--- q={q}{' [EXTRAP]' if _extrap(q) else ''} ---", flush=True)
        res = build_model(q, reps)
        ev, arrays, case, params = res["eval"], res["arrays"], res["case"], res["params"]
        tau, _, alpha, beta, _ = arrays
        use = (tau >= case["t_nr"][0]) & (tau <= case["t_nr"][-1])
        common_t = case["t_nr"][ev["common"]]
        p = creative.unpack(params)
        t_switch_nr = float(np.interp(p["p0"], ev["p_loss"], ev["tau"]))
        errors[q] = ev["error"]
        print(f"  mathcalE = {ev['error']:.6g}   coverage = {ev.get('coverage', float('nan')):.4f}")
        all_paths += [
            save_waveform_plot(q, form, common_t, ev["h_ref"], ev["h_model"], t_switch_nr, ev["error"]),
            save_parameter_plot(q, form, tau[use], alpha[use], beta[use]),
            save_zoomed_plot(q, form, common_t, ev["h_ref"], ev["h_model"]),
        ]
    all_paths.append(save_param_vs_q(reps, form))

    print("\n=== Summary ===")
    for q in Q_INPUTS:
        flag = " [extrap]" if _extrap(q) else "        "
        print(f"  q={q:<4}{flag}  nu={npade.nu_of(q):.4f}  mathcalE={errors[q]:.4e}")
    print("\nPlots:", PLOT_DIR)
    for pth in all_paths:
        print(" ", pth.name)


if __name__ == "__main__":
    main()
