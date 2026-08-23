"""
Plotting script for the remnant-partial (factorised beta_r) model.

Generates diagnostic plots for q = 2, 3, 5, 8 in
Agentic_plots/Remnant_cases/.  q = 2 is an extrapolation test.

Change Q_INPUTS to plot different values; no other edits needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
import fit_scaling_PN_opt_remnant_partial as rpartial

mpl.rcdefaults()

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
MD_PATH  = ROOT / "scaling_PN_opt_remnant_partial.md"
PLOT_DIR = ROOT / "Agentic_plots" / "Remnant_cases"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Coefficient loading
# ---------------------------------------------------------------------------

def read_markdown_coefficients(md_path: Path) -> dict[str, np.ndarray]:
    text = md_path.read_text()
    heading = "## Selected Master Coefficients"
    start = text.index(heading)
    lines = text[start:].splitlines()
    table_lines, in_table = [], False
    for line in lines:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            table_lines.append(s)
            in_table = True
        elif in_table:
            break
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(c.replace(":", "").replace("-", "").replace("★", "") == "" for c in cells):
            continue
        rows.append(cells)
    if not rows:
        raise ValueError(f"No coefficient table found in {md_path}")
    coeffs: dict[str, np.ndarray] = {}
    for row in rows[1:]:
        name = row[0].rstrip(" ★")
        coeffs[name] = np.array([float(v) for v in row[1:]], dtype=float)
    missing = [n for n in qdep.PARAM_NAMES if n not in coeffs]
    if missing:
        raise ValueError(f"Missing rows: {missing}")
    return {n: coeffs[n] for n in qdep.PARAM_NAMES}


# ---------------------------------------------------------------------------
# Build model for a given q
# ---------------------------------------------------------------------------

def build_model(q: float, coeffs: dict[str, np.ndarray]) -> dict:
    qdep.generate_and_cache_waveform(q)
    case = qdep.load_case(q, source_stride=3, nr_stride=5)
    params = rpartial.constrained_master_params(q, coeffs)
    params, _ = rpartial.polish_nuisance(params, case)
    eval_data = creative.evaluate_model(
        params,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"],
        case["losses"], min_coverage=qdep.MIN_COVERAGE,
    )
    arrays = creative.model_arrays(params, case["t_bhpt"], case["h_bhpt"], case["losses"])
    return {"eval": eval_data, "params": params, "arrays": arrays, "case": case}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _extrap(q: float) -> bool:
    return q < 3.0 or q > 8.0


def _tag(q: float) -> str:
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if _extrap(q) else "")


def _title(q: float, error: float = float("nan")) -> str:
    chi_f_val = rpartial.get_chi_f(q)
    mf_val, _ = rpartial.get_remnant(q)
    brp = rpartial.beta_r_physical(mf_val, chi_f_val, q)
    extrap = " [extrapolation]" if _extrap(q) else ""
    err_str = f";  mathcalE = {error:.3g}" if np.isfinite(error) else ""
    return (f"Remnant-partial  q={q}{extrap}"
            f";  chi_f={chi_f_val:.3f}  beta_r_phys={brp:.3f}{err_str}")


def save_waveform_plot(q, common_t, h_nr, h_model, t_switch_nr, error) -> Path:
    split = float(np.clip(t_switch_nr, -900.0, -50.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    gridspec_kw={"width_ratios": [3, 2]})
    for ax, xlim in [(ax1, (-1000, split)), (ax2, (split, 100))]:
        ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
        ax.plot(common_t, np.real(h_model), label="BHPT calibrated", alpha=0.8)
        ax.set_xlim(*xlim)
        ax.grid(True)
        ax.set_xlabel("t/M")
    ax1.legend(fontsize=8)
    fig.suptitle(_title(q, error))
    fig.tight_layout()
    path = PLOT_DIR / f"remnant_partial_{_tag(q)}_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_plot(q, tau, alpha, beta) -> Path:
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(q, [creative.MODE])
    mf_val, chi_f_val = rpartial.get_remnant(q)
    brp = rpartial.beta_r_physical(mf_val, chi_f_val, q)
    label = f"remnant-partial  chi_f={chi_f_val:.3f}  beta_r_phys={brp:.3f}"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(tau, alpha, label=label)
    ax1.axhline(ab["alpha_l2m2"], color="k", ls="--", label="BHPTNRSurrogate")
    ax1.set_xlim(-1000, 100); ax1.grid(True)
    ax1.set_xlabel("t/M"); ax1.set_ylabel(r"$\alpha(t)$")
    ax1.legend(fontsize=7)
    ax2.plot(tau, beta, label=label)
    ax2.axhline(ab["beta"], color="k", ls="--", label="BHPTNRSurrogate")
    ax2.set_xlim(-1000, 100); ax2.grid(True)
    ax2.set_xlabel("t/M"); ax2.set_ylabel(r"$\beta(t)$")
    fig.suptitle(_title(q))
    fig.tight_layout()
    path = PLOT_DIR / f"remnant_partial_{_tag(q)}_params.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_zoomed_plot(q, common_t, h_nr, h_model) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax.plot(common_t, np.real(h_model), label="BHPT calibrated", alpha=0.8)
    ax.set_xlim(-100, 100); ax.grid(True)
    ax.set_xlabel("t/M"); ax.legend(fontsize=8)
    ax.set_title(_title(q))
    fig.tight_layout()
    path = PLOT_DIR / f"remnant_partial_{_tag(q)}_zoomed.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_beta_r_decomposition(coeffs: dict) -> Path:
    """Show beta_r physical, correction factor r, and resulting beta_r across q."""
    q_fine = np.linspace(1.5, 9.0, 200)
    brphys_arr  = np.array([rpartial.beta_r_physical(*rpartial.get_remnant(float(q)), float(q))
                             for q in q_fine])
    r_arr       = np.array([rpartial.eval_poly(float(q), coeffs["beta_r"])
                             for q in q_fine])
    br_model    = r_arr * brphys_arr

    # per-q fitted beta_r from cache
    cache_path = ROOT / "PN_opt_creative_q_dep_results" / "per_q_cache.json"
    qs_perq, br_perq = [], []
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        for qk in sorted(cache.keys()):
            qs_perq.append(float(qk))
            br_perq.append(cache[qk]["params"][6])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(q_fine, brphys_arr, "b-", label=r"$\beta_r^{\rm phys}$ (QNM ratio)")
    axes[0].axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
    axes[0].axvline(3.0, color="gray", ls="--", lw=0.8, label="training boundary")
    axes[0].set_xlabel("q"); axes[0].set_ylabel(r"$\beta_r^{\rm phys}$")
    axes[0].set_title("Physical QNM prediction")
    axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].plot(q_fine, r_arr, "r-", label=r"$r_\beta({\rm chi}_f)$ polynomial")
    if qs_perq:
        r_perq = [br/rpartial.beta_r_physical(*rpartial.get_remnant(qq), qq)
                  for qq, br in zip(qs_perq, br_perq)]
        axes[1].scatter(qs_perq, r_perq, s=8, c="k", zorder=5, label="per-q fit")
    axes[1].axvline(3.0, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("q"); axes[1].set_ylabel(r"$r_\beta = \beta_r / \beta_r^{\rm phys}$")
    axes[1].set_title("Correction factor polynomial")
    axes[1].legend(fontsize=8); axes[1].grid(True)

    axes[2].plot(q_fine, br_model, "g-", label=r"$\beta_r = r_\beta \cdot \beta_r^{\rm phys}$")
    if qs_perq:
        axes[2].scatter(qs_perq, br_perq, s=8, c="k", zorder=5, label="per-q fit")
    axes[2].axvline(3.0, color="gray", ls="--", lw=0.8, label="training boundary")
    axes[2].set_xlabel("q"); axes[2].set_ylabel(r"$\beta_r$")
    axes[2].set_title(r"Reconstructed $\beta_r$")
    axes[2].legend(fontsize=8); axes[2].grid(True)

    fig.suptitle("beta_r factorisation: physical × correction")
    fig.tight_layout()
    path = PLOT_DIR / "remnant_partial_beta_r_decomp.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_coefficient_table(coeffs: dict) -> Path:
    degree = len(next(iter(coeffs.values()))) - 1
    columns = ["parameter", *[f"c{i}" for i in range(degree + 1)], "value q=5"]
    table_rows = []
    for name in qdep.PARAM_NAMES:
        val = rpartial.eval_poly(5.0, coeffs[name])
        note = " [r_beta]" if name == "beta_r" else ""
        table_rows.append([name + note, *[f"{v:.5g}" for v in coeffs[name]], f"{val:.5g}"])
    fig_h = max(4.0, 0.36 * len(table_rows) + 1.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=columns,
                   loc="center", cellLoc="right", colLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.25)
    ax.set_title("Remnant-partial: polynomial coefficients (chi_f basis; beta_r row = r_beta correction)")
    fig.tight_layout()
    path = PLOT_DIR / "remnant_partial_coefficients.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    coeffs = read_markdown_coefficients(MD_PATH)
    all_paths = []

    errors = {}
    for q in Q_INPUTS:
        extrap_str = " [EXTRAPOLATION]" if _extrap(q) else ""
        print(f"\n--- q={q}{extrap_str} ---", flush=True)

        result = build_model(q, coeffs)
        ev = result["eval"]
        arrays = result["arrays"]
        case = result["case"]
        params = result["params"]

        tau, _, alpha, beta, _ = arrays
        use = (tau >= case["t_nr"][0]) & (tau <= case["t_nr"][-1])
        common_t = case["t_nr"][ev["common"]]
        p = creative.unpack(params)
        t_switch_nr = float(np.interp(p["p0"], ev["p_loss"], ev["tau"]))

        mf_val, chi_f_val = rpartial.get_remnant(q)
        brp = rpartial.beta_r_physical(mf_val, chi_f_val, q)
        errors[q] = ev["error"]

        print(f"  chi_f        = {chi_f_val:.4f}")
        print(f"  M_f          = {mf_val:.5f}")
        print(f"  beta_r_phys  = {brp:.4f}")
        print(f"  beta_r_model = {params[6]:.4f}")
        print(f"  r_beta       = {params[6]/brp:.4f}")
        print(f"  mathcalE     = {ev['error']:.6g}")

        all_paths += [
            save_waveform_plot(q, common_t, ev["h_ref"], ev["h_model"],
                               t_switch_nr, ev["error"]),
            save_parameter_plot(q, tau[use], alpha[use], beta[use]),
            save_zoomed_plot(q, common_t, ev["h_ref"], ev["h_model"]),
        ]

    all_paths += [
        save_coefficient_table(coeffs),
        save_beta_r_decomposition(coeffs),
    ]

    print("\n=== Summary ===")
    for q in Q_INPUTS:
        flag = " [extrap]" if _extrap(q) else "        "
        chi_f_val = rpartial.get_chi_f(q)
        mf_val, _ = rpartial.get_remnant(q)
        brp = rpartial.beta_r_physical(mf_val, chi_f_val, q)
        print(f"  q={q:<4}{flag}  chi_f={chi_f_val:.4f}  "
              f"beta_r_phys={brp:.4f}  mathcalE={errors[q]:.4e}")

    print("\nPlots:", PLOT_DIR)
    for p in all_paths:
        print(" ", p.name)


if __name__ == "__main__":
    main()
