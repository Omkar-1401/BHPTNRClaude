"""
Plotting script for the wf_nu_q_dep model.

Waveform-flux loss coordinates + PP-anchored nu polynomial regression.
Run fit_scaling_wf_nu_q_dep.py first to generate scaling_wf_nu_q_dep.md.

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]  — q=2 is an extrapolation test.
Plots written to Agentic_plots/wf_nu_q_dep/.
"""
from __future__ import annotations

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
import fit_scaling_wf_nu_q_dep as wfnu

mpl.rcdefaults()

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
MD_PATH  = ROOT / "scaling_wf_nu_q_dep.md"
PLOT_DIR = ROOT / "Agentic_plots" / "wf_nu_q_dep"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Coefficient loading from markdown
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
        if all(c.replace(":", "").replace("-", "") == "" for c in cells):
            continue
        rows.append(cells)
    if not rows:
        raise ValueError(f"No coefficient table found in {md_path}")
    # columns: parameter | anchor | c0 | c1 | c2 | [c3]
    coeffs: dict[str, np.ndarray] = {}
    for row in rows[1:]:  # skip header
        name = row[0].strip()
        # Skip anchor column (row[1]); numerical coefficients start at row[2]
        coeffs[name] = np.array([float(v) for v in row[2:]], dtype=float)
    missing = [n for n in wfnu.PARAM_NAMES if n not in coeffs]
    if missing:
        raise ValueError(f"Missing coefficient rows: {missing}")
    return {n: coeffs[n] for n in wfnu.PARAM_NAMES}


# ---------------------------------------------------------------------------
# Build model for a given q
# ---------------------------------------------------------------------------

def build_model(q: float, coeffs: dict[str, np.ndarray]) -> dict:
    qdep.generate_and_cache_waveform(q)
    case = wfnu.load_case(q, source_stride=3, nr_stride=5)
    params = wfnu.constrained_master_params(q, coeffs)
    params, _ = wfnu.polish_nuisance(params, case)
    ev = creative.evaluate_model(
        params,
        case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"],
        case["losses"], min_coverage=wfnu.MIN_COVERAGE,
    )
    return {"eval": ev, "params": params, "case": case}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _extrap(q: float) -> bool:
    return q < 3.0 or q > 8.0


def _tag(q: float) -> str:
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if _extrap(q) else "")


def _title(q: float, error: float = float("nan")) -> str:
    nu_val = wfnu.get_nu(q)
    extrap = " [extrapolation]" if _extrap(q) else ""
    err_str = f";  mathcalE = {error:.3g}" if np.isfinite(error) else ""
    return f"wf-nu  q={q}{extrap};  nu={nu_val:.4f}{err_str}"


def save_waveform_plot(
    q: float, common_t: np.ndarray,
    h_nr: np.ndarray, h_model: np.ndarray, error: float,
) -> Path:
    split = -200.0
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
    path = PLOT_DIR / f"wf_nu_{_tag(q)}_waveform.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_parameter_plot(
    q: float, tau_use: np.ndarray,
    alpha_use: np.ndarray, beta_use: np.ndarray,
    e_hat_use: np.ndarray, j_hat_use: np.ndarray,
) -> Path:
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(q, [creative.MODE])
    nu_val = wfnu.get_nu(q)
    label = f"wf-nu  nu={nu_val:.4f}"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(tau_use, alpha_use, label=label)
    axes[0].axhline(ab["alpha_l2m2"], color="k", ls="--", label="BHPTNRSurrogate α")
    axes[0].set_xlim(-1000, 100); axes[0].grid(True)
    axes[0].set_xlabel("t_NR / M"); axes[0].set_ylabel(r"$\alpha(t)$")
    axes[0].legend(fontsize=7)

    axes[1].plot(tau_use, beta_use, label=label)
    axes[1].axhline(ab["beta"], color="k", ls="--", label="BHPTNRSurrogate β")
    axes[1].set_xlim(-1000, 100); axes[1].grid(True)
    axes[1].set_xlabel("t_NR / M"); axes[1].set_ylabel(r"$\beta(t)$")
    axes[1].legend(fontsize=7)

    sort_idx = np.argsort(e_hat_use)
    axes[2].plot(e_hat_use[sort_idx], alpha_use[sort_idx], lw=1, label=label)
    axes[2].set_xlabel(r"$\hat{E}$  (wf-flux)"); axes[2].set_ylabel(r"$\alpha$")
    axes[2].set_title(r"$\alpha$ vs $\hat{E}$ (waveform flux)")
    axes[2].grid(True)

    fig.suptitle(_title(q))
    fig.tight_layout()
    path = PLOT_DIR / f"wf_nu_{_tag(q)}_params.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_loss_coord_plot(
    q: float, t_bhpt: np.ndarray, losses: dict,
) -> Path:
    """Show Ehat, Jhat, p_loss vs time — lets us verify wf-flux coordinates."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    t_plot = t_bhpt[(t_bhpt >= -1500) & (t_bhpt <= 150)]
    for key, ax, label in [
        ("e_hat",  axes[0], r"$\hat{E}$"),
        ("j_hat",  axes[1], r"$\hat{J}$"),
        ("p_loss", axes[2], r"$p_\mathrm{loss}$"),
    ]:
        vals = losses[key]
        mask = (t_bhpt >= -1500) & (t_bhpt <= 150)
        ax.plot(t_bhpt[mask], vals[mask], lw=1)
        ax.axvline(0.0, color="gray", ls="--", lw=0.8)
        ax.axhline(0.0, color="k",    ls=":",  lw=0.6, alpha=0.5)
        ax.set_xlabel("t / M"); ax.set_ylabel(label)
        ax.set_title(f"{label} (wf flux)")
        ax.grid(True)

    fig.suptitle(f"wf-nu loss coordinates  q={q}  nu={wfnu.get_nu(q):.4f}")
    fig.tight_layout()
    path = PLOT_DIR / f"wf_nu_{_tag(q)}_loss_coords.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_zoomed_plot(
    q: float, common_t: np.ndarray,
    h_nr: np.ndarray, h_model: np.ndarray,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
    ax.plot(common_t, np.real(h_model), label="BHPT calibrated", alpha=0.8)
    ax.set_xlim(-100, 100); ax.grid(True)
    ax.set_xlabel("t/M"); ax.legend(fontsize=8)
    ax.set_title(_title(q))
    fig.tight_layout()
    path = PLOT_DIR / f"wf_nu_{_tag(q)}_zoomed.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_coefficient_table(coeffs: dict[str, np.ndarray]) -> Path:
    degree = len(next(iter(coeffs.values()))) - 1
    columns = ["parameter", "anchor", *[f"c{i}" for i in range(degree + 1)], "value q=5"]
    table_rows = []
    for name in wfnu.PARAM_NAMES:
        anc = wfnu.PP_ANCHORS[name]
        anc_str = f"{anc:.4g}" if anc is not None else "free"
        val = wfnu.eval_poly(5.0, coeffs[name])
        table_rows.append([name, anc_str, *[f"{v:.5g}" for v in coeffs[name]], f"{val:.5g}"])
    fig_h = max(4.0, 0.4 * len(table_rows) + 1.2)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=columns,
                   loc="center", cellLoc="right", colLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.3)
    ax.set_title("wf-nu: polynomial coefficients (nu basis, PP-anchored)")
    fig.tight_layout()
    path = PLOT_DIR / "wf_nu_coefficients.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_coupling_vs_q(coeffs: dict[str, np.ndarray]) -> Path:
    q_fine = np.linspace(1.5, 9.0, 250)
    coupling_names = ["alpha_E", "alpha_J", "beta_r", "beta_L"]

    fig, axes = plt.subplots(1, len(coupling_names), figsize=(14, 4))
    for ax, name in zip(axes, coupling_names):
        vals = [wfnu.eval_poly(float(q), coeffs[name]) for q in q_fine]
        ax.plot(q_fine, vals, lw=1.5)
        ax.axvline(3.0, color="gray", ls="--", lw=0.8, label="training boundary")
        ax.axvline(8.0, color="gray", ls="--", lw=0.8)
        anc = wfnu.PP_ANCHORS[name]
        if anc is not None:
            ax.axhline(anc, color="b", ls=":", lw=0.8, alpha=0.7, label=f"PP anchor={anc}")
        ax.set_xlabel("q"); ax.set_ylabel(name, fontsize=9)
        ax.set_title(f"{name}(q)")
        ax.grid(True); ax.legend(fontsize=7)

    fig.suptitle("wf-nu coupling parameters vs q  (gray dashed = training boundary)")
    fig.tight_layout()
    path = PLOT_DIR / "wf_nu_couplings_q.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_alpha_beta_vs_q(coeffs: dict[str, np.ndarray]) -> Path:
    q_fine = np.linspace(1.5, 9.0, 250)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name, ylabel in [
        (axes[0], "alpha_i", r"$\alpha_i$ (amplitude factor)"),
        (axes[1], "beta_i",  r"$\beta_i$ (inspiral time-stretch)"),
        (axes[2], "beta_r",  r"$\beta_r$ (ringdown time-stretch)"),
    ]:
        vals = [wfnu.eval_poly(float(q), coeffs[name]) for q in q_fine]
        ax.plot(q_fine, vals, lw=1.5, label="polynomial")
        ax.axvline(3.0, color="gray", ls="--", lw=0.8, label="train boundary")
        ax.axvline(8.0, color="gray", ls="--", lw=0.8)
        anc = wfnu.PP_ANCHORS[name]
        if anc is not None:
            ax.axhline(anc, color="b", ls=":", lw=0.8, label=f"PP anchor={anc}")
        ax.set_xlabel("q"); ax.set_ylabel(ylabel)
        ax.set_title(f"{name}(q)")
        ax.grid(True); ax.legend(fontsize=7)

    fig.suptitle("wf-nu: amplitude/time-stretch parameters vs q")
    fig.tight_layout()
    path = PLOT_DIR / "wf_nu_alpha_beta_q.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_nu_vs_q() -> Path:
    """Show nu=q/(1+q)² vs q with training boundary and PP anchor marked."""
    q_fine = np.linspace(1.0, 10.0, 300)
    nu_fine = np.array([wfnu.get_nu(q) for q in q_fine])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q_fine, nu_fine, lw=1.5)
    ax.axvline(3.0, color="gray", ls="--", lw=0.8, label="training boundary q=3,8")
    ax.axvline(8.0, color="gray", ls="--", lw=0.8)
    ax.axvline(2.0, color="r", ls=":",  lw=1.0, label="extrapolation q=2")
    ax.axhline(0.0, color="b", ls=":", lw=0.8, label="PP anchor nu→0 (q→∞)")
    ax.scatter([2, 3, 8], [wfnu.get_nu(2), wfnu.get_nu(3), wfnu.get_nu(8)],
               color=["r", "gray", "gray"], zorder=5)
    ax.set_xlabel("q"); ax.set_ylabel(r"$\nu = q/(1+q)^2$")
    ax.set_title(r"Regression coordinate $\nu(q)$")
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout()
    path = PLOT_DIR / "wf_nu_nu_vs_q.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    coeffs = read_markdown_coefficients(MD_PATH)
    all_paths: list[Path] = []
    errors: dict[float, float] = {}

    for q in Q_INPUTS:
        extrap_str = " [EXTRAPOLATION]" if _extrap(q) else ""
        print(f"\n--- q={q}{extrap_str} ---", flush=True)

        result = build_model(q, coeffs)
        ev = result["eval"]
        params = result["params"]
        case = result["case"]

        p = creative.unpack(params)
        nu_val = wfnu.get_nu(q)
        errors[q] = ev["error"]

        print(f"  nu       = {nu_val:.4f}")
        for name in wfnu.PARAM_NAMES:
            print(f"  {name:<10} = {p[name]:.6g}")
        print(f"  mathcalE = {ev['error']:.6g}")

        tau_use   = ev["tau"]
        alpha_use = ev["alpha"]
        beta_use  = ev["beta"]
        e_hat_use = ev["e_hat"]
        j_hat_use = ev["j_hat"]
        common_t  = case["t_nr"][ev["common"]]

        all_paths += [
            save_waveform_plot(q, common_t, ev["h_ref"], ev["h_model"], ev["error"]),
            save_parameter_plot(q, tau_use, alpha_use, beta_use, e_hat_use, j_hat_use),
            save_loss_coord_plot(q, case["t_bhpt"], case["losses"]),
            save_zoomed_plot(q, common_t, ev["h_ref"], ev["h_model"]),
        ]

    all_paths += [
        save_coefficient_table(coeffs),
        save_coupling_vs_q(coeffs),
        save_alpha_beta_vs_q(coeffs),
        save_nu_vs_q(),
    ]

    print("\n=== Summary ===")
    for q in Q_INPUTS:
        flag = " [extrap]" if _extrap(q) else "        "
        nu_val = wfnu.get_nu(q)
        print(f"  q={q:<4}{flag}  nu={nu_val:.4f}  mathcalE={errors[q]:.4e}")

    print(f"\nPlots: {PLOT_DIR}")
    for path in all_paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
