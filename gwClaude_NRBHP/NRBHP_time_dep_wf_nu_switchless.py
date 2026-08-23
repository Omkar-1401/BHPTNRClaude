"""
Plotting script for the wf_nu_switchless model.

Run fit_scaling_wf_nu_switchless.py first.
Plots written to Agentic_plots/wf_nu_switchless/ (always saved).

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]  — q=2 is the extrapolation test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTutils"))

import bhpt_utils
import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_wf_nu_switchless as wfsl

mpl.rcdefaults()
mpl.rcParams["text.usetex"] = False   # bhpt_utils re-enables usetex; latex not installed

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
MD_PATH  = ROOT / "scaling_wf_nu_switchless.md"
PLOT_DIR = ROOT / "Agentic_plots" / "wf_nu_switchless"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def read_markdown_coefficients(md_path: Path) -> dict[str, np.ndarray]:
    text    = md_path.read_text()
    start   = text.index("## Selected Master Coefficients")
    lines   = text[start:].splitlines()
    table_lines, in_table = [], False
    for line in lines:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            table_lines.append(s); in_table = True
        elif in_table:
            break
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(c.replace(":", "").replace("-", "") == "" for c in cells):
            continue
        rows.append(cells)
    coeffs = {}
    for row in rows[1:]:
        coeffs[row[0].strip()] = np.array([float(v) for v in row[2:]], dtype=float)
    missing = [n for n in wfsl.PARAM_NAMES if n not in coeffs]
    if missing:
        raise ValueError(f"Missing coefficient rows: {missing}")
    return {n: coeffs[n] for n in wfsl.PARAM_NAMES}


def build_model(q, coeffs):
    qdep.generate_and_cache_waveform(q)
    case   = wfsl.load_case(q, source_stride=3, nr_stride=5)
    params = wfsl.constrained_master_params(q, coeffs)
    params, _ = wfsl.polish_nuisance(params, case)
    ev = wfsl.evaluate_model_switchless(
        params, case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"], case["losses"])
    return {"eval": ev, "params": params, "case": case}


def _extrap(q): return q < 3.0 or q > 8.0
def _tag(q):
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if _extrap(q) else "")
def _title(q, error=float("nan")):
    ex  = " [extrapolation]" if _extrap(q) else ""
    es  = f";  mathcalE = {error:.3g}" if np.isfinite(error) else ""
    return f"wf-nu-switchless  q={q}{ex};  nu={wfsl.get_nu(q):.4f}{es}"


def save_waveform_plot(q, common_t, h_nr, h_model, error):
    split = -200.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2]})
    for ax, xlim in [(ax1, (-1000, split)), (ax2, (split, 100))]:
        ax.plot(common_t, np.real(h_nr),    label="NR 22 mode")
        ax.plot(common_t, np.real(h_model), label="BHPT switchless", alpha=0.8)
        ax.set_xlim(*xlim); ax.grid(True); ax.set_xlabel("t/M")
    ax1.legend(fontsize=8)
    fig.suptitle(_title(q, error)); fig.tight_layout()
    path = PLOT_DIR / f"switchless_{_tag(q)}_waveform.pdf"
    fig.savefig(path); plt.close(fig); return path


def save_parameter_plot(q, tau, alpha, beta, e_hat, j_hat):
    ab    = bhpt_utils.all_alpha_beta_1D_scaling_params(q, [creative.MODE])
    label = f"wf-nu-switchless  nu={wfsl.get_nu(q):.4f}"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(tau, alpha, label=label)
    axes[0].axhline(ab["alpha_l2m2"], color="k", ls="--", label="BHPTNRSurrogate alpha")
    axes[0].set_xlim(-1000, 100); axes[0].grid(True)
    axes[0].set_xlabel("t_NR / M"); axes[0].set_ylabel("alpha(t)"); axes[0].legend(fontsize=7)
    axes[1].plot(tau, beta, label=label)
    axes[1].axhline(ab["beta"], color="k", ls="--", label="BHPTNRSurrogate beta")
    axes[1].set_xlim(-1000, 100); axes[1].grid(True)
    axes[1].set_xlabel("t_NR / M"); axes[1].set_ylabel("beta(t)"); axes[1].legend(fontsize=7)
    si = np.argsort(e_hat)
    axes[2].plot(e_hat[si], alpha[si], lw=1, label=label)
    axes[2].set_xlabel("Ehat (wf-flux)"); axes[2].set_ylabel("alpha")
    axes[2].set_title("alpha vs Ehat"); axes[2].grid(True)
    fig.suptitle(_title(q)); fig.tight_layout()
    path = PLOT_DIR / f"switchless_{_tag(q)}_params.pdf"
    fig.savefig(path); plt.close(fig); return path


def save_zoomed_plot(q, common_t, h_nr, h_model):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr),    label="NR 22 mode")
    ax.plot(common_t, np.real(h_model), label="BHPT switchless", alpha=0.8)
    ax.set_xlim(-100, 100); ax.grid(True); ax.set_xlabel("t/M"); ax.legend(fontsize=8)
    ax.set_title(_title(q)); fig.tight_layout()
    path = PLOT_DIR / f"switchless_{_tag(q)}_zoomed.pdf"
    fig.savefig(path); plt.close(fig); return path


def save_coupling_vs_q(coeffs):
    q_fine = np.linspace(1.5, 9.0, 250)
    names  = ["alpha_E", "alpha_J", "beta_E", "beta_J"]
    fig, axes = plt.subplots(1, len(names), figsize=(15, 4))
    for ax, name in zip(axes, names):
        ax.plot(q_fine, [wfsl.eval_poly(float(q), coeffs[name]) for q in q_fine], lw=1.5)
        ax.axvline(3.0, color="gray", ls="--", lw=0.8, label="train boundary")
        ax.axvline(8.0, color="gray", ls="--", lw=0.8)
        ax.axvline(2.0, color="r", ls=":", lw=1.0, label="q=2 extrap")
        ax.axhline(0.0, color="b", ls=":", lw=0.8, label="PP anchor=0")
        ax.set_xlabel("q"); ax.set_ylabel(name); ax.set_title(f"{name}(q)")
        ax.grid(True); ax.legend(fontsize=7)
    fig.suptitle("wf-nu-switchless coupling parameters vs q"); fig.tight_layout()
    path = PLOT_DIR / "switchless_couplings_q.pdf"
    fig.savefig(path); plt.close(fig); return path


def save_alpha_beta_overlay(results):
    """All q alpha(t)/beta(t) overlaid — the key shape comparison."""
    cols = {2.0: "C3", 3.0: "C0", 5.0: "C2", 8.0: "C4"}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
    for q in Q_INPUTS:
        r = results[q]; ev = r["eval"]; case = r["case"]
        tau = ev["tau"]; t_nr = case["t_nr"]
        m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
        t = t_nr[m]; lw = 2.2 if q == 2 else 1.3
        axa.plot(t, np.interp(t, tau, ev["alpha"]), color=cols[q], lw=lw,
                 label=f"q={q:.0f}{' [extrap]' if q == 2 else ''}  E={ev['error']:.2e}")
        axb.plot(t, np.interp(t, tau, ev["beta"]), color=cols[q], lw=lw, label=f"q={q:.0f}")
    for ax, yl, tl in [(axa, "alpha(t)", "alpha(t) — master"), (axb, "beta(t)", "beta(t) — master")]:
        ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80); ax.grid(True)
        ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl); ax.set_title(tl); ax.legend(fontsize=8)
    fig.suptitle("wf-nu-switchless: master alpha/beta across q"); fig.tight_layout()
    path = PLOT_DIR / "switchless_alpha_beta_overlay.pdf"
    fig.savefig(path); plt.close(fig); return path


def main():
    coeffs  = read_markdown_coefficients(MD_PATH)
    results = {}
    errors  = {}
    all_paths = []

    for q in Q_INPUTS:
        print(f"\n--- q={q}{' [EXTRAPOLATION]' if _extrap(q) else ''} ---", flush=True)
        r = build_model(q, coeffs)
        ev = r["eval"]; params = r["params"]; case = r["case"]
        errors[q] = ev["error"]; results[q] = r
        print(f"  nu = {wfsl.get_nu(q):.4f}")
        for i, name in enumerate(wfsl.PARAM_NAMES):
            print(f"  {name:<9} = {params[i]:.6g}")
        print(f"  mathcalE = {ev['error']:.6g}")
        common_t = case["t_nr"][ev["common"]]
        all_paths += [
            save_waveform_plot(q, common_t, ev["h_ref"], ev["h_model"], ev["error"]),
            save_parameter_plot(q, ev["tau"], ev["alpha"], ev["beta"], ev["e_hat"], ev["j_hat"]),
            save_zoomed_plot(q, common_t, ev["h_ref"], ev["h_model"]),
        ]

    all_paths += [save_coupling_vs_q(coeffs), save_alpha_beta_overlay(results)]

    print("\n=== Summary ===")
    for q in Q_INPUTS:
        flag = " [extrap]" if _extrap(q) else "        "
        print(f"  q={q:<4}{flag}  nu={wfsl.get_nu(q):.4f}  mathcalE={errors[q]:.4e}")
    print(f"\nPlots: {PLOT_DIR}")
    for p in all_paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
