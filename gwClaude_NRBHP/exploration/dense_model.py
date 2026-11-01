"""Dense E(q, chi1) heatmap for the pass-39 L4 leader, on the grid of spin_dep/mismatch_heatmap_dense.py:
    q in linspace(2, 8, 20)  (q < 3 extrapolated),  chi1 in linspace(-0.6, 0.8, 20).
Same per-point number as every pass-36 score (P6._full; full-window E, t0 searched, phi0 analytic).  No fitting.
The stored T96 is reproduced on T96[::8] first.  The interior of this grid is H270, so those 270 values must equal
plots_p39/p39_scores.json (checked).
    python p39_dense.py [--plot-only]   -> plots_p39/p39_mismatch_grid_dense.json, p39_mismatch_heatmap_dense.{pdf,png}
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PLOT_ONLY = "--plot-only" in sys.argv
QS = np.linspace(2.0, 8.0, 20); CHIS = np.linspace(-0.6, 0.8, 20)
MODEL = [a for a in sys.argv[1:] if not a.startswith("--")][0]
import modelsetup                               # noqa: E402
_M = modelsetup.load(MODEL) if not PLOT_ONLY else None
OUT = HERE / ("plots_" + MODEL); TAG = MODEL; GRID = OUT / f"{TAG}_mismatch_grid_dense.json"
if not PLOT_ONLY:
    P6 = _M["P6"]; R = _M["R"]; TH = _M["TH"]


def _full(k):
    try:
        return P6._full((TH, k))
    except Exception as exc:
        print(f"  [fail] q={k[0]:.3f} chi1={k[1]:+.3f}: {exc!r}", flush=True); return float("nan")


def compute():
    chk = P6.T96[::8]
    pts = [(float(q), float(c)) for q in QS for c in CHIS]
    with mp.get_context("fork").Pool(96) as pool:
        e = np.array(pool.map(_full, chk))
        E = np.array(pool.map(_full, pts, chunksize=1)).reshape(len(QS), len(CHIS))
    r = e / np.array([R["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"reproduce stored T96: {r.min():.4f}..{r.max():.4f}", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    h = json.loads((OUT / f"{TAG}_scores.json").read_text())["H270"]
    d = [E[i, j] / h[f"{q:.6f}_{c:+.4f}"] for i, q in enumerate(QS) for j, c in enumerate(CHIS) if f"{q:.6f}_{c:+.4f}" in h]
    print(f"interior = H270: {len(d)} points, ratio {min(d):.4f}..{max(d):.4f}", flush=True)
    print(f"dense: min {np.nanmin(E):.2e} median {np.nanmedian(E):.2e} max {np.nanmax(E):.2e} nan {int(np.isnan(E).sum())}", flush=True)
    GRID.write_text(json.dumps({"q": QS.tolist(), "chi1": CHIS.tolist(), "E": E.tolist()}, indent=1))


LABELS = {"p40": "pass 40 (18 coef)", "p40lean": "pass 40 lean (16 coef, OT onset)", "p48": "pass 48 (18 coef, closed form)", "p48lean": "pass 48 lean (16 coef, closed form, OT onset)", "p49b12": "pass 49b lean-12 (12 coef)"}


def plot():
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcdefaults(); matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    g = json.loads(GRID.read_text()); q, c = np.array(g["q"]), np.array(g["chi1"]); E = np.array(g["E"], float).T
    norm = LogNorm(1e-4, 1e-1)          # the installed models' dense heatmaps use this scale; below 1e-4 saturates
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.pcolormesh(q, c, E, norm=norm, cmap="viridis", shading="nearest")
    cs = ax.contour(q, c, E, levels=[1e-4, 1e-3, 1e-2], colors="w", linewidths=1.0, linestyles=[":", "-", "--"])
    ax.clabel(cs, fmt={1e-4: r"$10^{-4}$", 1e-3: r"$10^{-3}$", 1e-2: r"$10^{-2}$"}, fontsize=8)
    ax.axvline(3.0, color="r", ls="--", lw=1.2)
    ax.set_xlabel(r"$q$"); ax.set_ylabel(r"$\chi_1$"); fig.colorbar(im, ax=ax, label=r"$\mathcal{E}$", extend="min")
    ax.set_title(LABELS[MODEL] + "\n" + r"$\mathcal{E}(q,\chi_1)$, " + f"{len(q)}x{len(c)} grid  (red dashed: q<3 "
                 "extrapolated; white: 1e-4 dotted, 1e-3 solid, 1e-2 dashed)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{TAG}_mismatch_heatmap_dense.{ext}", dpi=200)
    print(f"wrote {OUT.name}/{TAG}_mismatch_heatmap_dense.{{pdf,png}}", flush=True)


if __name__ == "__main__":
    if not PLOT_ONLY:
        compute()
    plot()
