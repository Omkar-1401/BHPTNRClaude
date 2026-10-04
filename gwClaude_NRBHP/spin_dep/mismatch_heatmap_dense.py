"""Dense E(q, chi1) heatmaps for all four spin models -- 20 q x 20 chi1.

The dense counterpart of the coarse heatmaps in plot_spin_chi_overlays.py (which only
re-used the 4 x {4,5} overlay grid).  Same per-point number: each model's own plot-module
curve() -- same coefficients, same t0 scan, same full-window mismatch -- evaluated on
    q    in linspace(2, 8, 20)        (q < 3 is extrapolated in nu AND below the surrogate)
    chi1 in linspace(-0.6, 0.8, 20)   (the spin fits' training span)
Points are farmed out with a fork-based Pool, one model at a time (so no two workers ever
build the same (q, chi1) cache file at once).

Outputs, per model, in Agentic_spin_plots/<model>/:
    <model>_mismatch_grid_dense.json     the grid (written as soon as the model finishes)
    <model>_mismatch_heatmap_dense.{pdf,png}
All four heatmaps share one log colour scale.

Usage:
    python mismatch_heatmap_dense.py                    # compute all four, then plot
    python mismatch_heatmap_dense.py --models physical  # subset
    python mismatch_heatmap_dense.py --plot-only        # redraw from the saved JSONs
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import plot_spin_chi_overlays as O

QS = np.linspace(2.0, 8.0, 20)
CHIS = np.linspace(-0.6, 0.8, 20)

_CURVE = None                        # set in the parent before fork; inherited by workers


def _point(qc):
    q, c = qc
    try:
        return float(_CURVE(float(q), float(c))[3])
    except Exception as exc:         # one bad point must not kill the whole grid
        print(f"  [fail] q={q:.3f} chi1={c:+.3f}: {exc!r}", flush=True)
        return float("nan")


def grid_path(key):
    folder = O.MODELS[key][0]
    return O.PLOTS / folder / f"{folder}_mismatch_grid_dense.json"


def compute(key, nproc):
    global _CURVE
    folder, loader, _ = O.MODELS[key]
    print(f"\n{'='*74}\n{key}: {len(QS)}x{len(CHIS)} grid on {nproc} procs\n{'='*74}",
          flush=True)
    _CURVE, _, _ = loader()
    pts = [(q, c) for q in QS for c in CHIS]
    t0 = time.time()
    with mp.get_context("fork").Pool(nproc) as pool:
        E = pool.map(_point, pts, chunksize=1)
    E = np.array(E).reshape(len(QS), len(CHIS))
    print(f"  done in {time.time()-t0:.0f}s   min {np.nanmin(E):.2e}  "
          f"median {np.nanmedian(E):.2e}  max {np.nanmax(E):.2e}  "
          f"nan {int(np.isnan(E).sum())}", flush=True)
    p = grid_path(key)
    p.write_text(json.dumps({"q": QS.tolist(), "chi1": CHIS.tolist(),
                             "E": E.tolist()}, indent=1))
    print("wrote", p, flush=True)


def plot(keys):
    grids = {k: json.loads(grid_path(k).read_text()) for k in keys if grid_path(k).exists()}
    if not grids:
        print("no dense grids found")
        return
    allE = np.concatenate([np.ravel(g["E"]) for g in grids.values()]).astype(float)
    allE = allE[np.isfinite(allE) & (allE > 0)]
    norm = LogNorm(10 ** np.floor(np.log10(allE.min())),
                   10 ** np.ceil(np.log10(allE.max())))
    for key, g in grids.items():
        folder, _, title = O.MODELS[key]
        q, c = np.array(g["q"]), np.array(g["chi1"])
        E = np.array(g["E"], float).T                  # rows chi1, cols q
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        im = ax.pcolormesh(q, c, E, norm=norm, cmap="viridis", shading="nearest")
        cs = ax.contour(q, c, E, levels=[1e-3, 1e-2], colors="w",
                        linewidths=1.0, linestyles=["-", "--"])
        ax.clabel(cs, fmt={1e-3: r"$10^{-3}$", 1e-2: r"$10^{-2}$"}, fontsize=8)
        ax.axvline(O.Q_TRAIN_LO, color="r", ls="--", lw=1.2)
        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$\chi_1$")
        fig.colorbar(im, ax=ax, label=r"$\mathcal{E}$")
        ax.set_title(f"{title}\n" + r"$\mathcal{E}(q,\chi_1)$, "
                     f"{len(q)}x{len(c)} grid  (red dashed: q<3 extrapolated; "
                     "white: 1e-3 solid, 1e-2 dashed)", fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = O.PLOTS / folder / f"{folder}_mismatch_heatmap_dense.{ext}"
            fig.savefig(p)
            print("wrote", p, flush=True)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(O.MODELS), choices=list(O.MODELS))
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE",
                                                                    os.cpu_count())))
    args = ap.parse_args()
    if not args.plot_only:
        for key in args.models:
            compute(key, args.nproc)
    mpl.rcdefaults()
    mpl.rcParams["text.usetex"] = False
    plot(args.models)


if __name__ == "__main__":
    main()
