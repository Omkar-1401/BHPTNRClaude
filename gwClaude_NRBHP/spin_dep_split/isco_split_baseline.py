"""ISCO-split mismatch diagnostic for the four installed spin models (user design, 2026-10-03).

The NR window is cut at t_isco into an inspiral and a merger-ringdown (MR) segment, scored separately:
    E_insp = ||r - g||^2 / (2 ||r||^2)   over NR t in [-5000.1, t_isco)
    E_MR   = same                         over NR t in [t_isco, +100]
r = NRHybSur3dq8 (2,2), g = the model's calibrated waveform with ONE shared alignment: the t0 and
phi0 that minimise the model's usual full-window mathcalE (so E_full reproduces the stored values).

t_isco (one per (q, chi1), the same for every model):
  1. on the ppBHPT waveform, the first time before its |h22| peak where the (2,2) frequency
     reaches 2 Omega_ISCO(chi1), Omega_ISCO = 1/(r_isco^(3/2) + chi1) in m1 units, r_isco the
     Bardeen-Press-Teukolsky prograde/retrograde Kerr ISCO of the primary (chi1 signed);
     linear interpolation between samples;
  2. carried to NR time by the leading-order map t_NR - t_peak,NR = X1^(6/5) (t_pp - t_peak,pp).
Flag: crossings inside the 2d surrogate's join window [-9, -5] m1 (its frequency is not smooth there).

Identity checked per case: E_insp ||r||^2_insp + E_MR ||r||^2_MR = E_full ||r||^2 (exact).

  python isco_split_baseline.py --compute    (compute node; reads caches only)
  python isco_split_baseline.py --plot -> spin_dep_split/Agentic_spin_split_plots/baseline/isco_split_<model>.{png,pdf}
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent                     # spin_dep_split/
sys.path.insert(0, str(HERE.parent / "spin_dep"))
import plot_spin_diagnostics as D            # get_case(), calibrated(), peak(), NAME, PLOTS, gate data
F, S = D.F, D.S

OUT = HERE / "results" / "isco_split_baseline.json"
JOIN = (-9.0, -5.0)


def kerr_isco(chi):
    """r_isco and orbital Omega_isco (units of m1) for aligned (chi > 0) / anti-aligned spin."""
    a = abs(chi)
    z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = np.sqrt(3 * a * a + z1 * z1)
    r = 3 + z2 - np.sign(chi) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    return float(r), float(1.0 / (r ** 1.5 + chi))


assert abs(kerr_isco(0.0)[0] - 6) < 1e-12 and abs(kerr_isco(1.0)[0] - 1) < 1e-9 \
    and abs(kerr_isco(-1.0)[0] - 9) < 1e-9


def t_isco_nr(q, chi, case):
    t, h = case["t_bhpt"], case["h_bhpt"]
    w = np.abs(np.gradient(np.unwrap(np.angle(h)), t))
    tpk_pp, _ = D.peak(t, h)
    _, om = kerr_isco(chi)
    ipk = int(np.argmax(np.abs(h)))
    i = np.flatnonzero(w[:ipk] >= 2 * om)
    if not len(i):
        raise RuntimeError(f"ppBHPT never reaches its ISCO frequency: q={q} chi={chi}")
    i = i[0]
    t_pp = t[i - 1] + (2 * om - w[i - 1]) * (t[i] - t[i - 1]) / (w[i] - w[i - 1])
    tpk_nr, _ = D.peak(case["t_nr"], case["h_nr"])
    X1 = q / (1 + q)
    return dict(t_isco_pp=float(t_pp), t_peak_pp=float(tpk_pp), t_peak_nr=float(tpk_nr),
                t_isco=float(tpk_nr + X1 ** 1.2 * (t_pp - tpk_pp)),
                in_join_window=bool(JOIN[0] <= t_pp <= JOIN[1]))


def work(task):
    model, q, chi = task
    case = D.get_case(q, chi, "grid")
    ti = t_isco_nr(q, chi, case)
    e, tc, r, g, n1 = D.calibrated(model, q, chi, case)
    res = np.abs(r - g) ** 2; nr = np.abs(r) ** 2
    ins = tc < ti["t_isco"]; mr = ~ins
    e_in = float(res[ins].sum() / (2 * nr[ins].sum())); e_mr = float(res[mr].sum() / (2 * nr[mr].sum()))
    assert abs((e_in * nr[ins].sum() + e_mr * nr[mr].sum()) / (e * n1) - 1) < 1e-9
    return dict(model=model, q=q, chi1=chi, E_full=e, E_insp=e_in, E_MR=e_mr,
                norm_frac_MR=float(nr[mr].sum() / n1), err_frac_MR=float(res[mr].sum() / res.sum()), **ti)


def compute(nproc):
    tasks = [(m, q, c) for m in D.NAME for q in S.TRAIN_Q for c in S.CHIS]
    for m, q, c in tasks[:len(S.TRAIN_Q) * len(S.CHIS)]:
        D.get_case(q, c, "grid")                                  # cache checks before forking
    with get_context("fork").Pool(nproc) as pool:
        res = pool.map(work, tasks, chunksize=1)
    bad = 0
    for m in D.NAME:                                              # reproduction gate (E_full)
        st = json.loads(Path(F.STORED[m]).read_text())["mismatches"]
        rs = np.array([r["E_full"] / st[next(k for k in st if abs(float(k.split("_")[0]) - r["q"]) < 1e-3
                                              and abs(float(k.split("_")[1]) - r["chi1"]) < 1e-6)]
                       for r in res if r["model"] == m])
        ok = np.all(np.abs(rs - 1) <= 0.01); bad += not ok
        print(f"GATE {m}: ratio {rs.min():.5f}..{rs.max():.5f} {'ok' if ok else 'FAIL'}", flush=True)
    if bad:
        sys.exit("gate failed")
    OUT.write_text(json.dumps(dict(note=__doc__.split("\n\n")[1], results=res), indent=1))
    jw = sorted({(r["q"], r["chi1"]) for r in res if r["in_join_window"]})
    print("crossings in the join window:", jw if jw else "none")
    print("wrote", OUT)


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()                       # gw_remnant's plotter changes rcParams on import (CLAUDE.md)
    R = json.loads(OUT.read_text())["results"]
    qs, cs = sorted({r["q"] for r in R}), sorted({r["chi1"] for r in R})

    def grid(rows, key, f=lambda v: v):
        Z = np.full((len(cs), len(qs)), np.nan)
        for r in rows:
            Z[cs.index(r["chi1"]), qs.index(r["q"])] = f(r[key])
        return Z

    for m, name in D.NAME.items():
        rows = [r for r in R if r["model"] == m]
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
        Zi, Zm = grid(rows, "E_insp", np.log10), grid(rows, "E_MR", np.log10)
        lo, hi = np.nanmin([Zi, Zm]), np.nanmax([Zi, Zm])
        for ax, Z, ttl in ((axs[0], Zi, r"inspiral: $\log_{10}\mathcal{E}$ on $[-5000,\,t_{\rm isco})$"),
                           (axs[1], Zm, r"merger-ringdown: $\log_{10}\mathcal{E}$ on $[t_{\rm isco},\,+100]$")):
            im = ax.imshow(Z, origin="lower", cmap="viridis", vmin=lo, vmax=hi, aspect="auto")
            for i in range(len(cs)):
                for j in range(len(qs)):
                    ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if Z[i, j] < lo + 0.55 * (hi - lo) else "black")
            ax.set_title(ttl, fontsize=10); fig.colorbar(im, ax=ax, shrink=0.9)
        T = grid(rows, "t_isco")
        im = axs[2].imshow(T, origin="lower", cmap="magma", aspect="auto")
        for i in range(len(cs)):
            for j in range(len(qs)):
                axs[2].text(j, i, f"{T[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if T[i, j] < np.nanmin(T) + 0.6 * np.ptp(T[~np.isnan(T)]) else "black")
        axs[2].set_title(r"$t_{\rm isco}$ on the NR clock (M; NR peak at about 0)", fontsize=10)
        fig.colorbar(im, ax=axs[2], shrink=0.9)
        for ax in axs:
            ax.set_xticks(range(len(qs))); ax.set_xticklabels([f"{q:.2f}" for q in qs], rotation=60, fontsize=8)
            ax.set_yticks(range(len(cs))); ax.set_yticklabels([f"{c:+.1f}" for c in cs], fontsize=8)
            ax.set_xlabel("q"); ax.set_ylabel(r"$\chi_1$")
        fig.suptitle(f"{name}: mismatch split at the primary's Kerr ISCO (installed coefficients, one shared "
                     r"$t_0,\phi_0$; each segment normalised by its own NR norm)", fontsize=11)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(HERE / "Agentic_spin_split_plots" / "baseline" / f"isco_split_{name}.{e}", dpi=150)
        plt.close(fig)
        print("plotted", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true"); ap.add_argument("--plot", action="store_true")
    ap.add_argument("--nproc", type=int, default=100)
    a = ap.parse_args()
    if a.compute:
        compute(a.nproc)
    if a.plot:
        plot()


if __name__ == "__main__":
    main()
