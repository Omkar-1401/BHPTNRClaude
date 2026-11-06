"""Figures backing the SUMMARY of README.md, from the stored results only (no refits, no new scoring).

    fig1_v300_ecdf.png     V300 error distribution: dense data (N = 572-714) and the same 96 training cases as pass 40
    fig2_training_size.png V300 median and max vs number of training cases (exact training_size grids)
    fig3_extrap.png        per-case E outside the training range (q < 3, q = 9-10)
    fig4_where_wins.png    log10(E_ML / E_pass40) over (q, chi1) on V300

    python plot_bench.py      -> plots/*.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
TS = HERE.parent / "training_size"
RES = HERE / "results"; OUT = HERE / "plots"; OUT.mkdir(exist_ok=True)
J = lambda f: json.loads((RES / f).read_text())

INK, INK2, GRID, SURF = "#1f1f1e", "#5f5e5a", "#e4e3de", "#fcfcfb"
SER = dict(p40=INK, mlp="#2a78d6", rbf="#eb6834", gpr="#1baf7a", et="#eda100", xgb="#e87ba4")
NAME = dict(p40="pass 40 (hand, 18 coef.)", mlp="MLP, E-weighted", rbf="RBF (PCA)", gpr="GPR (PCA)",
            et="ExtraTrees (PCA)", xgb="XGBoost (PCA)")
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "legend.frameon": False, "lines.linewidth": 2})

V300 = [tuple(map(float, l.split())) for l in (TS / "V300.txt").read_text().split("\n") if l.strip()]
P40 = json.loads((TS / "results" / "p40__G96__t96.json").read_text())
p40_v = np.array([P40["per_case"]["V300"][f"{q:.6f}_{c:+.3f}"] for q, c in V300])
assert len(p40_v) == 300 and np.isclose(np.median(p40_v), P40["stats"]["V300"]["median"])

fin, mlp, extra, ext = J("direct_final.json"), J("nn_mlp.json"), J("nn_mlp_extra.json"), J("extrap_direct.json")
v300 = dict(p40=p40_v, mlp=np.array(mlp["chosen"]["V300_cases"]),
            **{m: np.array(fin[m]["V300_cases"]) for m in ("rbf", "gpr", "et", "xgb")})


def ecdf(ax, e, key, label, ls="-"):
    x = np.sort(e); y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=SER[key], ls=ls, lw=2.6 if key == "p40" else 1.8, label=label)


# ---- fig 1: V300 distributions
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for k in ("p40", "mlp", "rbf", "gpr", "et", "xgb"):
    n = "96" if k == "p40" else "572" if k == "mlp" else "714"
    ecdf(a, v300[k], k, f"{NAME[k]}  (N={n})")
a.set_title("Dense training data (ML N = 572-714; pass 40 N = 96)", loc="left", fontsize=10)
ecdf(b, p40_v, "p40", f"{NAME['p40']}  (T96)")
ecdf(b, np.array(extra["mlpw|T96|V300"]["cases"]), "mlp", "MLP, E-weighted  (exactly T96)")
for m in ("rbf", "gpr"):
    ecdf(b, np.array(ext[f"{m}|96|V300"]["cases"]), m, f"{NAME[m]}  (96 random pool pts)", ls="--")
b.set_title("~96 training cases", loc="left", fontsize=10)
for ax in (a, b):
    ax.set_xscale("log"); ax.set_xlabel("waveform error E on V300 (300 held-out cases)")
    ax.axvline(p40_v.max(), color=INK2, lw=0.8, ls=":")
    ax.legend(loc="upper left", fontsize=8)
a.set_ylabel("fraction of cases with error ≤ E")
b.text(p40_v.max() * 1.1, 0.04, "pass 40 worst case", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(OUT / "fig1_v300_ecdf.png", dpi=150); plt.close(fig)

# ---- fig 2: training-size curves (exact grids G24/G48/T96/G192)
cur = json.loads((TS / "summary.json").read_text())["p40"]["curve"]
rob = J("robust_direct.json")
NS = [24, 48, 96, 192]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
for ax, st in zip(axs, ("median", "max")):
    ax.plot(NS, [cur[str(n)]["V300"][st] for n in NS], "o-", color=SER["p40"], lw=2.6, ms=7, label=NAME["p40"])
    for m, key in (("rbf", "rbf"), ("gpr", "gpr")):
        ax.plot(NS, [rob[f"{m}|{n}|grid"][st] for n in NS], "o-", color=SER[key], ms=6, label=NAME[key])
    ax.plot([96], [extra["mlpw|T96|V300"][st]], "s", color=SER["mlp"], ms=9, label=NAME["mlp"] + " (T96)")
    ax.plot([572], [mlp["chosen"]["V300"][st]], "D", color=SER["mlp"], ms=8, mfc=SURF, mew=2,
            label=NAME["mlp"] + " (572 pool)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(NS + [572]); ax.set_xticklabels(map(str, NS + [572]))
    ax.minorticks_off(); ax.set_xlabel("training cases (same grids for every method)")
    ax.set_title(f"V300 {st} error", loc="left", fontsize=10)
axs[0].set_ylabel("E"); axs[0].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "fig2_training_size.png", dpi=150); plt.close(fig)

# ---- fig 3: extrapolation, per case vs q
fig, axs = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
rows = [("p40", lambda s: ext[f"p40|{s}"]["cases"], "o", NAME["p40"] + " (T96)", True),
        ("mlp", lambda s: extra[f"mlpw|T96|{s}"]["cases"], "s", "MLP, E-weighted (T96)", True),
        ("mlp", lambda s: extra[f"mlpw|572|{s}"]["cases"], "D", "MLP, E-weighted (572)", False),
        ("gpr", lambda s: ext[f"gpr|714|{s}"]["cases"], "^", "GPR (714)", True),
        ("et", lambda s: ext[f"et|714|{s}"]["cases"], "v", "ExtraTrees (714)", True)]
for ax, s, ttl in zip(axs, ("lowq", "vlowq", "highq"), ("2.5 ≤ q < 3", "q < 2.5 (BHPT input extrapolated too)",
                                                       "q = 9, 10 (NR surrogate extrapolated)")):
    q = np.array([k[0] for k in ext["sets"][s]])
    for i, (key, get, mk, lab, filled) in enumerate(rows):
        jit = (i - 2) * (0.012 if s != "highq" else 0.05)
        ax.scatter(q + jit, get(s), marker=mk, s=26, color=SER[key] if filled else SURF, edgecolors=SER[key],
                   linewidths=1.2, label=lab, zorder=3 if key == "p40" else 2, alpha=0.9)
    ax.set_yscale("log"); ax.set_xlabel("q"); ax.set_title(ttl, loc="left", fontsize=10)
axs[0].set_ylabel("E (each point = one (q, χ1) case)"); axs[0].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig3_extrap.png", dpi=150); plt.close(fig)

# ---- fig 4: where ML beats pass 40 on V300
cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a07f", "#b53a1c"])
Q, X = np.array(V300).T
T96 = np.array([tuple(map(float, k.split("_"))) for k in P40["per_case"]["T96"]])
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (e, ttl) in zip(axs, ((np.array(extra["mlpw|T96|V300"]["cases"]), "MLP trained on exactly T96"),
                              (v300["mlp"], "MLP trained on 572 pool cases"))):
    r = np.log10(e / p40_v)
    sc = ax.scatter(Q, X, c=r, cmap=cmap, norm=TwoSlopeNorm(0, -2, 2), s=30, edgecolors=INK2, linewidths=0.3)
    ax.scatter(*T96.T, marker="x", s=14, color=INK, linewidths=0.9, label="T96 training points (pass 40)")
    ax.set_xlabel("q"); ax.set_title(f"{ttl}\nMLP better in {np.mean(r < 0):.0%} of V300 cases", loc="left", fontsize=10)
    ax.grid(False)
axs[0].set_ylabel("χ1"); axs[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.3))
cb = fig.colorbar(sc, ax=axs, shrink=0.9, pad=0.02)
cb.set_label("log10(E_MLP / E_pass40)\nblue: MLP better · red: pass 40 better")
fig.savefig(OUT / "fig4_where_wins.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))

# ---- figs 5-7: alpha, beta of every paradigm vs the NR equal-phase truth (results of ab_compare.py)
abc = json.loads((RES / "ab_compare.json").read_text()) if (RES / "ab_compare.json").exists() else None
if abc:
    cv = np.load(RES / "ab_compare_curves.npz"); s = cv["s"]
    AB_SER = [("p40", SER["p40"], "pass 40 (native α, β)"), ("p40_x", "#8a8984", "pass 40 (from its waveform)"),
              ("abtau_rbf", SER["rbf"], "RBF on α and clock τ (714)"), ("rbf_direct", SER["gpr"], "RBF on ΔlnA, Δφ (714)"),
              ("mlp_T96", SER["mlp"], "MLP (T96)"), ("mlp_572", SER["et"], "MLP (572)")]
    CASES = [("p40_median", "typical case\n(pass 40 median E)"), ("mid", "q ≈ 5.5, χ1 ≈ 0.2"),
             ("retro_highq", "q ≈ 7.5, χ1 ≈ −0.5"), ("mlpT96_vs_p40", "MLP (T96) worst\nrelative to pass 40"),
             ("p40_worst", "pass 40 worst case")]
    XL = (-1500, 60)
    V = abc["keys"]
    tt = lambda n: f"{dict(CASES)[n]}\nq={V[abc['keep'][n]][0]:.2f}, χ1={V[abc['keep'][n]][1]:+.2f}"

    # fig 5: pass 40's fitted alpha, beta against the NR truth in two gauges
    fig, axs = plt.subplots(2, len(CASES), figsize=(16, 6.2), sharex=True)
    for j, (n, _) in enumerate(CASES):
        ins = cv[f"{n}|inside"]
        for r, q_ in enumerate(("alpha", "beta")):
            ax = axs[r, j]
            ax.plot(s[ins], cv[f"{n}|nr|{q_}"][ins], color="#b4b3ab", lw=1.6, ls="--", label="NR truth, phases zeroed at peak")
            ax.plot(s[ins], cv[f"{n}|nr|p40|{q_}"][ins], color=SER["mlp"], lw=1.6, label="NR truth, pass 40's gauge")
            ax.plot(s[ins], cv[f"{n}|p40|{q_}"][ins], color=INK, lw=2.2, label="pass 40 fitted")
            ax.set_xlim(*XL)
            if r == 0: ax.set_title(tt(n), loc="left", fontsize=9)
            if j == 0: ax.set_ylabel({"alpha": "α", "beta": "β = dτ/dt  [M / m1]"}[q_])
            if r == 1: ax.set_xlabel("s = t − t_peak,pp  [m1]")
        lo, hi = np.percentile(np.r_[cv[f"{n}|nr|beta"][ins], cv[f"{n}|p40|beta"][ins]], [1, 99])
        axs[1, j].set_ylim(lo - 0.05, min(hi + 0.1, 2.0))
    axs[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "fig5_p40_alpha_beta.png", dpi=150); plt.close(fig)

    # fig 6: residuals of every model vs the NR truth in its own matched gauge
    fig, axs = plt.subplots(3, len(CASES), figsize=(16, 8.5), sharex=True)
    for j, (n, _) in enumerate(CASES):
        ins = cv[f"{n}|inside"]
        for key, col, lab in AB_SER:
            aN, bN, tN = (cv[f"{n}|nr|{key}|{x}"] for x in ("alpha", "beta", "tau"))
            a_, b_, t_ = (cv[f"{n}|{key}|{x}"] for x in ("alpha", "beta", "tau"))
            m = ins & (s >= XL[0]) & (s <= XL[1])
            dt = t_ - tN; dt = dt - np.median(dt[m])
            lw = 2.2 if key == "p40" else 1.4
            axs[0, j].plot(s[ins], (np.log(a_) - np.log(aN))[ins], color=col, lw=lw, label=lab)
            axs[1, j].plot(s[ins], (b_ / bN - 1)[ins], color=col, lw=lw)
            axs[2, j].plot(s[ins], dt[ins], color=col, lw=lw)
        axs[0, j].set_title(tt(n), loc="left", fontsize=9)
        for r, lim in enumerate((0.3, 0.3, 3.0)):
            axs[r, j].set_ylim(-lim, lim); axs[r, j].set_xlim(*XL)
        axs[2, j].set_xlabel("s = t − t_peak,pp  [m1]")
    for r, lab in enumerate(("ln α − ln α_NR", "β / β_NR − 1", "τ − τ_NR  [M]")):
        axs[r, 0].set_ylabel(lab)
    axs[0, 0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "fig6_alpha_beta_residuals.png", dpi=150); plt.close(fig)

    # fig 7: V300 distributions of the alpha, beta, tau deviations (matched gauge)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, st, lab in zip(axs, ("dlna", "dbeta", "dtau"),
                           ("rms(ln α − ln α_NR)", "rms(β / β_NR − 1)", "rms(τ − τ_NR)  [M]")):
        for key, col, name in AB_SER:
            x = np.sort(abc["metrics"][key + "|g"][st]); y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=col, lw=2.6 if key == "p40" else 1.6, label=name)
        ax.set_xscale("log"); ax.set_xlabel(lab + ", weighted by |h_NR|²")
    axs[0].set_ylabel("fraction of V300 cases"); axs[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig7_alpha_beta_v300.png", dpi=150); plt.close(fig)
    print("wrote figs 5-7")
