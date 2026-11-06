"""Figures + summary table for paradigm E (ML alpha, beta fitted through the map on T96; efit.py, efit_sr.py, efit_lin.py).

    figE1_v300_ecdf.png       V300 E distributions vs pass 40 (networks / symbolic | classic regressors)
    figE2_summary.png         V300 and T96 median / max per method
    figE3_alpha_beta.png      alpha(t), beta(t) on representative V300 cases (each model in its own fitted gauge)
    figE4_pysr_tradeoff.png   PySR complexity vs E (T96 = training objective, V300 = report only)
    figE5_where_wins.png      log10(E_ML / E_pass40) over (q, chi1) on V300 for the best methods
    results/efit_summary.json table used by the page

    python plot_efit.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RES, OUT, TS = HERE / "results", HERE / "plots", HERE.parent / "training_size"
INK, INK2, GRID, SURF = "#1f1f1e", "#5f5e5a", "#e4e3de", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLE = {"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "legend.frameon": False, "lines.linewidth": 2, "font.family": "sans-serif", "text.usetex": False,
                     "mathtext.fontset": "dejavusans"}
plt.rcParams.update(STYLE)

NETS = [("mlp", "MLP (identity start)"), ("gru", "GRU (identity start)"), ("sr", "PySR from MLP (identity start)"),
        ("mlp_t", "MLP (model-free start)"), ("sr_t", "PySR from MLP (model-free start)")]
LINS = [("knn", "KNN"), ("rbf", "RBF"), ("gpr", "GPR"), ("rf", "Random forest"), ("et", "ExtraTrees"),
        ("xgb", "XGBoost"), ("sparse", "Sparse library")]
COL = {k: CAT[i] for i, (k, _) in enumerate(NETS)} | {k: CAT[i] for i, (k, _) in enumerate(LINS)}
LAB = dict(NETS + LINS, p40="pass 40 (hand, 18 coef.)", lean12="lean-12 (hand, 12 coef.)")
REFS = ["p40", "lean12"]                       # hand-built reference models


def load():
    R = {}
    for k, fn in (("mlp", "mlp_T96"), ("gru", "gru_T96"), ("mlp_t", "mlp_T96_tinit")):
        f = RES / f"efit_{fn}.json"
        if f.exists():
            d = json.loads(f.read_text())
            R[k] = dict(T96=d["train"], V300=d["V300"], n=d["n_params"], train_s=d["train_s"] + d.get("pre_s", 0.0))
    for k, fn in (("sr", "sr_T96"), ("sr_t", "sr_T96_from_mlp_T96_tinit")):
        f = RES / f"efit_{fn}.json"
        if f.exists():
            d = json.loads(f.read_text()); c = d["chosen"]
            R[k] = dict(T96=c["T96"], V300=c["V300"], n=c["n_constants"], complexity=c["complexity"],
                        train_s=None, candidates=d["candidates"], expr=(c["expr_alpha"], c["expr_beta"]))
    for k, _ in LINS:
        f = RES / f"efit_lin_{k}.json"
        if f.exists():
            d = json.loads(f.read_text())
            R[k] = dict(T96=d["T96"], V300=d["V300"], n=d.get("n_nonzero", d["n_values"]), train_s=d["train_s"],
                        init_T96=d["init_T96"])
    P = json.loads((TS / "results" / "p40__G96__t96.json").read_text())
    V300 = [tuple(map(float, l.split())) for l in (TS / "V300.txt").read_text().split("\n") if l.strip()]
    pv = np.array([P["per_case"]["V300"][f"{q:.6f}_{c:+.3f}"] for q, c in V300])
    pt = np.array(list(P["per_case"]["T96"].values()))
    R["p40"] = dict(V300=dict(median=float(np.median(pv)), mean=float(pv.mean()), max=float(pv.max()), cases=pv.tolist()),
                    T96=dict(median=float(np.median(pt)), mean=float(pt.mean()), max=float(pt.max())), n=18)
    P = json.loads((TS / "results" / "lean12__G96__t96.json").read_text())     # pass 49b, G96 refit from the t96 start
    lv = np.array([P["per_case"]["V300"][f"{q:.6f}_{c:+.3f}"] for q, c in V300]); lt = np.array(list(P["per_case"]["T96"].values()))
    R["lean12"] = dict(V300=dict(median=float(np.median(lv)), mean=float(lv.mean()), max=float(lv.max()), cases=lv.tolist()),
                       T96=dict(median=float(np.median(lt)), mean=float(lt.mean()), max=float(lt.max())), n=12)
    return R, V300


def curves(R, V300, idx):
    """alpha, beta on S_C for V300[idx]: pass 40 native, mlp, gru, sr, and every lin model (rebuilt + loaded)"""
    import common as C, efit as F, gate, torch
    from direct_targets import _tpeak
    keys = [V300[i] for i in idx]; out = {"s": F.S_C}
    a, b = [], []
    for k in keys:
        c = C.case(k); al, be = gate.ab_fun(k, c)("full"); s = c["t_bhpt"] - _tpeak(c["t_bhpt"], c["h_bhpt"])
        a.append(np.interp(F.S_C, s, al)); b.append(np.interp(F.S_C, s, be))
    out["p40"] = (np.array(a), np.array(b))
    for k, fn in (("mlp", "mlp_T96"), ("gru", "gru_T96"), ("mlp_t", "mlp_T96_tinit")):
        f = F.MOD / f"efit_{fn}.pt"
        if f.exists():
            m = F.GRU() if k == "gru" else F.MLP(); m.load_state_dict(torch.load(f)); out[k] = F.ab_numpy(m, keys)
    for k, fn in (("sr", "sr_T96"), ("sr_t", "sr_T96_from_mlp_T96_tinit")):
        f = RES / f"efit_{fn}_abV300.npz"
        if f.exists():
            z = np.load(f); out[k] = (z["alpha"][idx], z["beta"][idx])
    lin = [k for k, _ in LINS if (F.MOD / f"efit_lin_{k}.pt").exists()]
    if lin:
        import efit_lin as L
        X = F.pfeat(list(C.T96)).numpy(); Y = L.init_targets(list(C.T96))
        for k in lin:
            m, _ = L.build(k, X, Y)
            try:
                m.load_state_dict(torch.load(F.MOD / f"efit_lin_{k}.pt"))
            except RuntimeError as e:                              # saved under an older parametrisation
                print(f"skip curves for {k}: {str(e)[:80]}"); continue
            out[k] = F.ab_numpy(m, keys)
    return out


def roughness(a):
    """rms second difference of ln(.) along the S_C index (pre-merger part), per curve"""
    x = np.log(np.clip(a[:, :1500], 1e-12, None)); return np.sqrt(np.mean(np.diff(x, 2, axis=1) ** 2, axis=1))


def ecdf(ax, e, col, lab, lw=1.7, ls="-"):
    x = np.sort(e); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, lw=lw, ls=ls, label=lab)


def main():
    R, V300 = load()
    have = [k for k, _ in NETS + LINS if k in R]
    print("results:", have)
    pv = np.array(R["p40"]["V300"]["cases"]); V = np.array(V300)

    # E1
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax, group, ttl in zip(axs, (NETS, LINS), ("Neural networks and symbolic regression", "Classic regressors (refined through E)")):
        ecdf(ax, pv, INK, LAB["p40"], lw=2.8)
        ecdf(ax, np.array(R["lean12"]["V300"]["cases"]), INK2, LAB["lean12"], lw=2.4, ls="--")
        for k, lab in group:
            if k in R:
                ecdf(ax, np.array(R[k]["V300"]["cases"]), COL[k], lab)
        ax.set_xscale("log"); ax.set_xlabel("waveform error E on V300"); ax.axvline(pv.max(), color=INK2, lw=0.8, ls=":")
        ax.set_title(ttl + " — all trained on T96", loc="left", fontsize=10); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    axs[0].set_ylabel("fraction of V300 cases with error ≤ E")
    fig.tight_layout(); fig.savefig(OUT / "figE1_v300_ecdf.png", dpi=150); plt.close(fig)

    # E2
    order = REFS + sorted(have, key=lambda k: R[k]["V300"]["median"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.42 * len(order) + 1.4), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, st in zip(axs, ("median", "max")):
        for yy, k in zip(y, order):
            col = {"p40": INK, "lean12": INK2}.get(k) or COL[k]
            ax.plot([R[k]["T96"][st]], [yy], "o", mfc=SURF, mec=col, mew=1.6, ms=7)
            ax.plot([R[k]["V300"][st]], [yy], "o", color=col, ms=7)
        ax.axvline(R["p40"]["V300"][st], color=INK, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_title(f"{st} E   (filled: V300, open: T96 training)", loc="left", fontsize=10)
    axs[0].set_yticks(y); axs[0].set_yticklabels([LAB[k] for k in order])
    fig.tight_layout(); fig.savefig(OUT / "figE2_summary.png", dpi=150); plt.close(fig)

    # E3: alpha, beta curves on representative V300 cases
    near = lambda q, c: int(np.argmin((V[:, 0] - q) ** 2 / 25 + (V[:, 1] - c) ** 2 / 1.96))
    sel = dict(typical=int(np.argsort(pv)[150]), high_spin=near(3.5, 0.75), retrograde=near(7.5, -0.55), worst_p40=int(np.argmax(pv)))
    idx = list(sel.values()); cv = curves(R, V300, idx)
    plt.rcParams.update(STYLE)                     # the physics modules imported by curves() change the matplotlib style
    z = np.load(RES / "lean12_E3_curves.npz")      # from training_size/lean12_curves.py (own process, gated)
    cv["lean12"] = tuple(np.array([np.interp(cv["s"], z[f"{n}_s"], z[f"{n}_{w}"]) for n in sel]) for w in ("alpha", "beta"))
    bestlin = min([k for k, _ in LINS if k in R], key=lambda k: R[k]["T96"]["mean"], default=None)   # lowest T96 (info only)
    show = REFS + [k for k in ("mlp", "mlp_t", "sr_t", "knn", "gpr") if k in cv]      # fixed display set
    C3 = dict(mlp=CAT[0], mlp_t=CAT[3], sr_t=CAT[2], knn=CAT[6], gpr=CAT[4])
    s = cv["s"]; m = (s > -1500) & (s < 80)
    fig, axs = plt.subplots(2, len(idx), figsize=(16, 6.8), sharex=True)
    for j, (name, i) in enumerate(sel.items()):
        for k in show:
            col, lw, ls = dict(p40=(INK, 2.4, "-"), lean12=(INK2, 2.0, "--")).get(k, (C3.get(k), 1.5, "-"))
            axs[0, j].plot(s[m], cv[k][0][j][m], color=col, lw=lw, ls=ls, label=LAB[k])
            axs[1, j].plot(s[m], cv[k][1][j][m], color=col, lw=lw, ls=ls)
        SH = dict(p40="p40", lean12="lean12", mlp="MLP-id", mlp_t="MLP-mf", sr_t="PySR-mf", knn="KNN", gpr="GPR")
        es = [f"{SH[k]} {(pv[i] if k == 'p40' else R[k]['V300']['cases'][i]):.1e}" for k in show if k in R]
        axs[0, j].set_title(f"{name.replace('_', ' ')}: q={V[i, 0]:.2f}, χ1={V[i, 1]:+.2f}\nE  " + " · ".join(es[:3])
                            + "\n    " + " · ".join(es[3:6]) + ("\n    " + " · ".join(es[6:]) if es[6:] else ""), loc="left", fontsize=8)
        axs[1, j].set_xlabel("s = t − t_peak,pp  [m1]")
        lo, hi = np.nanpercentile(np.concatenate([cv[k][1][j][m] for k in show]), [0.5, 99.5]); axs[1, j].set_ylim(lo - 0.05, hi + 0.05)
    for ax in axs[0]:
        ax.set_yscale("log")
    axs[0, 0].set_ylabel("α (log scale)"); axs[1, 0].set_ylabel("β  [M / m1]"); axs[0, 0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "figE3_alpha_beta.png", dpi=150); plt.close(fig)

    # roughness over the representative cases (smoothness of alpha, beta)
    rough = {k: dict(alpha=float(np.median(roughness(cv[k][0]))), beta=float(np.median(roughness(cv[k][1])))) for k in cv if k != "s"}

    # E4: PySR trade-off
    if "sr" in R or "sr_t" in R:
        fig, ax = plt.subplots(figsize=(7.5, 4.4)); cx = []
        for k in ("sr", "sr_t"):
            if k not in R:
                continue
            cand = R[k]["candidates"]; x = [c["complexity"] for c in cand]; cx += x
            ax.plot(x, [c["T96"]["median"] for c in cand], "o--", color=COL[k], mfc=SURF, mew=1.6, label=f"{LAB[k]}: T96 median")
            ax.plot(x, [c["V300"]["median"] for c in cand], "o-", color=COL[k], label=f"{LAB[k]}: V300 median")
            ax.plot(x, [c["V300"]["max"] for c in cand], "s:", color=COL[k], label=f"{LAB[k]}: V300 max")
        ax.axhline(R["p40"]["V300"]["median"], color=INK, lw=1, ls=":"); ax.axhline(R["p40"]["V300"]["max"], color=INK, lw=1, ls="-.")
        ax.text(min(cx), R["p40"]["V300"]["median"] * 1.1, "pass 40 V300 median", fontsize=8, color=INK2)
        ax.text(min(cx), R["p40"]["V300"]["max"] * 1.1, "pass 40 V300 max", fontsize=8, color=INK2)
        ax.set_yscale("log"); ax.set_xlabel("total expression complexity (α + β, PySR count)"); ax.set_ylabel("E")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "figE4_pysr_tradeoff.png", dpi=150); plt.close(fig)

    # E5: where the best network / regressor beats pass 40
    pick = [k for k in ("mlp_t", "gpr") if k in R] or [k for k in ("mlp", "gpr") if k in R]
    if pick:
        cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a07f", "#b53a1c"])
        fig, axs = plt.subplots(1, len(pick), figsize=(5.6 * len(pick) + 1, 4.4), sharey=True, squeeze=False)
        for ax, k in zip(axs[0], pick):
            r = np.log10(np.array(R[k]["V300"]["cases"]) / pv)
            sc = ax.scatter(V[:, 0], V[:, 1], c=r, cmap=cmap, norm=TwoSlopeNorm(0, -2, 2), s=30, edgecolors=INK2, linewidths=0.3)
            ax.set_title(f"{LAB[k]} (T96)\nbetter than pass 40 in {np.mean(r < 0):.0%} of V300 cases", loc="left", fontsize=10)
            ax.set_xlabel("q"); ax.grid(False)
        axs[0, 0].set_ylabel("χ1")
        cb = fig.colorbar(sc, ax=axs[0], shrink=0.9, pad=0.02); cb.set_label("log10(E_ML / E_pass40)\nblue: ML better · red: pass 40 better")
        fig.savefig(OUT / "figE5_where_wins.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    summ = {k: dict(T96={x: R[k]["T96"][x] for x in ("median", "mean", "max")}, V300={x: R[k]["V300"][x] for x in ("median", "mean", "max")},
                    n=R[k]["n"], train_s=R[k].get("train_s"), init_T96=R[k].get("init_T96", {}).get("median") if R[k].get("init_T96") else None,
                    better_than_p40=float(np.mean(np.array(R[k]["V300"]["cases"]) < pv)) if k != "p40" else None,
                    better_than_lean12=float(np.mean(np.array(R[k]["V300"]["cases"]) < np.array(R["lean12"]["V300"]["cases"]))) if k != "lean12" else None,
                    roughness=rough.get(k)) for k in REFS + have}
    for k in ("sr", "sr_t"):
        if k in R:
            summ[k]["complexity"] = R[k]["complexity"]; summ[k]["expr"] = R[k]["expr"]
    summ["_cases_E3"] = {n: list(V300[i]) for n, i in sel.items()}; summ["_best_regressor_T96"] = bestlin
    (RES / "efit_summary.json").write_text(json.dumps(summ, indent=1))
    for k in REFS + have:
        v = summ[k]["V300"]; print(f"{k:7s} V300 {v['median']:.2e}/{v['mean']:.2e}/{v['max']:.2e}  T96 mean {summ[k]['T96']['mean']:.2e}  n={summ[k]['n']}")


if __name__ == "__main__":
    main()
