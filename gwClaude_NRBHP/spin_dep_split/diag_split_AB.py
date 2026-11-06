"""Diagnostic figures for split A and split B (2026-10-15), the set the published spin models carry
(spin_dep/plot_spin_diagnostics.py, fig10_tend/fig10_ourmodels.py, spin_alpha_beta_overlay.py, mismatch_heatmap_dense.py):

  fig10_tend_q4         Islam+ 2407.18319 Fig. 10 method: at q = 4, chi1 = -0.6 .. +0.6, per case and per window
                        NR [-4500, t_end] (t_end = 75, 0, -50, -100, -300 M) re-fit the model's own per-case form.
                        Split per-case form = 8 numbers: inspiral (a_pp, a_f, b_pp, b_e) with the derived SO factor, MR
                        (a_pp', a_f', b_pp', b_e'); the window S and every anchor stay as in the model.  E_installed = the
                        model's own coefficients on the same window.
  error_by_band_q4      full-window E split into NR time bands (bands add up to E; phi0 global), q = 4, chi1 = -0.6..+0.6
  residual_vs_time_q4   phase and amplitude residual vs NR time, q = 4, chi1 = -0.6, 0, +0.6
  peak_errors_grid      peak amplitude error and peak time shift of the calibrated (2,2) mode on T96
  chi_const/            alpha, beta on the NR clock across q = 2, 3, 5, 8 at chi1 = -0.6, -0.2, +0.2, +0.6 (+ contact sheet)
  q_const/              alpha, beta across the 8 T96 spins at q = 2, 3, 5, 8 (+ contact sheet)
  mismatch_heatmap_dense  E on the 20 x 20 (q, chi1) grid, q in [2, 8] (q < 3 extrapolated), one log scale for A and B
E everywhere = full-window E, full alignment, the number the model was scored with.  Gate: E rebuilt here equals the
stored per-case T96 score (ratio within 1e-6 on all 96), and the 8-number per-case form reproduces the model exactly.
Caches are read only (q = 4 from fig10_tend/, the rest via extrap_lowq.get_case).

    python diag_split_AB.py --part cheap      everything but the refits      -> results/diag_AB/cheap.json
    python diag_split_AB.py --part tend       the Fig. 10 refits (130 tasks) -> results/diag_AB/tend.json
    python diag_split_AB.py --plot
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cv_split as C
X, FF, M = C.X, C.FF, C.M
S, FS, SO = M.S, X.FS, X.SO
D = M.D
OUTR = HERE / "results" / "diag_AB"; OUTR.mkdir(parents=True, exist_ok=True)
PLOTS = HERE / "Agentic_spin_split_plots"
Q4_CHIS = [round(-0.6 + 0.1 * k, 1) for k in range(13)]
RES_CHIS = [-0.6, 0.0, 0.6]
BANDS = [(-5000.1, -1000.0), (-1000.0, -300.0), (-300.0, -100.0), (-100.0, -50.0), (-50.0, 0.0), (0.0, 100.0)]
BAND_LABELS = ["< -1000", "-1000..-300", "-300..-100", "-100..-50", "-50..0", "0..+100"]
T_START, T_ENDS = -4500.0, [75.0, 0.0, -50.0, -100.0, -300.0]
OV_Q, OV_CHI = [2.0, 3.0, 5.0, 8.0], [-0.6, -0.2, 0.2, 0.6]
DQ, DC = np.linspace(2, 8, 20), np.linspace(-0.6, 0.8, 20)
DENSE = [(float(q), float(c)) for q in DQ for c in DC]
Q4 = [(4.0, c) for c in Q4_CHIS]

_get = D.get_case


def get_case(q, chi, where="grid"):
    if q == 4.0:                                                  # exact q = 4: the Fig. 10 study's caches
        S.CACHE, FS.LOSSCACHE = ROOT / "fig10_tend" / "spin_fmt_cache", ROOT / "fig10_tend" / "loss_cache"
        w = S.CACHE / f"wf_q{q:.6f}_chi{chi:+.3f}.npz"; l = FS.LOSSCACHE / f"loss_q{q:.6f}_chi{chi:+.3f}.npz"
        if not (w.exists() and l.exists()):
            sys.exit(f"missing q = 4 cache {w} / {l}")
        return FS.build_case(q, chi)
    return _get(q, chi, where)


D.get_case = get_case


def models():
    a = json.loads((X.OUT / "slim__win-2.0.json").read_text())
    b = json.loads((HERE / "results" / "cv" / "tline__foldall.json").read_text())
    rows = lambda rs: {(r["q"], r["chi1"]): r["full_E_full"] for r in rs}
    return {"split_A_slim_T-2": dict(label="split A: slim, window from T = -2 (14 coef)", ti=np.array(a["theta_insp"]),
                                     x=a["x_mr"], window=("const", -2.0), T96=rows(a["cases"]["T96"])),
            "split_B_b0_T-3": dict(label="split B: inspiral b = 0, window from T = -3 (13 coef + T0)",
                                   ti=np.array(b["theta_insp"]), x=b["x_mr"], window=("line", b["chosen"]["T0"], b["chosen"]["T1"]),
                                   T96=rows(b["train_rows"]))}


def set_win(w):
    X.set_window(w[1]) if w[0] == "const" else C.set_tline(w[1], w[2])


MOD = {}
RUN1 = {}
ONLY = None


# ---------------------------------------------------------------- the model as an 8-number per-case form
def installed(q, chi):
    d = MOD["m"]; names = X.MR_FORMS["slim"][0]
    tm = np.array(d["ti"], float).copy(); tm[[X.NAMES.index(n) for n in names]] = d["x"]
    return np.r_[FS.params_at(q, chi, d["ti"]), FS.params_at(q, chi, tm)]


def ab8(p, q, chi, c, key):
    pre = "fit_" if key == "fit" else ""
    los = c[pre + "losses"]; F, E = los["flux_hat"], los["e_oft"]; Sb = c[pre + "S"]
    xc = SO.xnote(q, c, pre)
    ai = p[0] * (1 + p[1] * F); bi = p[2] * (1 + p[3] * E) * (1 + SO.so_coef(q, corrected=True) * chi * xc ** 1.5)
    am = p[4] * (1 + p[5] * F); bm = p[6] * (1 + p[7] * E)
    return (1 - Sb) * ai + Sb * am, (1 - Sb) * bi + Sb * bm


def model_ab(q, chi, c, key):
    d = MOD["m"]; return X.ab("split", d["ti"], ("form", "slim", d["x"]), q, chi, c, key)


def err(p, c, key, t0, wide, q, chi, t_end=None):
    """E on NR [T_START, t_end] (t_end None: the model's full window) with global phi0; returns (E, t0)."""
    pre = "fit_" if key == "fit" else ""
    t, h, tn, hn = c[pre + "t_bhpt"], c[pre + "h_bhpt"], c[pre + "t_nr"], c[pre + "h_nr"]
    if t_end is not None:
        m = (tn >= T_START) & (tn <= t_end); tn, hn = tn[m], hn[m]
    al, be = ab8(p, q, chi, c, key)
    if np.any(be <= 0) or not np.all(np.isfinite(np.r_[al, be])):
        return 50.0, t0
    ts = M.F.tau_shape(be, t)
    if np.min(np.diff(ts)) <= 0:
        return 50.0, t0
    f = lambda v: S._err_at_t0(v, al, ts, t, h, tn, hn)
    if wide:
        g = np.linspace(t0 - 200, t0 + 200, 161); cen, half = float(g[int(np.argmin([f(v) for v in g]))]), 3.0
    else:
        cen, half = t0, 10.0
    r = minimize_scalar(f, bounds=(cen - half, cen + half), method="bounded", options={"xatol": 1e-4, "maxiter": 80})
    return float(r.fun), float(r.x)


def calibrated(k):
    """Full-window, full alignment exactly as the scorer (M.score), then the residual pieces."""
    q, chi = k; c = M.CASES[k]
    al_f, be_f = model_ab(q, chi, c, "fit")
    _, t0 = M.score(al_f, be_f, c, "fit", "full", -50.0, wide=True)
    al, be = model_ab(q, chi, c, "full")
    errs, t1 = M.score(al, be, c, "full", "full", t0, wide=True)
    a8, b8 = ab8(installed(q, chi), q, chi, c, "full")
    assert np.allclose(a8, al, rtol=1e-12, atol=0) and np.allclose(b8, be, rtol=1e-12, atol=0), k
    t, h, tn, hn = c["t_bhpt"], c["h_bhpt"], c["t_nr"], c["h_nr"]
    tau = t1 + M.F.tau_shape(be, t)
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
    tc, r = tn[m], hn[m]
    g = np.interp(tc, tau, al) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
    g = g * np.exp(1j * np.angle(np.vdot(g, r)))
    n1 = np.sum(np.abs(r) ** 2); e = float(np.sum(np.abs(r - g) ** 2) / (2 * n1))
    assert abs(e / errs[2] - 1) < 1e-6, (k, e, errs[2])
    return dict(E=e, tc=tc, r=r, g=g, n1=n1, tau=tau, alpha=al, beta=be)


def peak(t, h):
    a = np.abs(h); i = min(max(int(np.argmax(a)), 1), len(a) - 2)
    y0, y1, y2 = a[i - 1], a[i], a[i + 1]; dd = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return t[i] + dd * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * dd


def work_cheap(task):
    kind, k = task; w = calibrated(k); out = dict(kind=kind, q=k[0], chi1=k[1], E=w["E"])
    tc, r, g, n1 = w["tc"], w["r"], w["g"], w["n1"]
    if kind == "q4":
        out["bands"] = [float(np.sum(np.abs((r - g)[(tc >= lo) & ((tc < hi) if hi < 100 else (tc <= hi))]) ** 2) / (2 * n1))
                        for lo, hi in BANDS]
        assert abs(sum(out["bands"]) / w["E"] - 1) < 1e-8, k
        if k[1] in RES_CHIS:
            s = slice(None, None, 5)
            out["series"] = dict(t=tc[s].tolist(), dphi=np.unwrap(np.angle(g * np.conj(r)))[s].tolist(),
                                 damp_pk=((np.abs(g) - np.abs(r)) / np.max(np.abs(r)))[s].tolist())
    elif kind == "grid":
        tp_nr, ap_nr = peak(tc, r); tp_m, ap_m = peak(tc, g)
        out.update(dt_peak=float(tp_m - tp_nr), damp_peak=float(ap_m / ap_nr - 1))
    elif kind == "ov":
        tn = M.CASES[k]["t_nr"]; m = (tn >= max(tn[0], w["tau"][0])) & (tn <= min(tn[-1], w["tau"][-1])) & (tn >= -1000) & (tn <= 80)
        tt = tn[m][::5]
        out.update(t=tt.tolist(), alpha=np.interp(tt, w["tau"], w["alpha"]).tolist(), beta=np.interp(tt, w["tau"], w["beta"]).tolist())
    return out


def work_tend(task):
    k, te, extra = task; q, chi = k; c = M.CASES[k]; p0 = installed(q, chi); t = time.time()
    _, t0 = err(p0, c, "fit", -50.0, True, q, chi, te)
    e_inst, t0i = err(p0, c, "full", t0, True, q, chi, te)
    st = {"t0": t0}
    def obj(p):
        _, bf = ab8(p, q, chi, c, "full")
        if not np.all(bf > 0):
            return 50.0 + 1e3 * float(-np.min(bf))
        e, tt = err(p, c, "fit", st["t0"], False, q, chi, te)
        if e < 1.0:
            st["t0"] = tt
        return e
    cands = [(p0, t0i)]
    R1 = RUN1.get((q, chi, te))
    if R1 is not None:                                    # run 1 = the installed start, already optimised
        cands.append((np.array(R1["params"]), R1["t0"]))
    for s0 in ([] if R1 is not None else [p0]) + ([np.array(extra)] if extra is not None else []):   # + previous window's refit
        _, st["t0"] = err(s0, c, "fit", t0, True, q, chi, te); x = np.array(s0, float)
        for _ in range(3):
            r = minimize(obj, x, method="Nelder-Mead", options={"maxiter": 2500 * len(x), "xatol": 1e-8, "fatol": 1e-13, "adaptive": True})
            done = abs(r.fun - obj(x)) < 1e-4 * r.fun; x = r.x
            if done:
                break
        cands += [(x, st["t0"]), (s0, st["t0"])]
    (e_fit, t0f), p = min([(err(cc, c, "full", tt, True, q, chi, te), cc) for cc, tt in cands], key=lambda z: z[0][0])
    return dict(q=q, chi1=chi, t_end=te, E_refit=e_fit, E_installed=e_inst, t0=t0f, params=list(map(float, p)),
                params_installed=list(map(float, p0)), seconds=round(time.time() - t))


def compute(part):
    Dm = models(); keys = list(X.T96) + Q4 + ([(q, c) for q in OV_Q for c in set(OV_CHI) | set(M.S.CHIS)] + DENSE if part == "cheap" else [])
    for k in dict.fromkeys(keys):
        M.CASES[k] = M.prepare(*k)
    res = {}
    for name, d in Dm.items():
        if ONLY and name != ONLY:
            continue
        set_win(d["window"]); MOD["m"] = d; t = time.time()
        r1 = OUTR / "tend_run1.json"
        RUN1.clear(); RUN1.update({(r["q"], r["chi1"], r["t_end"]): r for r in json.loads(r1.read_text())["results"][name]} if r1.exists() else {})
        with get_context("fork").Pool(128) as pool:
            gate = np.array([w["E"] for w in pool.map(work_cheap, [("gate", k) for k in X.T96])])
            r = gate / np.array([d["T96"][k] for k in X.T96])
            print(f"GATE {name}: T96 rebuilt / stored {r.min():.7f}..{r.max():.7f}", flush=True)
            assert np.allclose(r, 1, atol=1e-6), r
            if part == "cheap":
                ov = [(q, c) for q in OV_Q for c in sorted(set(OV_CHI) | set(M.S.CHIS))]
                tasks = [("q4", k) for k in Q4] + [("grid", k) for k in X.T96] + [("ov", k) for k in ov] + [("dense", k) for k in DENSE]
                res[name] = pool.map(work_cheap, tasks, chunksize=1)
            else:
                # chained as fig10_ourmodels.py: each window also starts from the previous (longer) window's refit
                prev = {k: None for k in Q4}; res[name] = []
                for te in T_ENDS:
                    rr = pool.map(work_tend, [(k, te, prev[k]) for k in Q4], chunksize=1)
                    res[name] += rr; prev = {(r["q"], r["chi1"]): r["params"] for r in rr}
        print(f"{name}: {part} done ({time.time() - t:.0f}s)", flush=True)
    fn = OUTR / (f"{part}_{ONLY}.json" if ONLY else f"{part}.json")
    fn.write_text(json.dumps(dict(bands=BAND_LABELS, T_START=T_START, results=res)))
    print("wrote", fn, flush=True)


# ---------------------------------------------------------------- figures
def plot():
    import matplotlib; matplotlib.use("Agg"); matplotlib.rcdefaults()
    import matplotlib.pyplot as plt, matplotlib.ticker, matplotlib.colors as mcolors
    Dm = models()
    ch = json.loads((OUTR / "cheap.json").read_text())["results"]
    tn = {}                                               # per-model chained files win over the combined one
    for name in Dm:
        f = OUTR / f"tend_{name}.json"
        if f.exists():
            tn[name] = json.loads(f.read_text())["results"][name]
        elif (OUTR / "tend.json").exists():
            tn[name] = json.loads((OUTR / "tend.json").read_text())["results"][name]
    grey = {"axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.grid": True, "grid.color": "white",
            "grid.linewidth": 1.0, "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0}
    def save(fig, name, folder, sub=""):
        out = PLOTS / folder / sub; out.mkdir(parents=True, exist_ok=True)
        for e in ("png", "pdf"):
            fig.savefig(out / f"{name}.{e}", dpi=150, bbox_inches="tight")
        plt.close(fig); print(f"  wrote {folder}/{sub + '/' if sub else ''}{name}", flush=True)
    dense_all = np.array([r["E"] for n in ch for r in ch[n] if r["kind"] == "dense"])
    dn = mcolors.LogNorm(vmin=dense_all.min(), vmax=dense_all.max())
    for name, d in Dm.items():
        R = ch[name]; lab = d["label"]
        # residual vs time
        plt.rcdefaults(); plt.rcParams.update(grey)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True); lim = 0.0; off = []
        for chi, col in zip(RES_CHIS, ("tab:blue", "black", "tab:red")):
            s = next(r for r in R if r["kind"] == "q4" and r["chi1"] == chi)["series"]
            t = np.array(s["t"]); dphi = np.array(s["dphi"])
            a1.plot(t, dphi, color=col, lw=1.3, label=rf"$\chi_1={chi:+.1f}$"); a2.plot(t, 100 * np.array(s["damp_pk"]), color=col, lw=1.3)
            lim = max(lim, np.max(np.abs(dphi[t <= 30.0]))); off.append((chi, float(dphi[np.argmax(np.abs(dphi))]), col))
        ylim = 1.3 * lim + 0.1; a1.set_ylim(-ylim, ylim)
        for j, (chi, v, col) in enumerate(o for o in off if abs(o[1]) > ylim):
            a1.text(0.02, 0.92 - 0.1 * j, rf"$\chi_1={chi:+.1f}$ reaches {v:+.0f} rad by +100 M (off scale)", transform=a1.transAxes,
                    fontsize=9, color=col, va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ticks = [-5000, -1000, -300, -100, -50, -20, 0, 20, 50, 100]
        for a in (a1, a2):
            a.set_xscale("symlog", linthresh=50, linscale=1.0); a.axvline(0, color="grey", ls="--", lw=0.8)
            a.set_xticks(ticks); a.set_xticklabels([str(v) for v in ticks]); a.set_xlim(-5000, 100)
            a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a1.set_ylabel(r"$\Delta\phi$  (rad)", fontsize=12)
        a2.set_ylabel(r"$(|h_{\rm model}|-|h_{\rm NR}|)\,/\,|h_{\rm NR}|_{\rm peak}$  (%)", fontsize=11)
        a2.set_xlabel(r"$t$  (M, NR peak at 0; logarithmic before $-50$, linear after)", fontsize=11)
        a1.legend(ncol=3, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        a1.set_title(f"{lab}, q = 4: residual vs time", fontsize=11, pad=26)
        fig.tight_layout(); save(fig, "residual_vs_time_q4", name)
        # error by band
        plt.rcdefaults(); plt.rcParams.update(grey)
        rr = sorted([r for r in R if r["kind"] == "q4"], key=lambda r: r["chi1"])
        chis = np.array([r["chi1"] for r in rr]); E = np.array([r["E"] for r in rr]); B = np.array([r["bands"] for r in rr])
        frac = B / B.sum(axis=1, keepdims=True)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.6]))
        a1.semilogy(chis, E, "ko-"); a1.set_ylabel(r"$\mathcal{E}_{22}$ (full window)", fontsize=11)
        cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(BAND_LABELS))); bot = np.zeros(len(chis))
        for j, bl in enumerate(BAND_LABELS):
            a2.bar(chis, 100 * frac[:, j], width=0.08, bottom=bot, color=cols[j], label=f"{bl} M"); bot += 100 * frac[:, j]
        a2.set_ylim(0, 100); a2.set_ylabel("share of the error (%)", fontsize=11); a2.set_xlabel(r"$\chi_1$", fontsize=13); a2.set_xticks(chis)
        a2.legend(ncol=3, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, title="time band", title_fontsize=8.5)
        a1.set_title(f"{lab}, q = 4: where the error comes from", fontsize=11)
        fig.tight_layout(); save(fig, "error_by_band_q4", name)
        # peak errors
        plt.rcdefaults()
        G = [r for r in R if r["kind"] == "grid"]; qs, cs = sorted({r["q"] for r in G}), sorted({r["chi1"] for r in G})
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, kk, tl, fac in ((axs[0], "damp_peak", "peak amplitude error (%)", 100), (axs[1], "dt_peak", "peak time shift, model - NR (M)", 1)):
            Z = np.full((len(cs), len(qs)), np.nan)
            for r in G:
                Z[cs.index(r["chi1"]), qs.index(r["q"])] = fac * r[kk]
            v = np.nanmax(np.abs(Z)); im = ax.imshow(Z, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
            for i in range(len(cs)):
                for j in range(len(qs)):
                    ax.text(j, i, f"{Z[i, j]:+.1f}", ha="center", va="center", fontsize=6.5, color="white" if abs(Z[i, j]) > 0.6 * v else "black")
            ax.set_xticks(range(len(qs))); ax.set_xticklabels([f"{q:.2f}" for q in qs], rotation=60, fontsize=8)
            ax.set_yticks(range(len(cs))); ax.set_yticklabels([f"{c:+.1f}" for c in cs], fontsize=8)
            ax.set_xlabel("q"); ax.set_ylabel(r"$\chi_1$"); ax.set_title(tl, fontsize=11); fig.colorbar(im, ax=ax, shrink=0.9)
        fig.suptitle(f"{lab}: calibrated (2,2) peak vs NRHybSur3dq8, training grid", fontsize=11)
        fig.tight_layout(); save(fig, "peak_errors_grid", name)
        # alpha / beta overlays
        OV = {(r["q"], r["chi1"]): r for r in R if r["kind"] == "ov"}
        QC = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
        def panel(axa, axb, pairs, legend=True):
            for (q, c), col, lb, lw in pairs:
                r = OV[(q, c)]
                axa.plot(r["t"], r["alpha"], color=col, lw=lw, label=f"{lb}  E={r['E']:.2e}"); axb.plot(r["t"], r["beta"], color=col, lw=lw, label=lb)
            for ax, yl in ((axa, r"$\alpha(t)$"), (axb, r"$\beta(t)$")):
                ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80); ax.grid(True)
                ax.set_xlabel(r"$t_{\rm NR}\,/\,M$"); ax.set_ylabel(yl)
                if legend:
                    ax.legend(fontsize=7.5)
        qpairs = lambda c: [((q, c), QC[q], f"q={q:g}" + (" [extrap]" if q < 3 else ""), 2.2 if q < 3 else 1.3) for q in OV_Q]
        cm = plt.get_cmap("coolwarm"); nrm = mcolors.Normalize(-0.6, 0.8)
        cpairs = lambda q: [((q, c), cm(nrm(c)), rf"$\chi_1$={c:+.1f}", 1.3) for c in M.S.CHIS]
        for tag, sub, items, pairs, fmt in (("chi", "chi_const", OV_CHI, qpairs, lambda c: f"chi{c:+.1f}".replace("+", "p").replace("-", "m")),
                                            ("q", "q_const", OV_Q, cpairs, lambda q: f"q{q:g}")):
            for v in items:
                fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4)); panel(axa, axb, pairs(v))
                fig.suptitle(f"{lab}: " + (rf"$\alpha/\beta$ across $q$ at $\chi_1$ = {v:+.1f}" if tag == "chi" else rf"$\alpha/\beta$ across $\chi_1$ at q = {v:g}")
                             + (" (q = 2: extrapolated, also below the ppBHPT surrogate)" if tag == "q" and v < 3 else ""), fontsize=10)
                fig.tight_layout(); save(fig, f"{name}_alpha_beta_overlay_{fmt(v)}", name, sub)
            fig, axes = plt.subplots(len(items), 2, figsize=(12, 3.6 * len(items)))
            for i, v in enumerate(items):
                panel(axes[i, 0], axes[i, 1], pairs(v)); axes[i, 0].set_title(f"$\\chi_1$ = {v:+.1f}" if tag == "chi" else f"q = {v:g}", fontsize=10)
            fig.tight_layout(); fig.suptitle(f"{lab}: alpha / beta overlays", fontsize=11, y=1.005)
            save(fig, f"{name}_alpha_beta_overlay_{'all' if tag == 'chi' else 'q_all'}", name, sub)
        # dense heatmap
        plt.rcdefaults()
        Z = np.full((len(DC), len(DQ)), np.nan)
        for r in R:
            if r["kind"] == "dense":
                Z[int(np.argmin(np.abs(DC - r["chi1"]))), int(np.argmin(np.abs(DQ - r["q"])))] = r["E"]
        fig, ax = plt.subplots(figsize=(8, 5.6))
        im = ax.pcolormesh(DQ, DC, Z, norm=dn, cmap="viridis", shading="nearest")
        ax.axvline(3.0, color="white", ls="--", lw=1); ax.text(2.05, 0.75, "q < 3 extrapolated", color="white", fontsize=8)
        tq = [k[0] for k in X.T96]; tcc = [k[1] for k in X.T96]
        ax.scatter(tq, tcc, s=10, facecolors="none", edgecolors="white", lw=0.6, label="T96 (training)")
        fig.colorbar(im, ax=ax, label=r"$\mathcal{E}$"); ax.set_xlabel("q"); ax.set_ylabel(r"$\chi_1$")
        ax.set_title(f"{lab}\n20 x 20 grid, median {np.nanmedian(Z):.2e} (one colour scale for A and B)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8); fig.tight_layout(); save(fig, f"{name}_mismatch_heatmap_dense", name)
        # Fig. 10 t_end
        if name in tn:
            plt.rcdefaults(); plt.rcParams.update(grey)
            COL = {75.0: "blue", 0.0: "red", -50.0: "green", -100.0: "cyan", -300.0: "black"}
            fig, ax = plt.subplots(figsize=(7.0, 5.0))
            for te in T_ENDS:
                rr = sorted([r for r in tn[name] if r["t_end"] == te], key=lambda r: r["chi1"])
                ax.semilogy([r["chi1"] for r in rr], [r["E_refit"] for r in rr], "o-", color=COL[te], label=rf"$t_{{\rm end}}={te:.0f}M$")
            ax.set_xlabel(r"$\chi_1$", fontsize=14); ax.set_ylabel(r"$\mathcal{E}_{22}$", fontsize=14)
            ax.set_title(f"{lab},  q = 4\nper-case refit on NR [-4500, t_end]", fontsize=11, pad=34)
            ax.legend(ncol=5, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.5, columnspacing=0.9)
            fig.tight_layout(); save(fig, "fig10_tend_q4", name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", choices=["cheap", "tend"]); ap.add_argument("--plot", action="store_true")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    global ONLY
    ONLY = a.model
    if a.part:
        compute(a.part)
    if a.plot:
        plot()


if __name__ == "__main__":
    main()
