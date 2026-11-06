"""Build the figures page (HTML, figures embedded) from plots/*.png and results/efit_summary.json.

    python make_page.py <out.html>
"""
import base64, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P, R = HERE / "plots", HERE / "results"
S = json.loads((R / "efit_summary.json").read_text())
INIT = json.loads((R / "efit_lin_init.json").read_text()) if (R / "efit_lin_init.json").exists() else {}
img = lambda f: "data:image/png;base64," + base64.b64encode((P / f).read_bytes()).decode()
NAME = dict(p40="pass 40 (hand forms)", lean12="lean-12, pass 49b (hand forms)", mlp="MLP, identity start", mlp_t="MLP, model-free start", gru="GRU, identity start",
            sr="PySR, from MLP (identity start)", sr_t="PySR, from MLP (model-free start)", knn="KNN", rbf="RBF",
            gpr="GPR", rf="Random forest", et="ExtraTrees", xgb="XGBoost", sparse="Sparse Legendre library")
KIND = dict(p40="hand-written", lean12="hand-written", mlp="black box", mlp_t="black box", gru="black box", sr="functional", sr_t="functional",
            knn="regressor", rbf="regressor", gpr="regressor", rf="regressor", et="regressor", xgb="regressor", sparse="functional")
e = lambda x: f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def nparam(k):
    v = S[k]; n = v["n"]
    if k in ("sr", "sr_t"):
        return f"{n} constants, complexity {v['complexity']}"
    if k == "sparse":
        return f"{n:,} values (15 α + 14 β library terms × 1900 grid)"
    if k in ("mlp", "mlp_t", "gru"):
        return f"{n:,} weights"
    if k == "p40":
        return "18 coefficients"
    if k == "lean12":
        return "12 coefficients"
    return f"{n:,} values (96 points × 3800)" if n == 364800 else f"{n:,} leaf values"


rows = []
REFS = ["p40", "lean12"]
keys = REFS + sorted([k for k in S if not k.startswith("_") and k not in REFS], key=lambda k: S[k]["V300"]["median"])
for k in keys:
    v = S[k]; b = v.get("better_than_p40"); bl = v.get("better_than_lean12")
    tr = v.get("train_s"); tr = f"{tr / 60:.0f} min" if tr else ("PySR 16 min + refit 12 min" if k.startswith("sr") else "6–40 min" if k in REFS else "—")
    ini = INIT.get(k, {}).get("median")
    rows.append(f"<tr class='{'ref' if k in REFS else ''}'><th scope='row'>{NAME[k]}<span class='kind'>{KIND[k]}</span></th>"
                f"<td>{e(v['V300']['median'])}</td><td>{e(v['V300']['mean'])}</td><td>{e(v['V300']['max'])}</td>"
                f"<td>{e(v['T96']['median'])}</td><td>{'—' if b is None else f'{b:.0%}'}</td><td>{'—' if bl is None else f'{bl:.0%}'}</td>"
                f"<td>{e(ini) if ini else '—'}</td><td class='np'>{nparam(k)}</td><td>{tr}</td></tr>")
table = "\n".join(rows)

FIGS_E = [
    ("figE1_v300_ecdf.png", "V300 error distributions",
     "Every model was trained on the same 96 T96 cases by minimising the waveform error E through the map, with φ0 solved analytically and "
     "t0 fitted per case, then scored on V300 with the same scorer as pass 40. Curves further left are better; the right end is the worst case. "
     "GPR and RBF sit left of pass 40 for most cases (better in 79% and 73% of V300) but cross it near the top: their worst cases are about "
     "1.7× pass 40's. The sparse Legendre library stays within about 1.3× of pass 40 across the whole distribution. The networks and PySR "
     "formulas are well to the right. Dashed grey is lean-12, the 12-coefficient pruning of pass 40 (pass 49b): about 1.6× pass 40 at the "
     "median, but the lowest worst case of any model here (8.8e-4), and still the tightest distribution: its whole curve spans less than a decade."),
    ("figE2_summary.png", "Median and worst case per method",
     "Filled markers are V300, open markers are the T96 training error. A large gap means the method memorised the training cases: KNN, RBF "
     "and XGBoost reach about 1e-6 on T96 but 1e-4 to 1e-3 on V300. Pass 40, the sparse library and both MLPs have almost no gap; the MLPs "
     "are limited by how far 30 minutes of training gets them, not by overfitting."),
    ("figE3_alpha_beta.png", "The fitted α(t) and β(t)",
     "α and β for four V300 cases, each model in the gauge its own fit chose (MLP-id: identity start; MLP-mf, PySR-mf: model-free start). "
     "Pass 40 and lean-12 (dashed) are smooth everywhere and nearly coincide. KNN and GPR follow the model-free α, β they were initialised from, including violent post-merger "
     "oscillations of β in the retrograde case. The MLP from the identity map settled in a different solution, with β spiking to about 2.4 "
     "at merger and α to about 10 in the ringdown, and visibly rough curves; started from the model-free α, β it stays close to GPR. Measured "
     "as the rms second difference of ln β before merger, every ML model is 25 to 5000 times rougher than pass 40 (lean-12 is as smooth), "
     "except PySR from the model-free MLP, whose β formula is smoother still but whose error is about 200 times worse."),
    ("figE4_pysr_tradeoff.png", "PySR: complexity against accuracy",
     "PySR searched closed forms of ln α and ln(β/X1) in (log time to merger, ν, χ1) by distilling a trained MLP, and each candidate's "
     "constants were then refitted through E on T96. The choice uses T96 only; V300 is shown for every candidate to expose the trade-off. "
     "Error falls slowly with complexity and stays 200 to 2000 times above pass 40: a 0.4% relative error in the β formula already "
     "integrates into a large phase error. At complexity 70 one V300 case fails outright (E = 1)."),
    ("figE6_testmass.png", "The test-mass limit (q = 1000)",
     "α(t) and β(t)/X1 at q = 1000 (ν ≈ 1e-3), where the exact answer is the identity map, α = β/X1 = 1 (dotted). No NR exists here, so this "
     "shows behaviour, not error. Pass 40 and lean-12 (they coincide) sit on the limit by construction: every finite-mass term carries "
     "g(ν) → 0, and the residual 0.1% is the X1^(6/5) scaling at finite q. Above q = 10 their remnant input is the O(ν) ISCO-plunge "
     "result, since NRSur3dq8Remnant stops at q = 10. GPR and RBF, trained on q = 3–8 only, carry their edge-of-data values out: GPR's "
     "inspiral α settles about 4% low, RBF's about 6% high, and at merger both break down (RBF's β/X1 reaches 4.6e4, i.e. the clock "
     "stops making sense; off-scale peaks are written in each panel)."),
    ("figE5_where_wins.png", "Where ML beats pass 40",
     "log10 of the ML error over the pass 40 error for each V300 case. Blue: ML better; red: pass 40 better. GPR wins most of the interior "
     "but loses along the edges of the training range, at χ1 ≈ −0.6 and at the highest spins. The MLP from the model-free start beats "
     "pass 40 in only 4% of cases."),
]
FIGS_OLD = [
    ("fig1_v300_ecdf.png", "Earlier: direct-correction models", "Predict amplitude and phase corrections and bypass α, β entirely (paradigm A). Not the intended setup."),
    ("fig2_training_size.png", "Earlier: training-set size", "Direct-correction models against pass 40 on the training_size grids."),
    ("fig3_extrap.png", "Earlier: outside the training range", "Direct-correction models and pass 40 at q < 3 and q = 9–10."),
]


def fig(f, t, c, n):
    return (f'<figure><figcaption><span class="num">{n}</span><h3>{t}</h3></figcaption>'
            f'<div class="plate"><img src="{img(f)}" alt="{t}"></div><p>{c}</p></figure>')


body_e = "\n".join(fig(f, t, c, f"Fig. E{i + 1}") for i, (f, t, c) in enumerate(FIGS_E) if (P / f).exists())
body_o = "\n".join(fig(f, t, c, f"Fig. {i + 1}") for i, (f, t, c) in enumerate(FIGS_OLD) if (P / f).exists())
html = f'''<title>ML vs Pass 40 Figures</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap">
<style>
/* layout: one reading column; a results table, then each figure as a plate with its caption below, like a paper's figure pages */
:root {{ --bg:#f6f6f3; --fg:#1f1f1e; --muted:#5f5e5a; --rule:#dcdbd5; --plate:#fcfcfb; --accent:#2a78d6; --refrow:#ecebe5;
  --display:"Source Serif 4", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#191918; --fg:#efeee9; --muted:#b4b3ab; --rule:#33332f; --plate:#fcfcfb; --accent:#5598e7; --refrow:#262624; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#191918; --fg:#efeee9; --muted:#b4b3ab; --rule:#33332f; --plate:#fcfcfb; --accent:#5598e7; --refrow:#262624; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.6 var(--body); }}
main {{ max-width:1100px; margin:0 auto; padding-block:40px 64px; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); display:grid; gap:44px; }}
header, section {{ display:grid; gap:14px; }}
h1 {{ font:600 2rem/1.15 var(--display); margin:0; text-wrap:balance; }}
h2 {{ font:600 1.5rem/1.2 var(--display); margin:0; text-wrap:balance; }}
h3 {{ font:600 1.2rem/1.2 var(--display); margin:0; }}
p {{ margin:0; max-width:75ch; }}
.muted {{ color:var(--muted); }}
.ref {{ font-family:var(--mono); font-size:.85em; }}
ul {{ margin:0; padding-left:1.2em; max-width:75ch; display:grid; gap:6px; }}
.tablewrap {{ overflow-x:auto; border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); }}
table {{ border-collapse:collapse; width:100%; min-width:960px; font-size:.86rem; font-variant-numeric:tabular-nums; }}
th, td {{ padding:7px 10px; text-align:right; border-bottom:1px solid var(--rule); white-space:nowrap; }}
thead th {{ font-weight:600; color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.04em; vertical-align:bottom; white-space:normal; }}
tbody th {{ text-align:left; font-weight:500; }}
tr.ref {{ background:var(--refrow); }}
.kind {{ display:block; font:400 .72rem var(--mono); color:var(--muted); }}
td.np {{ text-align:left; color:var(--muted); }}
figure {{ margin:0; display:grid; gap:12px; border-top:1px solid var(--rule); padding-top:20px; }}
figcaption {{ display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }}
.num {{ font:500 .75rem var(--mono); letter-spacing:.06em; text-transform:uppercase; color:var(--accent); }}
.plate {{ background:var(--plate); border-radius:6px; padding:8px; overflow-x:auto; }}
.plate img {{ display:block; width:100%; min-width:640px; height:auto; }}
</style>
<main>
<header>
<h1>ML function classes for α and β against pass 40</h1>
<p>Each method replaces only the hand-written forms of the exploration. The map h<sub>NR</sub>(τ) ≈ α e<sup>iφ₀</sup> h<sub>pp</sub>(t) with τ = t₀ + ∫β dt, the objective (mean waveform error E over the 96 T96 cases), the analytic φ₀ and the per-case t₀ search are unchanged. Every model gives global α(t; ν, χ1) and β(t; ν, χ1), and is scored on the 300 held-out V300 cases by the same scorer as pass 40. Nothing was selected on V300. R100/L20 remain sealed.</p>
<p class="ref">gwClaude_NRBHP/ml_bench: efit.py, efit_lin.py, efit_sr.py → plot_efit.py</p>
</header>
<section>
<h2>What came out</h2>
<ul>
<li><b>Pass 40 is not beaten across the board.</b> No ML model matches its worst case (9.4e-4) on V300, and none matches its smoothness.</li>
<li><b>Lean-12 makes the same point with fewer numbers.</b> With 12 coefficients its V300 median is 2.3e-4 (1.6× pass 40), but its worst case, 8.8e-4, is the lowest of every model here. Only GPR, RBF, ExtraTrees and the sparse library beat its median, each with thousands to millions of stored values and worst cases 1.4 to 4× higher.</li>
<li><b>The ML models' worst cases sit at the edges.</b> For GPR, RBF, ExtraTrees and KNN all ten worst V300 cases lie within 10% of the boundary of the (q, χ1) box, and GPR's median error is 4× higher there than in the interior. Pass 40's worst cases are mostly interior, near χ1 ≈ 0 at q ≈ 3.4–4.2.</li>
<li><b>GPR and RBF win the typical case.</b> Refined through E, GPR reaches a V300 median of 5.4e-5 (2.7× better than pass 40) and RBF 7.9e-5, but their worst cases are 1.7e-3 and 1.6e-3, and they carry 364,800 values each against pass 40's 18 coefficients.</li>
<li><b>The sparse Legendre library comes closest overall</b>: within about 1.3× of pass 40 on median, mean and max, with almost no gap between training and V300 error.</li>
<li><b>Training through E matters.</b> The same regressors fitted only to model-free α, β are 4 to 12× worse on V300 for GPR, the sparse library and the two forests; for the exact interpolators (KNN, RBF, XGBoost) it adds little.</li>
<li><b>Networks and PySR are far behind at this budget.</b> The MLP reaches 3.9e-4 median only when started from model-free α, β (the identity start lands in a worse, spiky solution); the GRU (1.6e-2) is under-trained; PySR formulas stay at 3e-2 or worse, because small errors in β integrate into large phase errors.</li>
<li><b>Cost.</b> The ML training here took about 5 node-hours, against 6 to 40 minutes per node for a pass 40 fit.</li>
</ul>
</section>
<section>
<h2>Results on V300</h2>
<div class="tablewrap"><table>
<thead><tr><th scope="col" style="text-align:left">Method</th><th scope="col">V300 median</th><th scope="col">V300 mean</th><th scope="col">V300 max</th><th scope="col">T96 median (training)</th><th scope="col">cases better than pass 40</th><th scope="col">cases better than lean-12</th><th scope="col">V300 median before E refinement</th><th scope="col" style="text-align:left">size</th><th scope="col">training</th></tr></thead>
<tbody>
{table}
</tbody></table></div>
<p class="muted">Training on one Lonestar6 node. "Before E refinement" is the regressor fitted to model-free α, β only, the starting point that was then trained through E. Pass 40 fits take 6–40 minutes per node.</p>
</section>
<section>
<h2>Figures</h2>
{body_e}
</section>
<section>
<h2>Earlier diagnostics</h2>
<p>These came from the first version of the benchmark, which predicted waveform corrections directly instead of fitting α and β through E. They are kept for reference only.</p>
{body_o}
</section>
</main>'''
Path(sys.argv[1]).write_text(html); print(len(html) // 1024, "KB")
