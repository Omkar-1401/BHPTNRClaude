# Model history — dates and headline results

## Provenance of the dates

Three independent sources, of differing quality. Every row below is labelled with which
one it came from.

* **`local`** — file mtime from the machine where the models were originally fitted
  (supplied 2026-08-25). This is **last modification, so an upper bound on creation**: a
  model edited weeks after it was built shows the later date. For most of these one-shot
  model scripts it is probably close to the build date, but it is not proof.
* **`in-file`** — a date an author wrote into a `.md` or `.py`. Records when a **result
  was written down**, which can lag the model.
* **`git`** — commit date. Only exists from 2026-08-23; the repo begins with a single
  `Initial commit` (`1751bd5`) and has 5 commits, all that day.

**TACC filesystem dates are worthless for this** — everything copied over on 2026-08-23
carries that mtime. The pre-August history exists only because of the local listing.

Where `local` and `in-file` disagree, both are shown. They are usually consistent, and
the disagreements are informative: `gw_remnant_energy` is `local` 08-01 but its RESUME was
opened 08-03, i.e. built first, written up two days later.

---

## Chronological

### July — the PN-optimised line

| date | src | model / event | headline result |
|:---|:---|:---|:---|
| 2026-07-02 | local | **PN_opt_creative** ← *the first model* | q=5 only, 10 parameters, full-window **6.0368e-5**. Key insight: inspiral beta drifts linearly in the PN-loss coordinate; releasing that one DOF **halved** the error floor. The 1.2e-4 floor of constant-beta models is not irreducible |
| 2026-07-02 | local | **PN_opt_creative_q_dep** (script) | 40-q grid, 1/q polynomial. Median 7.66e-5, max 2.13e-4 |
| 2026-07-04 | local | `time_dep_justification.md` | why time-dependent α, β at all |
| 2026-07-08 | local | **PN_opt_nu_Pade** | PP-anchored deg-4 polynomial in ν, chosen by bake-off. Median 7.05e-5, max 1.68e-4, q=2 ~**6.5%** |
| 2026-07-09 | local | `extrapolation_summary.md` | the 1/q polynomial diverges at q=2 (φ0 → ~369 rad); bounded monotone coordinates + endpoint anchoring tame it |
| 2026-07-12 | local | **q_dep_classic** | constant α, β (Islam+22 model class), PP-anchored quartic in 1/q. ~1e-3 in-range, ~1e-2 at q=2 |
| 2026-07-13 | local | **PN_opt_remnant_partial** | 64-q grid [3,8], β_r factorised, 9 params on χ_f poly. Median 7.48e-5, max 5.98e-4, **q=2 ~0.37%** |
| 2026-07-14 | local | `model_validity.md` | |
| 2026-07-15 | local | **PN_opt_full_PN_q_dep** | no logistic switch; direct PN coupling, 9 params |
| 2026-07-15 | local | **wf_nu_q_dep** | same 10-param gated architecture but loss coordinates from the **waveform GW flux**, not PN. No surfinBH |
| 2026-07-17 | local | **wf_nu_hybrid_q_dep** | 11-param hybrid. Per-q median 8.27e-5, cubic master 1.27e-4, q=2 extrap 3.92e-3 |
| 2026-07-17 | local | **wf_nu_switchless** | 8-param, no switch. **FAILED as master**: median 8.8e-3, ~65× worse than gated. Beta's flux couplings scatter non-monotonically and feed the time map |
| 2026-07-17 | local | **wf_nu_switchless_alpha** | switch removed from α only. Less bad, still fails the 1e-2 gate. **α also needs the gate** |
| 2026-07-19 | local | **wf_nu_fluxes** | 12-param, two instantaneous-flux terms on α. **Best-ever per-q (6.86e-5)** but fails as master (6.31e-4): the two terms are degenerate, the per-q split scatters, and the basin-jump at q≈3.5–4 is unregressable |
| 2026-07-21 | local | **peaks Stage-1** | peak point clouds from the GW phase (extrema spaced by π) |
| 2026-07-22 | in-file | peaks Stage-2 built & diagnosed, paused mid-decision | hyperparameters chosen by bake-off |
| 2026-07-23 | local + in-file | **peaks Stage-2 shipped** | α, β read off peaks, **not L2-fit**. In-range median **0.26%**, q=2 extrap **0.55%**, coverage 98%. ~40× worse in-band than L2 models — a physical/low-q method, not an in-band replacement |
| 2026-07-26 | in-file | **wf_nu_hybrid_global** → even-ν training sampling | even-q FAILED the gate at q=2.25 (1.10e-2); even-ν clears q∈[2,2.75] (q2=4.04e-3). In-range median 1.81e-4, **max 5.2e-4** vs the per-q master's 9.6e-4 |
| 2026-07-28 | local + in-file | **pn_anchored** | q-dependence imposed analytically; only **9 ν-flat residual constants** fitted. q<3 is *evaluated not extrapolated* → monotonic, no q2.25 notch, **q2 = 2.55e-3**. Caveat: switch stays weak (S≤0.18), so the MR branch is under-engaged |
| 2026-07-28 | in-file | hm backbone on the `pn_anchored` base (never committed) | (3,3) 1.98e-3 / q2 6.5e-3; (4,4) 9.1e-3 / q2 7.0e-2. **(4,4) at q=2 fails in amplitude, not phase** — structural, likely spherical-spheroidal mixing |
| 2026-07-29 | local | `global_vs_regression.md` | |

### August — the energy-driven line

| date | src | model / event | headline result |
|:---|:---|:---|:---|
| 2026-08-01 | local + in-file | **gw_remnant_energy** (`gw_remnant` added to the env same day) | switchless, energy-driven, **no PN, no gate, no remnant fits**. 5 free params/q; in-range median **6.16e-4** / max 9.87e-4. E(t) supplies the merger ramp the logistic gate used to fake |
| 2026-08-02 | local | **gwr_energy_stiff** | 9 coefficients, ν-structure imposed from measurement. Median 6.24e-4, **q2 3.79e-3** (6.4× better than the 14-coef version). Lesson: **the global fit helps ONLY once the parameterisation is too stiff to reach the per-q floor by regression** |
| 2026-08-02 | local | **gwr_energy_global** | global joint fit of `gw_remnant_energy` — a **WASH**. q<3 is information-limited, not method-limited. The nonlinear-excursion regulariser hurts monotonically |
| 2026-08-03 | in-file | `RESUME_gwremnant.md` opened; "beta's LEVEL is physical, its DRIFT is gauge"; stiff degree choices flagged as unsupported | still-open degree issues on `alpha_E` and `beta_PP` |
| 2026-08-09 | local | **gwr_energy_anchored** | 6 coefficients, `X1^(6/5)` in both prefactors. Median 6.26e-4, **q2 1.53e-3** — 2.47× better than stiff at q=2, beats `pn_anchored`. **No longer PN-free** |
| 2026-08-09 | local | **gwr_energy_flux** | flux α on the same 14-coefficient layer. **25% better in-range at identical parameter count**, low-q unchanged → **α's COORDINATE controls in-band accuracy; the REGRESSION LAYER controls extrapolation** |
| 2026-08-09 | local | **gwr_energy_fluxanchored** ← *best model on both axes* | **7 coefficients.** Median **4.75e-4**, q2 1.31e-3; seeded variant q2 **7.59e-4**, q<3 as accurate as in-range for the first time in the workspace |
| 2026-08-11 | in-file | **"beta must rise at every q" pipeline opened** (user requirement) | drives everything after |
| 2026-08-11 | in-file | **THE KEY BETA RESULT** — the inspiral wants a RISING beta at every q | truncating to t≤−200 M flips `b_E` positive at q=3/5/8; forcing the opposite sign costs ×1.34–×96 |
| 2026-08-11 | in-file | settled non-fixes | `b_E ≥ 0` gives a **FLAT** beta; `gauge_aligned` attempt 1 **INVALID** (the fit, not the gauge); F does **not** peak at merger |
| 2026-08-12 | local | **gwr_beta_monotone** | `P(ν)=|a|·max(0,ν−ν_c) ≥ 0` so beta never decreases — but **FLAT above q=5.75**, and flat is not increasing |
| 2026-08-12/13 | in-file | **frequency-domain alpha** diagnostic | `α(f)=√(P_NR/P_pp)/a_PP` is a **q-universal S-curve** 0.970→1.015; deg-4 residual ~9e-4 vs ~2.6e-3 in time — **3× smoother**, and easier toward low q |
| 2026-08-13 | local + in-file | **fdomain alpha model**, 18 coefficients | median **1.171e-3**, q2 3.555e-3 — **2–3× worse** than fluxanchored with 18 coefficients against 7. The smoother shape did **not** buy a better model. Suspect: the `a_PP` consistency question |
| 2026-08-18 | local | **gwr_anchored2** | α, β as anchor-to-anchor interpolations in `Ehat` with ONE fitted coefficient; anchors `X1^(6/5)` and the peak amplitude ratio |
| 2026-08-18 | local | **gwr_anchored3** | anchored2 with α's anchor placed correctly (A1 is a value *at merger*, so normalise by `Ehat_merger`; α keeps falling after merger → separate post-merger slope). Both cost **4–37×** and did not deliver rising beta |
| 2026-08-18 | in-file | **RESOLVED: beta's falling drift is an artefact of the full-window L2 objective** | the cause is the objective, not the parameterisation — after five failed attempts to fix it by changing the form |
| 2026-08-19 | local | **gwr_qnmshape** | beta's **ν-shape** from the QNM/RemS anchor, one global scale `w` fitted; beta not pinned to B1 |
| 2026-08-19 | local + in-file | **gwr_betaqnm** (`gwModelRemS` installed) | first model with beta pinned between **TWO derived anchors**. **beta rises at every q** (+0.72% at q=8 to +10.1% at q=2) — which five earlier routes could not achieve — but NOT competitive: median 2.93e-3, q2 5.56e-2. **QNM anchor priced at 6.2×** |
| 2026-08-20 | in-file | **WHY the drift flips at q≈4 — SETTLED**; tug-of-war RETRACTED | all models cross at 3.799–4.003, mean **3.92**; nothing happens at q=4 — the crossing is where the anchor gap shrinks past the model's shortfall. Both regions flip *together* |

### August 23–25 — TACC, version control, and the window/coordinate experiments

| date | src | model / event | headline result |
|:---|:---|:---|:---|
| 2026-08-23 | git | workspace under version control (`BHPTNRClaude`, `zero_spin_calib`); TACC env documented | 5 commits |
| 2026-08-23 | git | **inspiral_only**: `inspiral_fluxanchored` + `inspiral_anchored` | rising beta at **64/64 q** in both; full-window `b_E` crosses zero at 3.799 / 3.928 vs the multi-model 3.92. Alpha-drive control: swapping F→E moves `b_E` **<1%** over q∈[3,7] |
| 2026-08-24 | git | inspiral_only **fitted (ν-regressed) layer**; α inherited from the parent | 3 free coefficients, all beta's. **Beta rises at 10/10 q including all four extrapolated below q=3**; boundary at **q≈13.3** where `P` crosses zero. α could not be fitted on the inspiral — the objective returns the wrong sign |
| 2026-08-24 | git | higher modes on the inspiral bases | **(3,3) transfers free** — at q≥4 it is *better* than the (2,2) it came from; best-constant ρ = 0.94–1.12, so ρ=1 is near-optimal |
| 2026-08-24 | git | `Ehat(x)` universality measured | **the energy drive is NOT a q-universal function of x** — spreads ~2× at mid-inspiral. So an E-driven model is an x-model with a *q-dependent* shape |
| 2026-08-25 | git | **x_drive** per-q + ν-regression | at 4 params/q, **x and E are the same drive** (6.069e-4 vs 6.029e-4, 0.7% apart). With free shape + the measured cut: median **3.667e-4**, max **4.692e-4** — beats fluxanchored in-range, **4.3× worse at q=2** |
| 2026-08-25 | git | x validity measured | x's rate peaks at **−5.2 M at every q**; saturates at **+24.2 to +26.6 M**; only 0.75–1.26% of its range lies past saturation |
| 2026-08-25 | git | **mr_only** — merger-ringdown only, QNM anchor | **the QNM anchor is free**: imposing B1 gives median 8.138e-4 at **3** params/q vs 8.204e-4 at 4. The 6.2× `gwr_betaqnm` price was the cost of imposing it *everywhere*, not of the anchor being wrong |

---

## Shape of the story

Three eras, each ~3 weeks:

1. **PN-optimised (Jul 2–19).** Start at q=5, extend to 64-q grids, discover that the
   regression *coordinate* — not the degree — governs q<3 extrapolation.
2. **Physical / measurement-driven (Jul 21 – Aug 1).** Peaks, PN-anchored, then the
   energy drive: replace fitted machinery with derived or measured structure.
3. **Anchors and objectives (Aug 2 – 25).** `X1^(6/5)`, the QNM ratio, and finally the
   realisation that beta's falling drift was never a parameterisation problem but a
   property of the full-window objective — which is what the window-restricted runs
   (`inspiral_only`, `mr_only`) and the coordinate swap (`x_drive`) were built to test.

## Open items carried forward

* Two-term alpha (`E` for the post-merger floor + `F` for the merger ramp) — indicated by
  the (4,4)-at-q=2 defect, never tried.
* `alpha_E` and `beta_PP` degree choices in `stiff` — measurements do not support the
  shipped values (flagged 2026-08-03, still open).
* fdomain `a_PP` consistency question — the prime suspect for that model's poor mismatch.
* Higher-mode Step 2, `rho_lm(t) = 1 + rho_E,lm·E(t)`.
* x_drive: `a_1 < 0` enforced at ka=2, to keep the second alpha term without breaking
  monotone-falling alpha.
