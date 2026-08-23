# Peaks-based BHPT→NR model — RESUME / handoff

## 2026-07-23 — STAGE-2 MODEL BUILT & SHIPPED ✅
Production model complete and validated. Files:
- `fit_scaling_peaks_stage2.py` — Stage-2 fit + validation; writes `peaks_results/stage2_coeffs.json`.
- `BHPTNRPeaks.py` — `generate_peaks_calibrated(q)` generator (mirrors BHPTNRClaude); caps at
  NR-validated window by default, `t_start=None` for full BHPT inspiral. No NR needed (φ0=0).
- `scaling_peaks.md` — full writeup + honest limitations. CLAUDE.md model table updated.
- `peaks_results/plots/stage2_validation.png` — q=2/3/5/8 recon vs NR.

**Architecture** (the fix vs the original x-map plan): centered-**Chebyshev** in x (α deg5, β deg4)
→ **ν-regression** PP-anchored (deg4); **TIME-MAP reconstruction** (evaluate model α,β ONLY at the
peaks, build the BHPT→NR map by interpolating through peaks in time). Trained **[2.25, 8]** so the
x-norm lower bound is real training data (q=2's early inspiral otherwise clips fragilely).

**Performance** (mathcalE vs NRHybSur3dq8, inspiral-through-merger, 98% coverage):
in-range [3,8] median **0.26%** (max 0.37%); anchors q2.75/2.5/2.25 = 0.38/0.33/0.27%;
**q=2 extrapolation = 0.55%** (clears <1% gate). q1.75=4.6% (fails, 2 steps out).

**Honest caveats (documented in scaling_peaks.md):** (1) in-band 0.26% is ~40× worse than the L2
models (7e-5) — peak ratios are coarser; this is a physical/low-q method, not an in-band
replacement. (2) Ringdown (~2% energy) NOT modelled — window ends at merger, so NOT directly
comparable to remnant_partial's full-window 0.37%. (3) q<2.25 degrades fast.

**Remaining optional work:** QNM/remnant ringdown branch (step 3 below) for full-window coverage
& a fair head-to-head vs remnant_partial. Machinery in `fit_scaling_PN_opt_remnant_partial.py`.

---
## (historical) status before the build

Status snapshot for the "using the peaks" model (Islam & Khanna arXiv:2307.03155,
Sec. II.A.3). **Paused before building the regression layer.** Everything below is
validated and cached; pick up at "NEXT STEP".

## Goal
q-in → calibrated (2,2) waveform, like `BHPTNRClaude.py`, but with α,β obtained
from the paper's peak-ratio measurement instead of an L2 waveform fit.
Hard requirement: **q=2 waveform mathcalE < 1%** (beat `remnant_partial`'s 0.37%).

## Method (paper's Sec. II.A.3), as implemented
At each waveform peak k (counted from merger, phase-defined extrema spaced by π
in unwrapped GW phase — robust, phase-offset-immune):
- `alpha_peak(k) = |h|_NR / |h|_ppBHPT`         (amplitude ratio)
- `beta_abs(k)   = t_NR / t_ppBHPT`  (merger-relative)  ← **CHOSEN** (paper-literal)
- each peak tagged with `x = (M ω_orb)^(2/3)` from the ppBHPT side (monotone
  reparam of the paper's ω_orb; PN corrections are low-order polys in x).

Uses only the RAW `calibrated=False` ppBHPT surrogate + NRHybSur3dq8, both already
cached in `.cache/q_dep/` (64 q's in [3,8] + q=1.5,1.75,2,2.25,2.5,2.75,9,9.2...,10).

## DECISIONS (locked)
1. **β = absolute ratio** `t_NR/t_BHPT`, not slope. In the (x,ν) regression it fits
   ~2.2× cleaner (RMS 1.29e-3 vs 2.88e-3); slope's only edge (merger-safety) is moot
   because the QNM branch owns the ringdown. Also paper-literal.
2. **Inspiral coordinate = ν** (PP-anchored: correction ∝ ν, →0 as ν→0 / q→∞).
   NOT χ_f: χ_f is degenerate for the inspiral once spin is added (many χ1,χ2 → same
   χ_f). User directive: **χ_f only in the ringdown branch**, where the remnant is the
   whole story.
3. **Ringdown β = QNM/remnant** via `beta_r_physical` (lifted from
   `fit_scaling_PN_opt_remnant_partial.py`), which IS legitimately χ_f-based.
   CAVEAT: raw `beta_r_physical` overshoots (~0.973 at q=5 vs measured merger ~0.88 and
   creative `beta_r=0.846`). remnant_partial fixes this with a learned `r_beta(χ_f)`
   correction factor — we must do the same (do NOT inject raw QNM at merger; that was
   the 3.3% blow-up in the first pass).
4. Ringdown α and near-/post-merger β are NOT taken from peak ratios — those ratios
   are meaningless there (α blows up to ~6.8 in ringdown as amplitudes decay). The
   paper imposed the ringdown falloff artificially; we use QNM physics. Confirmed OK
   with user (our inspiral α is flat ≈ mass-scale 1/(1+1/q), matching paper Fig.3).

## VALIDATED RESULTS
- **Reconstruction machinery is correct.** Abs-ratio time map `τ(t)=β_abs(x)·t`
  (merger-relative) with merger phase-alignment + 1-D φ0 refine:
  measured-α,β recon at q=5 → **mathcalE = 2.1e-4** (rate-integral map gives same 2.2e-4).
- **Method ceiling per q (own measured α,β, interpolated):**
  q5 0.021%, q3 0.042%, q2.5 0.054%, **q2 0.066%**, q1.5 0.079%.
  → q=2 < 1% is ACHIEVABLE; only the regression layer can throw it away.
- **Joint (x,ν) polynomial regression (α: Mx=4,Nnu=3; β: Mx=3,Nnu=2), x clipped to
  measured range, no switch:**
  in-range [3,8] median **0.29%** (max 0.32%) — regression-limited (α is the bottleneck,
  fit RMS 5e-3 vs β 1.3e-3);
  q<3 EXTRAP FAILS: q2.75 0.57%, q2.5 2.5%, q2.25 10%, **q2 34%**, worse below.
  → the joint poly does not preserve the clean low-q shape once ν leaves [0.099,0.1875].
- β coefficient extrapolation looks deceptively smooth (β(x=0.1): q→∞ 1.000 exact,
  q8 0.876, q5 0.822, q2 0.705) — but smooth coeffs ≠ good waveform (same nu_Pade trap).

## KEY NUMBERS / CONSTANTS
- QNM (Berti Kerr 220 fit, from remnant_partial): `F1,F2,F3 = 1.5251,-1.1568,0.1292`
  `beta_r_physical = [0.3683*(1+q)/q] / [(F1+F2*(1-χf)^F3)/Mf]`,
  `(Mf,χf)=surfinBH.LoadFits("NRSur3dq8Remnant").mf/.chif(q,[0,0,0],[0,0,0])`.
- Peak x-range covered by NR window: x∈[~0.05, ~0.22] (NR-length-limited, ~last 5000M).
- ν range in [3,8]: [0.0988, 0.1875]; q=2 is ν=0.2222 (modest extrapolation).

## FILES
- `fit_scaling_peaks.py` — Stage-1 extraction (`extract(q)`, `phase_peaks`, grid runner).
- `peaks_results/per_q/peaks_q*.npz` — measured point clouds for [3,8] (64 files).
  keys: q,nu,x,t_bhpt,t_nr,alpha,beta_abs,beta_slope,omega_gw_bhpt,...
- `peaks_results/plots/` — q2_alpha_beta_measured.png, q3_fig3_style_thru_ringdown.png.
- `peaks_results/prototypes/` — validated scratchpad scripts (reconstruction + bake-off
  + diagnostics). **The reconstruction recipe to reuse lives in `proto_recon2.py`
  (poly) and `proto_diag2.py` (measured-recon ceiling).**

## 2026-07-22 SESSION — Stage-2 regression built & diagnosed (PAUSED mid-decision)

Built the 2-stage regression and ran a full diagnostic sweep. **Two architecture-changing
findings that revise the original plan below.** All new prototypes in `prototypes/`:
`proto_stage2.py`, `proto_stage2_diag.py`, `proto_sens.py`, `proto_xmap.py`,
`proto_deg.py`, `proto_timemap.py`, `proto_coord.py`, `proto_extend.py`(not yet run).

### FINDING 1 — the x-map reconstruction is a structural wall at q=2 (~2%)
The original planned recipe (proto_recon2, `τ=β(x_sample)·trel` evaluated per BHPT sample)
CANNOT meet the q=2 gate, independent of regression quality:
- `proto_xmap.py`: at q=2 the x-map self-fit (smooth poly through q=2's OWN measured points)
  = **4.49%**; the TIME-map interp of the same points = **0.066%** (the true ceiling).
- `proto_deg.py`: sweeping β x-degree 3→10 leaves q=2 stuck at **1.8–2.4%** — degree doesn't help.
- Cause: a smooth β(x) leaves ~1–2.5e-3 RMS; at q=2's long inspiral that time-stretch error
  accumulates fatal phase. The time-interp escapes it by being near-exact at the dense peaks.

### FINDING 2 — TIME-MAP reconstruction fixes it; best pure-[3,8] q=2 = 2.04%
New recipe (`proto_timemap.py`): evaluate the model β,α **only AT THE PEAKS** (x clean there),
set `trel_nr(k)=β_k·trel_bhpt(k)`, build the BHPT→NR time map by interpolating through the
peaks in TIME, and interpolate α in time too. Avoids near-merger x-scrambling.
- Centered-x **Chebyshev** basis (conditioning: monomial max|coeff| 1.4e4 → centered 0.30, `proto_sens.py`).
- Best config: α x-deg 5, β x-deg 4; coeffs regressed in **ν** PP-anchored `coeff(ν)=ν·poly(ν)`;
  **β ν-deg 4 is the sweet spot** (deg3 underfits→49%, deg5 overfits/oscillates→27% at q=2).
- Result: in-range[3,8] median **2.6e-3**, max 3.2e-3; low-q q2.75=0.27% q2.5=0.43% q2.25=0.95%
  **q2.0=2.04%** q1.75=5.9%. β is the sole q=2 bottleneck (regressed-α matches the ceiling; `proto_sens.py`).
- `proto_coord.py`: ν beats 1/q and δ=√(1−4ν) decisively for the coeff regression.

### HONEST ASSESSMENT (the paused decision)
Pure [3,8]→q=2 extrapolation tops out at **~2%**. This does NOT meet the <1% gate and does
NOT beat `remnant_partial` (0.37%, also pure [3,8] extrapolation via χ_f-remnant physics).
Also **in-range 2.6e-3 is ~40× worse than the L2-fit accepted models (7e-5)** — peak-ratio
α,β are inherently coarser than L2-optimized scalings. So the peaks model currently wins on
NEITHER front. The theoretical 0.066% q=2 ceiling is only reachable WITH q=2's own NR data.

### DECISION NEEDED ON RESUME (paths forward)
- **(A) Extend training range down to q=2.25 (or 2.5)** using cached low-q waveforms
  (NOT q=2 itself) → turns q=2 into near-interpolation. `proto_extend.py` is written to test
  exactly this (compares [3,8] / [2.75,8] / [2.5,8] / [2.25,8]) — **RUN IT FIRST on resume**;
  it was interrupted before running. Likely clears <1%. Caveat: weaker claim than remnant_partial
  (which uses only [3,8]).
- **(B) Add QNM/remnant ringdown branch** (old step 3) — will polish merger/ringdown but q=2 is
  dominated by the INSPIRAL time-stretch, so unlikely to fix q=2 to <1% on its own.
- **(C) Physically anchor inspiral β** to a PN time-stretch so ν-extrapolation is tethered (research).
- **(D) Accept ~2% and document** the peaks method as underperforming the L2/remnant approach.

Recommendation: run (A) `proto_extend.py`; if [2.25,8] clears <1%, promote the time-map 2-stage
into a production `fit_scaling_peaks_stage2.py` + `BHPTNRClaude`-style generator, and frame the
model honestly as low-q-anchored. Otherwise report (D).

---
## ORIGINAL PLAN (SUPERSEDED by Findings 1–2 above — step 1-2 done, x-map part revised)
Replace the joint (x,ν) poly with a **2-stage regression** to preserve the 0.066% ceiling:
1. Per q in [3,8]: fit α(x), β(x) as low-order x-polynomials to the measured points
   (few shape params each — the curves are clean).  ← DONE (use Chebyshev, centered x)
2. Regress those per-q x-coefficients smoothly in **ν**, PP-anchored (each coeff → its
   ν→0 limit; α,β→1). Try degree bake-off.  ← DONE (ν, α-deg5/β-deg4, β ν-deg4)
3. Ringdown β: `r_beta(χf) * beta_r_physical`, `r_beta` fit vs χ_f (remnant_partial style),
   blended in via a logistic switch in x near merger (x0≈0.22).  ← NOT DONE (see path B)
4. Validate: in-range [3,8] mismatch (target →1e-4) AND q=2 (+2.25,2.5,2.75,1.75,1.5)
   waveform mathcalE (**gate: q=2 < 1%**). Compare to measured ceiling.
Then Stage-3: wrap into a `BHPTNRClaude`-style `q → (t,h)` generator + scaling markdown.

Reconstruction recipe — NOTE: use the TIME-MAP variant (proto_timemap), NOT the x-map below.
Old x-map recipe (from proto_recon2/diag2), SUPERSEDED — capped at ~2% at q=2:
```
x(t)   = ((|grad(unwrap(angle(h_bhpt)))|)/2)^(2/3)
τ(t)   = β(x;ν)·(t - t_bhpt_merger) + t_nr_merger      # abs-ratio map
h(τ)   = α(x;ν)·exp(i·φ0)·h_bhpt(t)                    # φ0 from merger phase, then 1-D refine
mathcalE via creative.mathcalE_error on the common NR support
# x clipped to measured [x_lo,x_hi] for poly eval; switch to QNM β_rd for x>~0.22
```
```
