# RESUME — gw_remnant energy/flux model line (as of 2026-08-03)

Everything below is verified working: all scripts import clean, all coefficient files and
caches exist, and `--model stiff` still reproduces 7.0098e-04 at q=5 after the refactor.

**This workspace is NOT under version control** (no `.git`). Everything lives on disk only.

## Also in this line, and NOT covered below: frequency-domain alpha (2026-08-12/13)

Everything below this section predates it.  `scaling_fdomain_alpha.md`,
`fit_scaling_fdomain_alpha.py`, `fdomain_alpha_results/coeffs.json`,
`diag_alpha_{fdomain,omega}.py`, `NRBHP_fdomain_alpha_plots.py`,
`Agentic_plots/fdomain_alpha/`.  Built on the per-q `mult` params, so it belongs to this
model line.

Headline: `alpha(f) = sqrt(P_NR/P_pp)/a_PP` is a q-universal S-curve 0.970 -> 1.015 whose
deg-4 residual is ~9e-04 against ~2.6e-03 in the time domain — **3x smoother** — and it
gets *easier* toward low q (7.0e-04 at q=2).  The regressed 18-coefficient model exists,
but its waveform mismatch (in-range median 1.171e-03, q=2 3.555e-03) is 2-3x WORSE than
`fluxanchored`, so the smoother shape has not yet bought a better model; the suspected
cause is the `a_PP` consistency question in that writeup.  **Beta is untouched there** — it
takes `b_PP, b_E, t0_nr` from the `mult` cache — so it inherits the falling-beta behaviour
and is orthogonal to the beta work below.

## Where things stand

Current best (2,2) model: **`gwr_energy_fluxanchored`**, 7 coefficients.

| model | coef | in-range med | max | q=2.75 | q=2.5 | q=2.25 | q=2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| fluxanchored, **seeded** | 7 | 4.90e-04 | 7.16e-04 | 6.46e-04 | 5.54e-04 | 5.02e-04 | **7.59e-04** |
| fluxanchored, global | 7 | 4.75e-04 | 7.15e-04 | 6.35e-04 | 5.94e-04 | 6.87e-04 | 1.31e-03 |
| anchored (E alpha) | 6 | 6.26e-04 | 9.79e-04 | 9.48e-04 | 9.28e-04 | 1.02e-03 | 1.53e-03 |
| flux (uniform deg-3) | 14 | 4.70e-04 | 7.16e-04 | 7.09e-04 | 1.72e-03 | 8.70e-03 | 4.60e-02 |
| stiff | 9 | 6.24e-04 | 9.74e-04 | 9.83e-04 | 1.12e-03 | 1.73e-03 | 3.79e-03 |
| gwr_energy_global | 14 | 6.26e-04 | 9.70e-04 | 9.9e-04 | 1.40e-03 | 4.9e-03 | 2.8e-02 |

## The two results that organise everything

1. **alpha's COORDINATE controls in-band accuracy; the REGRESSION LAYER controls
   extrapolation.**  They are independent, which is why combining them worked.
   Swapping E -> F = Edot/max(Edot) buys 25% in-range at fixed parameter count and does
   nothing at q<3; anchoring both prefactors to the derived `X1^(6/5)` and dropping
   degrees buys 18x at q=2 and does nothing in-range.
2. **The joint ("global") fit helps only when the forms are too stiff to reach the per-q
   floor.**  For `stiff` it helped everywhere; for `fluxanchored` the forms already sit at
   the floor, so it buys 0.15e-04 in-range and costs 5.5e-04 at q=2.  Hence the seeded
   variant is the better model there.

## Reproduce

```bash
conda activate ut_claude          # OMP/BLAS threads <= 4
python fit_scaling_gwr_energy_fluxanchored.py --global --maxiter 80    # best model
python fit_scaling_gwr_energy_anchored.py     --global --maxiter 80
python fit_scaling_gwr_energy_flux.py         --global --maxiter 40
python NRBHP_gwr_global_plots.py                                       # 3x15 PDFs
python NRBHP_hm_backbone_fig8_plots.py --q 2 4 \
    --model E_deg3 flux_deg3 E_anchored flux_anchored \
    --outdir consolidated_gwremnant_hm                                 # 8 PNGs
```

Caches that are expensive to regenerate and are already present:
`gw_remnant_energy_results/per_q_cache_mult.json`,
`gwr_energy_flux_results/per_q_cache_flux.json`,
`gwr_energy_flux_results/per_q_floor_lowq.json`,
`gwr_energy_stiff_results/per_q_truth_lowq.json`,
`.cache/q_dep/` (80 waveforms), `.cache/hm/` (5 multi-mode).

## Open, in priority order

1. **Two-term alpha — the one real defect.**  `alpha = alpha_PP(1 + alpha_F F)` snaps back
   to `alpha_PP` after merger because F is peak-normalised and non-monotonic, while the
   empirical alpha keeps falling through ringdown.  Visible in the q=2 params panel of
   `Agentic_plots/gwr_global_fluxanchored/`, and it **costs a mode**: at q=2 the flux base
   beats the E base on (2,2) and (3,3) but loses (4,4), 6.41e-02 vs 4.87e-02.
   Fix: `alpha = alpha_PP (1 + alpha_E E + alpha_F F)` — E for the post-merger floor, F for
   the merger ramp.  E and F are NOT degenerate the way `wf_nu_fluxes`'s two flux terms
   were, so this should regress.  Costs one coefficient.

   **Measured properties of F, before building on it (2026-08-11).**  F is rougher and
   worse-centred than "peak-normalised flux" suggests:

   | q | F peaks at | F at merger | pre-merger steps increasing | worst backward step |
   |---:|---:|---:|---:|---:|
   | 3 | **-15.0 M** | 0.6428 | 63.9% | -75.6% of range |
   | 5 | **-15.0 M** | 0.7223 | 68.7% | -62.7% of range |
   | 8 | **+6.0 M** | 0.8456 | 68.6% | -45.6% of range |

   So (a) **F does not peak at merger** — it peaks ~15 M early at q=3,5 (and 6 M late at
   q=8) and has already fallen to 0.64-0.85 by merger, so the "merger ramp" it supplies is
   not centred on the merger; and (b) `Edot` oscillates on the ORBITAL timescale, so only
   ~65% of F's pre-merger steps increase, with backward excursions up to 76% of its range.
   Neither is a defect of `fluxanchored` (it fits fine, and `alpha_F` only ever multiplies
   F), but both matter for item 1: if the second term is meant to supply a merger ramp, a
   SMOOTHED and merger-centred flux is likely the better basis than raw F.  This is also
   why the two-term BETA idea (E + F with both coefficients positive) does NOT give a
   guaranteed-rising inspiral — see the KEY BETA RESULT section.  Measurement is inline in
   the session log; regenerate with the `add_flux` output and `argmax(|h|)`.
2. **`alpha_F` and `P` nu-forms.**  They carry the *entire* remaining q=2 margin: at q=2 the
   X1^(6/5) prefactors extrapolate to +0.9% and -0.2%, while `alpha_F` is off -18.0% and
   `P` +23.0%.  Master/floor at q=2 is 1.78x, so better forms could take q=2 from 7.6e-04
   toward the measured floor of 4.3e-04.
3. **Robustness of fluxanchored to `--nperq` / `--ntrain`.**  A jump this large deserves a
   sensitivity check.  Untested.
4. **Consolidate the plotting scripts.**  `NRBHP_gwr_global_plots.py` is the generalised
   one and could absorb `NRBHP_gwr_energy_stiff_plots.py` and
   `NRBHP_gwr_energy_anchored_plots.py`.  Verify it reproduces their numbers first — it
   optimises t0 directly on the full grid where they use a fit-then-full two-step (agrees
   to 5 digits where checked).
5. Higher-mode Step 2 (`rho_lm(t) = 1 + rho_E,lm E(t)`), skipping (3,3) and skipping phase
   residuals on (4,4).  Downstream of item 1.
6. **Make `beta(t)` rise at every q, with no post-merger correction** (requested
   2026-08-11).  Three candidate routes, measurements and settled non-fixes in the
   section "Making beta rise at all q" below.  Start with the phase-gauge test: it is the
   cheapest and its answer decides whether the other two are worth doing.

## WHY the drift flips at q~4 — SETTLED 2026-08-20, read this before the two sections below

`Agentic_plots/beta_transition_q.pdf` (`NRBHP_beta_transition_q.py`) puts every model's
crossing on one axis: **3.799 … 4.003, mean 3.92**, across the alpha drive coordinate
(E vs Edot), 6 / 7 / 9 / 14 coefficients, per-q→regress vs global joint fit, and the two
raw per-q grids that have no regression at all.  `Agentic_plots/beta_transition_why.pdf`
(`NRBHP_beta_transition_why.py`) explains it, and **nothing happens at q=4**:

1. The derived anchor gap `g_b = (B1-B0)/B0` (B0 = X1^(6/5), B1 = QNM ratio; both verified
   to <1% against direct `omega_pp/omega_NR`) is POSITIVE at every q and decays
   monotonically: **+7.04% (q=3) → +3.82% (q=4) → +0.96% (q=8)**.  The true drift rises
   everywhere; the physics never flips.
2. The fitted drift sits a featureless **2–4 pp BELOW g_b** (−1.9pp at q=3, −4.3pp over
   3.8–4.6, −2.2pp at q=8), so `Delta beta = g_b − deficit` crosses zero where g_b decays
   past ~4% → **q = 3.80**.  The crossing is where the ANCHOR GAP SHRINKS PAST THE MODEL'S
   SHORTFALL.  The shortfall is the form's: one coefficient, linear in a drive that does
   66–82% of its range in the last 2% of the window, must serve both the inspiral climb and
   the terminal QNM plateau.  E(t)'s shape is q-quasi-universal, hence so is the deficit.

   **Localised to beta's FREE END** (`beta_drift_tests/beta_endpoints_vs_anchors.py`):
   `beta(start)` matches `X1^(6/5)` to **0.08–0.46% at every q in [2,8]**, while `beta(end)`
   vs the QNM `B1` misses by +3.95% / +0.34% / −1.95% / −3.79% / −3.68% / −2.08% at
   q = 2 / 2.5 / 3 / 4 / 5 / 8 (crossing zero at q~2.6 — the same curve as the drift
   crossing, displaced by g_b).  `B1` appears NOWHERE in `mult`/`anchored`/`fluxanchored`,
   so nothing makes the terminal beta land on it.  At q=8 the per-q fit reaches 2.46e-4
   while its terminal beta is 2.1% wrong: `sqrt(2E)` ~ 2.2%, so a 2% error in a quantity
   controlling the last 2% of the power hides inside a 0.025% mismatch.  **A small mismatch
   does not certify the ingredients — that is what the anchors are for.**

   **CORRECTION to the older framing in this file and in `ringdown_only_mismatch.py`'s
   write-up:** "the ringdown is only ~2% of the energy so L2 ignores it" is HALF WRONG.
   The power fraction after t=0 is indeed 1.79–2.34% (q=8/5/3), but the post-merger region
   carries **32–38% of the mismatch NUMERATOR** — ~20x over-represented relative to its
   power, precisely because that is where the model fits worst.  The fit is not indifferent
   to the ringdown; it cannot satisfy it with a single linear-in-E term.  Do not re-argue
   the "objective doesn't care" version.

   Same pathology on the OTHER parameter: in the flux models alpha is anchored early and
   free after merger, so it snaps back to `alpha_PP` while the empirical alpha keeps falling
   — the documented (4,4)-at-q=2 defect.  Both alpha and beta are pinned early and
   unconstrained late; the indicated two-term alpha (E for the post-merger floor + F for the
   merger ramp) is structurally the same fix as pinning beta's terminal value.
3. Why all models agree to 5%: `dDelta/dq ≈ −2.6 pp` per unit q at the crossing while the
   inter-model scatter in `Delta beta` at fixed q is only 0.1–0.4pp → 0.04–0.15 in q.  A
   smooth curve crossing an arbitrary level.  **`b_E = 0` is not a special point of the
   mismatch, which only ever sees `int(beta dt)`.**  So the tight clustering is arithmetic,
   NOT evidence of a physical scale — do not write it up as one.

**RETRACTED by this run: the tug-of-war story** ("inspiral wants rising, merger wants
falling, q~4 is where they cancel").  Splitting the mismatch numerator exactly by region —
additive at fixed t0 and phi0, `beta_drift_tests/transition_q_decompose.py` — gives the
SAME sign in both regions at every q, crossing within Dq = 0.4 of each other (inspiral 4.04,
merger 3.66).  They flip together.  The earlier "inspiral wants b_E > 0 at every q"
(`inspiral_only_sign.py`) was PROFILED — it refit `beta_PP` — and that is a degeneracy, not
an independent pull: moving `beta_PP` by ±0.5% drops inspiral-only error by up to 13x and
shifts the preferred `b_E` from −1.0 to +0.5…+2.0 (at q=8 the conditional curve varies x89
over the b_E grid, the profiled one only x8.3; `beta_drift_tests/be_bpp_degeneracy.py`).
The inspiral constrains `int(beta dt)`, not beta's slope, so it accepts a rising beta nearly
free but has almost no say in where the crossing lands.  **The merger sets it.  Reweighting
the objective toward the inspiral will not move the crossing; pinning the terminal value
will** — which is `fit_scaling_gwr_betaqnm.py --parfree`, crossing-free by construction.

## beta's LEVEL is physical; its DRIFT is gauge (2026-08-03) — read before "fixing" P

The models have `P < 0` for q > ~3.9, so `beta(t)` **falls** with time above that (P=0 at
q = 3.892 / 3.911 / 3.937 / 3.942 / 4.002 for fluxanchored-global / -seeded / anchored /
flux-deg3 / E-deg3; full list in the section above).  It is tempting to call this unphysical, since beta is roughly the
ratio `omega_pp / omega_NR` and one expects that ratio to rise through inspiral.  **That
argument does not hold, and the plots are not wrong.**

The exact phase-matching map is `tau(t) = phi_NR^{-1}(phi_pp(t) + dphi)` with **dphi a free
constant** — the model's own analytic `phi0`.  So "the empirical beta(t)" is a
one-parameter FAMILY, and its drift sign depends on which member you pick.  Measured, the
drift over (window start -> tau = -20 M) as a function of dphi:

| q | dphi=-4 | dphi=-2 | dphi=0 | dphi=+2 | dphi=+4 |
|---:|---:|---:|---:|---:|---:|
| 3 | +63.8% | +36.1% | +11.4% | -0.8% | -11.7% |
| 5 | +52.5% | +27.7% | +5.7% | -4.7% | -14.0% |
| 8 | +45.3% | +22.7% | +3.0% | -6.4% | -14.7% |

A shift of ~2 rad (a third of a cycle) flips the sign.  Aligning the two mergers (dphi=0)
gives a rising beta; the L2-optimal member gives a falling one above q~4.  Both describe
the same waveform pair.

**The decisive check:** the shipped model's own phase residual is **0.017-0.046 rad RMS**
over ~50 cycles (~300 rad accumulated) — 0.029/0.030/0.017/0.046 r at q=3/5/8/2, i.e. the
time map is right to ~1 part in 1e4, entirely consistent with the 2.3e-4..1.3e-3
mismatches.  There was never room for a percent-level time-map error.

**What IS gauge-robust is the LEVEL.**  Measured beta at the window start is 0.8685 at q=8
against `X1^(6/5) = 0.8682` — 0.04%, matching the 0.03% agreement `beta_PP` shows against
the same anchor.  So the X1^(6/5) result stands; only drift interpretations are suspect.

**Consequence for open item 2:** the target for `P` is *"make it extrapolate"* — a fitting
question — NOT *"make it physical"*.  The Bondi-mass agreement at q=3 (1.1%) compares
quantities in different gauges and may be coincidental, so
`scaling_gwr_energy_stiff.md` §"Where the physics runs out" over-reads it.

## RESOLVED: beta's falling drift is an ARTEFACT OF THE FULL-WINDOW L2 OBJECTIVE (2026-08-18)

**This supersedes the interpretation of every beta section below.  The falling beta above
q~4 is not a property of the waveforms — it is a property of the cost function.**

Test (`beta_drift_tests/ringdown_only_mismatch.py`): score the SHIPPED `fluxanchored`
coefficients (beta falls) against the same alpha with `b_E` replaced by the parameter-free
QNM anchor value (beta rises), on three windows:

| q | full window | ringdown only (t>0) | QNM only (t>+20) |
|---:|:---|:---|:---|
| 3 | shipped wins 6.8x | **anchor wins 2.7x** | **anchor wins 1.9x** |
| 5 | shipped wins 33x | **anchor wins 5.3x** | **anchor wins 2.6x** |
| 8 | shipped wins 83x | **anchor wins 5.5x** | **anchor wins 2.4x** |
| 2 | shipped wins 7x | shipped wins 2.6x | shipped wins 5.8x |

**The winner flips when only the ringdown is scored.**  The ringdown carries ~2% of the
energy, so the full-window fit trades a factor ~5 of ringdown accuracy for a few percent
in the inspiral-merger and wins on the metric.  The rising beta is right; L2 does not
reward it.  The effect GROWS with q (2.7x -> 5.5x), i.e. it is worst where beta's required
rise is smallest and easiest to ignore.

**q=2 is the exception and is NOT explained** — there the shipped model wins on the
ringdown too (2.6x, 5.8x).  The anchor demands a +15.6% rise there, far larger than
anywhere else, and q=2 is also outside the BHPT surrogate's `X_min = log10(2.5)` domain so
the ppBHPT ringdown itself is suspect.  This test cannot separate "anchor over-predicts at
q=2" from "input waveform is bad at q=2".

**Consequence: the five failed attempts below all shared the same objective, and that was
the confound.**  `b_E >= 0` (2x), the four-drive test (sign unchanged), the phase gauge
(worse), and the anchor-coupling models `gwr_anchored2`/`gwr_anchored3` (4-37x) each tried
to change the PARAMETERISATION while keeping full-window L2.  None could have worked.  The
lever is the COST FUNCTION, not the form.

**Also likely explains the higher-mode (4,4) failure** (open item 1 and
`scaling_hm_backbone.md`): higher modes carry more merger-ringdown weight, so a base that
is ~5x wrong in the ringdown should fail on them — which is exactly the observed
(4,4)-at-q=2 pattern.  **Untested; this is the first thing to check.**

## THE KEY BETA RESULT — the inspiral wants a RISING beta at every q (2026-08-11)

**Read this before anything else in the beta sections below; it corrects the working
assumption of every earlier run.**  `beta_drift_tests/inspiral_only_sign.{py,log,json}`
refits the 4 per-q params with the NR comparison window TRUNCATED (merger and ringdown
simply not scored):

| q | full-window b_E | inspiral only t<=-200 | t<=-500 | cost of forcing opposite sign |
|---:|---:|---:|---:|:---|
| 3 | +1.2907 | **+3.1677** | **+3.2676** | x96.2 / x90.4 |
| 5 | **-0.8187** | **+2.0721** | **+2.5819** | x27.6 / x44.2 |
| 8 | **-1.4777** | **+0.3576** | **+1.5433** | x1.34 / x5.23 |

**The inspiral wants beta RISING at every q, including q=8**, and strongly (x90 at q=3,
x44 at q=5; q=8 is weak at the -200 cut, x1.34, firming to x5.2 with more inspiral).

**So "the data wants a falling beta above q~4" is WRONG as previously written in these
notes.**  The correct statement: the INSPIRAL wants rising, the MERGER-RINGDOWN wants
falling, and single-term `beta = beta_PP + P*E(t)` has ONE sign to serve both.  E(t) does
66-82% of its range in the last 2% of the window, so the merger region wins the vote and
the inspiral is never consulted.  Corroborating: inspiral-only mismatches are 1e-06..1e-05
against 1.5e-04..6.7e-04 full-window, i.e. the model fits the inspiral 30-100x better and
essentially all leverage on `b_E` lives in the last few hundred M.

This retro-explains the rest: the GATED models have a separate merger branch (`beta_r`) so
their inspiral term is free to do what the inspiral wants (hence `beta_L`>0); pn_anchored's
PEAK is exactly inspiral-rise-then-merger-branch-pulls-down; and the four-drive test
returned one sign for all four drives because all four were single-term and scored
full-window — the merger's vote, four times over.

**INDICATED BUILD (no gate needed): make beta's E-dependence NONLINEAR so the two regions
get separate control.**

    beta(t) = beta_PP(q) * (1 + b1(nu)*Ehat + b2(nu)*Ehat^2),   Ehat = E/E(merger)

`b1 >= 0` carries the inspiral rise (Ehat small there, linear term dominates); `b2` free
and negative bends beta over near merger (Ehat -> 1).  Gives rise -> peak -> settle with
E(t) as the only drive, no switch, no post-merger branch.  Inspiral monotonicity is
`b1 + 2*b2*Ehat > 0`, which holds up to Ehat ~ 0.2-0.4 (the inspiral's share of E's range,
measured: 23.1/32.2/41.4% at q=3/5/8 for the -200 cut) for a wide range of `b2` — enforce
with a penalty and verify per q, do not assume.

**Also note `beta_monotone` (below) does NOT meet the requirement**: it is FLAT above
q=5.75, and flat is not increasing.  It solved "non-decreasing over the whole window",
which was my mis-reading of the ask.

## Making beta rise at all q — pipeline (opened 2026-08-11)

**Goal (user's, 2026-08-11):** `beta(t)` increasing at every q, achieved *without* a
post-merger correction term.

**Why it is one scalar, not a behaviour.**  `beta = beta_PP(q)*(1 + b_E*E(t))` and E(t) is
monotone increasing (verified q = 3,4,5,6,8), so `sign(dbeta/dt) = sign(b_E)` over the whole
window.  `b_E` crosses zero at q ~ 3.9 **in the raw per-q data**, on both the 16-point flux
grid (last + 0.2155 at q=3.6, first - 0.0012 at q=3.8) and the 64-point E grid (+0.0229 at
q=3.897, -0.0690 at q=4.0), so it is not an artifact of the linear `P(nu)`.

**Both signs are sharply determined** (`beta_drift_tests/scan_bE.py`, full-window mismatch, other
three params re-optimised at fixed `b_E`, t0 inner nuisance):

| `b_E` | q=3 | x opt | q=8 | x opt |
|---:|---:|---:|---:|---:|
| -1.50 | 3.50e-03 | 5.21 | **1.53e-04** | **1.00** |
| 0 | 1.21e-03 | 1.80 | 3.79e-04 | 2.47 |
| **+1.29** | **6.72e-04** | **1.00** | ~9.0e-04 | ~5.9 |

Cross-substituting costs 5-6x at both ends and `a_PP`/`a_F`/`b_PP` move < 0.5%, so they
cannot absorb it.  Inside a fixed phase convention the sign is real data, not a soft
direction.

**ROUTE 1 ATTEMPT 1 IS INVALID — the fit, not the gauge (2026-08-11).**  The run completed
all three q (`beta_drift_tests/gauge_aligned.{log,json}`), but the optimisation is broken and
**the result answers nothing.**  Do not quote the "aligned gauge" column.

| q | L2 gauge | aligned-gauge "best" | aligned params @ optimal phi0 | **L2 params rescored in aligned gauge** |
|---:|---:|---:|---:|---:|
| 3 | 6.7175e-04, b_E +1.2907 | 7.4668e-02, b_E +0.9984 | 1.1898e-03 (x1.77) | **3.7200e-02** |
| 5 | 5.3095e-04, b_E -0.8747 | 7.0385e-01, b_E +0.0000 | 6.2008e-01 (x1168) | **4.1401e-03** |
| 8 | 1.5320e-04, b_E -1.4777 | 7.1955e-01, b_E -1.4777 | 5.3014e-01 (x3461) | **8.9255e-04** |

**The proof it is broken:** at every q the returned "optimum" scores WORSE in the aligned
gauge than simply rescoring the L2 parameters there (7.47e-02 vs 3.72e-02; 7.04e-01 vs
4.14e-03; 7.20e-01 vs 8.93e-04) — and the L2 parameters *were among the seeds*.  A minimiser
cannot legitimately return a point worse than its own seed.  At q=8 the returned `b_E` equals
the seed to 4 dp (-1.4777), i.e. it never moved at all.

**Cause:** phi0 is read off `argmax|g|`, an integer index that JUMPS as the parameters vary,
so the objective is discontinuous and Nelder-Mead stalls or wanders.  (The mutating `st["t0"]`
across objective calls makes it non-deterministic too.)

**Corrected reading of the gauge's actual cost** — from the last column, which is a clean
evaluation with no optimisation involved: x55 (q=3), x7.8 (q=5), x5.8 (q=8), and those are
UPPER bounds since the parameters are not re-optimised for the aligned gauge.  So the
merger-aligned gauge is expensive but **nothing like the x111/x1326/x4697 the broken fit
suggested, and route 1 is still open.**  My earlier note that q=3's x111 "really is the price
of the convention" was wrong — that row is compromised the same way as the others.

Fixed in attempt 2: phi0 now read at the FIXED NR merger time (`argmax|h_NR|` is NR data, so
it does not move with the parameters), t0 bracketed around a fixed reference, and every seed
kept as a candidate so the returned best cannot be worse than a seed.  All three self-checks
pass and the L2 branch reproduces the shipped 6.7175e-04 / 5.3095e-04 / 1.5320e-04.

**ROUTE 1 IS SETTLED — IT DOES NOT WORK (attempt 2, 2026-08-11,
`beta_drift_tests/gauge_aligned.{py,log,json}`).**

| q | L2 gauge | aligned gauge | gauge cost | drift |
|---:|---:|---:|---:|:---|
| 3 | 6.7175e-04, b_E +1.291 | 7.1009e-04, b_E +1.182 | x1.06 | RISING |
| 5 | 5.3095e-04, b_E -0.875 | 9.6460e-04, b_E **-2.156** | x1.82 | **falling, HARDER** |
| 8 | 1.5320e-04, b_E -1.478 | 8.3844e-03, b_E +7.256 | **x54.7** | RISING but unusable |

Non-monotonic (rising / falling / rising) and neither "rising" is usable: q=3 was already
rising in the L2 gauge so it carries no information, and q=8's +7.256 sits at 8.4e-03 — 55x
worse than the L2 optimum, with the negative seeds (-1.478, -1, 0) all tried and rejected, so
no good solution exists in that gauge at q=8.  **The decisive point is q=5: the aligned gauge
made the drift MORE negative (-0.875 -> -2.156)** — the opposite of what the dphi table
predicts.  So the falling drift is NOT merely the phase convention.

**Why forcing phi0 was conceptually flawed anyway:** it removes a degree of freedom, so the
phase error must be absorbed *through the time map*, which distorts beta.  The aligned-gauge
`b_E` values are compensation for a fixed phase, not "the physical drift revealed" — the
x54.7 and the +7.256 excursion at q=8 are what that looks like.

**Two corrections to earlier notes in this section:** (a) the "gauge cost bounded at x5.8-x55
from clean rescoring" figure was computed with attempt 1's phi0 definition and is superseded
— with the corrected definition, rescoring the L2 parameters gives ~0.32-0.35 at every q (a
constant ~0.84 rad offset from L2-optimal) and re-fitting recovers to x1.06 / x1.82 / x54.7.
(b) The two alignment definitions give very different phi0, so **"the merger-aligned gauge" is
itself under-determined** — anyone reviving this must say which alignment they mean.  Note
also that the built-in check (aligned fit must beat L2-params-in-aligned-gauge) is necessary
but WEAK when that reference is ~0.33; "ok" at q=8 does not mean converged well.

**Consequence: the cause is in the switchless FORM, not the convention. Go to route 2.**

**ROUTE 2 (PN-FREE FORM) IS SETTLED — IT DOES NOT WORK, AND IT KILLS THE WHOLE PREMISE
(2026-08-11, `beta_drift_tests/beta_drive_pnfree.{py,log,json}`).**  alpha left on
`flux_hat`; only beta's drive swapped, each normalised to 0 at window start and 1 at merger
so `b_D` reads as the fractional drift to merger.  All four drives are PN-free (waveform and
grid only) and monotone.  `dbeta` to merger:

| drive | range done in first 83% (q=3/5/8) | q=3 | q=5 | q=8 |
|:---|:---|---:|---:|---:|
| `e_oft` (shipped) | 10.8 / 14.5 / 18.5% | +3.28% | -1.05% | -0.91% |
| `phi` (accum. GW phase) | **69.3 / 70.0 / 70.8%** | +1.16% | -0.74% | -0.83% |
| `lnw` (log GW freq) | 18.4 / 19.5 / 21.0% | +1.70% | -1.05% | -1.05% |
| `t` (raw time) | **83.3%** | +1.81% | -1.02% | -1.23% |

**Every drive gives the same sign pattern (rise at q=3, fall at q=5 and q=8), including the
maximally inspiral-weighted ones, and the drift MAGNITUDE is nearly invariant at ~-1% above
q~4 regardless of the coordinate's time profile.**  Mismatch: no drive beats E (`phi` costs
1.33x at q=3, `t` 1.26x at q=8, `lnw` ties).

**So the drive's time profile does NOT control the sign** — the "E is a merger clock, that is
why b_E goes negative" reasoning (mine) is WRONG, as the `E_sat` null result had already
hinted.  The ~-1% drift above q~4 is a property of the data, not of the coordinate.

**This predicts the remaining routes fail too, and they should NOT be run as stated:**
- Route 2's PN version (swap to `x = (M Omega)^(2/3)`) will not flip the sign either, since
  shape is irrelevant.  And `pn_anchored`'s rising beta does NOT come from `x`: its inspiral
  beta is nearly flat (0.8677 -> 0.8690 from -2000 to -200 M at q=8) and the visible rise is
  the spike AT merger, i.e. the switch to the QNM `beta_r`.  Same story as gated
  `wf_nu_hybrid`, whose `beta_L`(q=8) = +0.0001 (zero) with a +2.4% merger step.  (Inference
  from the overlay figures, not a refit.)
- Route 3 (two-term beta, both coefficients positive) is just "some monotone increasing
  beta", and the `b_E >= 0` test already drives any such beta to exactly zero above q=4.

**BOTTOM LINE for open item 6: within a switchless single-branch beta, a rising beta above
q~4 is not achievable without paying accuracy.**  It survives a gauge change (route 1) and a
drive-shape change (route 2).  The two honest options left are (a) a GATE / second branch —
which is what every model that does have rising beta actually uses (pn_anchored, wf_nu_hybrid)
— or (b) accept the `b_E >= 0` cost, median 4.69e-04 -> 7.11e-04, which buys *flat*, not
rising.  Both need the user's call, since (a) is the post-merger structure they excluded.

**Three routes, in the order to try them:**

1. **Change the phase gauge** (attempt 1 ran and was INVALID — see the section above; still
   open, needs the continuous-phi0 fix first).  beta and phi0
   are not independent: the map is `tau = phi_NR^{-1}(phi_pp + dphi)`, so shifting dphi
   forces a different optimal time-stretch.  Per the dphi table above, a ~2 rad shift flips
   the drift sign at every q and **dphi = 0 (mergers aligned) gives a RISING beta at
   q = 3, 5, 8**.  So: replace the analytic L2-optimal phi0 with merger-phase alignment and
   refit.  Rising beta at all q from one positive coefficient, no post-merger term, still
   switchless and PN-free.  Cost in mismatch unmeasured.  Script ready:
   `beta_drift_tests/gauge_aligned.py` (q = 3,5,8; reports the aligned-gauge `b_E`, the gauge
   cost, and both cross-scorings to separate gauge cost from parameter quality).  ~15 min.
   **It can come back negative** — that would place the cause in the switchless form rather
   than the convention, and would kill route 1 in favour of route 2.
2. **Give beta an inspiral-weighted drive instead of E.**  E(t) is a merger-ringdown clock:
   only 7.0% (q=3) / 14.1% (q=8) of its range accumulates over the first 83% of the window,
   and 81.8% / 65.9% lands in the last 2%.  A single monotone drive with that shape cannot
   supply an inspiral rise.  Precedent, both directions: `pn_anchored` drives beta on
   `x = (M Omega)^(2/3)` and **has rising beta at all four q** (see its new overlay figure);
   the old gated `wf_nu_hybrid` used `p_loss` and got `beta_L > 0` at 55/64 mass ratios.
   Swapping E -> x or p_loss in beta is a small change.  **Cost: x is PN-flavoured, so this
   forfeits the PN-free property `gwr_energy_stiff` is kept as evidence for.**
3. **Two-term beta,** `beta = beta_PP(1 + b1*X_insp + b2*X_merg)`, both coefficients
   constrained positive.  Costs a coefficient and needs two sign constraints, which risks
   the non-smooth nu-structure that killed `wf_nu_fluxes`.  Only if 1 and 2 both fail.

**Fallback if NON-DECREASING is acceptable:** the `b_E >= 0` constraint ships today at
median 4.69e-04 -> 7.11e-04 (max -> 8.45e-04) — but see the settled list: it gives *flat*
above q=4, not rising, and the kink at q~4 will likely cost the low-q extrapolation.

## Do NOT re-run these (settled)

- **E-normalisation does not explain `A`'s non-monotonicity** — `A_p = alpha_E nu^p` is
  non-monotonic at every p tested (0.903, 1.0, 1.31, 1.449).
- **Constraining `b_E >= 0` gives a FLAT beta, not a rising one** (2026-08-11,
  `beta_drift_tests/monotone_beta.py`).  Parameterised `b_E = u^2` so the fit may pick any
  positive value, it lands on **exactly zero at every q >= 4** — there is no interior
  positive optimum; the fit refuses a rising beta rather than merely preferring a falling
  one.  Per-q cost 1.00x (q=3) / 1.32x (q=5) / 1.91x (q=6) / 2.83x (q=7) / 2.47x (q=8),
  median 4.69e-04 -> 7.11e-04.  So no choice of coefficient inside this family delivers a
  rising beta.
- **Saturating the drive at merger does NOT change the sign** (same script).
  `E_sat = min(E, E(t_merger))` — monotone, frozen after merger, zero new coefficients —
  moves `b_E` only from -1.478 to -1.484 at q=8 (and -0.073 -> -0.163 at q=4, -0.875 ->
  -0.926 at q=5).  This **refutes the post-merger-forcing explanation**: the negative drift
  originates in the inspiral-merger region where the L2 weight is, not in the ringdown.
- **Two relabelings that look like fixes and are not:** driving beta on the Bondi mass
  `M = 1 - E` flips the coefficient's sign with *zero* change to beta(t) (`M - 1 == -E`);
  re-anchoring E to the merger changes what `beta_PP` means but not `d beta/dt`.
- **The gated-vs-switchless comparison is CONFOUNDED and does not settle the sign.**
  `wf_nu_hybrid` fits q=8 2.2x better (7.06e-05 vs 1.52e-04) *with* a rising beta, but it
  changes the functional form and the effective phase gauge at once.  It motivates routes 1
  and 2 above; it does not decide between them.
- **Fitting the invariant `Q = alpha_PP alpha_E = dalpha/dE` is much worse** (-29% at q=2),
  despite being the exact analogue of the beta-side `P` trick.
- **Pinning `alpha_PP` does not de-degenerate `alpha_E`** — they are -0.974 correlated but
  alpha_PP's excursion is 0.17% against alpha_E's 6.3%; alpha_E is the soft direction.
- The nonlinear-excursion regulariser (`--lam`) hurts monotonically; ship `--lam 0`.
- **Constraining `P` to be PP-anchored and positive costs a lot, on BOTH alpha
  coordinates.**  `P = p0*nu` (1 coef, seeded positive) or `P = exp(u)*nu` (positivity
  enforced): in-range median goes 6.26e-04 -> 2.23e-03 on the E-anchored model (3.6x) and
  4.75e-04 -> 2.12e-03 on the flux-anchored one (4.5x); q=2 goes to ~2.2e-02 in both
  (14-17x).  Pure Bondi `P = (6/5) beta_PP` (0 coef) is 27x worse in-range.  Both
  constrained fits land at nearly the same numbers regardless of alpha's coordinate, so
  this is a property of the beta family, not of alpha's shape — and per the section above
  the constraint was never imposing physics in the first place.  Scripts:
  `test_constrained_P.py`, `test_constrained_P_flux.py` (scratchpad, not in the project).

## Gotchas

- **`rcdefaults()` must come AFTER the model imports** in any plotting script.
  `gw_remnant/gw_utils/gw_plotter.py` sets `font.size=18`, `font.family='STIXGeneral'`,
  `axes.linewidth=1`, `figure.figsize=(14,10)` at import time.  Reset-then-import silently
  produces 18pt STIX figures instead of 10pt DejaVu Sans.
- **`gwr_energy_anchored` and `gwr_energy_fluxanchored` are NOT PN-free** — they impose
  `X1^(6/5)`.  Keep the 9-coefficient `gwr_energy_stiff` frozen as the *evidence* run: its
  value is that it discovered `X1^(6/5)` (to 0.03%) rather than importing it.  Report both.
- **q < 2.5 has a doubly-extrapolated input.**  `BHPTNRSur1dq1e4` sets
  `X_min = log10(2.5)` and only warns outside it, so q=2.25 and q=2.0 extrapolate the
  surrogate's own splines on top of the coefficient extrapolation.  q=2.5 is the honest gate.
- **The per-q floor is non-monotonic in q**, peaking near q~3.8 and falling toward both
  q=8 and q=2, in *both* model families.  Cause unknown; N_cycles, ringdown fraction and
  x-coverage are all monotonic and none explains it.  It coincides with `P` crossing zero
  (q~3.9) and `A`'s minimum (q~3.98).

## Writeups

`scaling_gwr_energy_fluxanchored.md` (best model + why fewer coefficients extrapolate
better), `scaling_gwr_energy_anchored.md`, `scaling_gwr_energy_stiff.md` (includes the
corrected post-adiabatic account, the 2PA/1PA test for which orders are meaningful, and the
three negative results), `scaling_hm_backbone.md` (higher modes + the four-base comparison).
CLAUDE.md carries one row per model.
