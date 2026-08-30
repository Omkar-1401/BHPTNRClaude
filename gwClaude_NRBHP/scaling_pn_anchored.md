# pn_anchored — PN-anchored ppBHPT→NR (2,2) scaling model

A global-joint-fit (2,2) model in which the **q-dependence of the leading scaling is imposed
analytically from post-Newtonian physics** rather than fitted, following the note
`alpha_beta_pn_scaling_note_revised.pdf`. Only 9 small, ν-suppressed residual constants are
calibrated. Trained on q ∈ [3, 8]; q ∈ [2, 2.75] is a supported extrapolation.

**One-liner:** the PN anchors convert the q-dependence from something you *fit-and-extrapolate*
into something you *evaluate* — for the inspiral, essentially for free (which is why q<3 stays
smooth, with no q=2.25 notch); the merger–ringdown is the part where that is not yet true.

## Model

`X1 = q/(1+q)`, `ν = q/(1+q)²`, base = `X1^(6/5)` (the Newtonian chirp + quadrupole anchor;
`α₂,LO = β_LO = X1^{6/5}`). Along the ppBHPT (2,2) waveform:
`x = (M Ω_orb)^{2/3} = (ω_gw/2)^{2/3}` from the phase, `xc = min(x, 0.26)`; `p_loss` = the
wf-flux loss coordinate `½(Ê+Ĵ)` (no surfinBH), used only for the switch.

```
S         = sigmoid( (p_loss − (p0_0 + p0_1·ν)) / w0 )
beta_insp = base · [ 1 + ν(b1·xc + b2·xc²) ]
alpha_insp= base · [ 1 + (55/42)·ν·xc + ν·a2·xc² ]         # 55/42 = 1PN slope, FIXED (not fit)
beta_r    = beta_r_physical(q) · r0                        # QNM remnant (NRSur3dq8Remnant)
alpha_mr  = base · (m0 + m1·ν)
beta(t)   = (1−S)·beta_insp + S·beta_r
alpha(t)  = (1−S)·alpha_insp + S·alpha_mr
tau(t)    = ∫ beta dt   (merger → t=0);   h(tau) = alpha · h_BHPT   (φ0 = 0 convention)
```
`beta_r_physical = [0.3683(1+q)/q] / [(F1+F2(1−χ_f)^F3)/M_f]`, `F1,F2,F3 = 1.5251,−1.1568,0.1292`,
`(M_f,χ_f)` from surfinBH `NRSur3dq8Remnant`.

**PP-anchoring:** the inspiral residuals are all ∝ ν, so `α_insp, β_insp → base = X1^{6/5} → 1`
as ν → 0 (q → ∞) — the correct test-particle limit.

## Fit
Global joint fit (Powell) of the 9 residual params against the mean waveform mismatch over the
even-ν 12-q training set in [3,8]. `t0`, `φ0` are per-q alignment nuisances (analytic φ0 + 1-D t0).
No per-q stage, no coefficient regression — the anchors already supply the q-dependence.
Reproduce with `python fit_scaling_pn_anchored.py --refit`.

## Performance (mathcalE vs NRHybSur3dq8, full window, t0/φ0-aligned)

| q | pn_anchored | wf_nu_hybrid_global (even-ν) |
|:--|:--|:--|
| 8.0 | 2.3e-4 | 3.9e-4 |
| 5.0 | 6.8e-4 | 1.2e-4 |
| 3.0 | 6.1e-4 | 5.2e-4 |
| 2.75 (extrap) | 6.8e-4 | 1.0e-3 |
| 2.5 (extrap) | 9.0e-4 | 2.5e-3 |
| **2.25 (extrap)** | **1.4e-3** | 9.4e-3 |
| **2.0 (extrap)** | **2.55e-3** | 4.0e-3 |
| in-range [3,8] median | 6.1e-4 | 1.8e-4 |
| in-range [3,8] max | 7.7e-4 | 5.2e-4 |

**q<3 is monotonic with no q=2.25 notch**, and q=2 beats both the even-ν global (4.0e-3) and
`remnant_partial` (3.7e-3) — because the q-dependence is evaluated, not extrapolated. The trade
is a ~3× worse in-range median (the ν-flat residuals can't match the free per-q flexibility).
An inspiral/merger split at q=2 gives inspiral-only **0.21%** (87% of the signal power) and
merger-ringdown **0.57%** — i.e. the well-anchored inspiral carries the good number.

## Coefficients (`pn_anchored_results/coeffs.json`)

| b1 | b2 | a2 | p0_0 | p0_1 | w0 | r0 | m0 | m1 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.352 | 8.553 | −22.9193 | 3.2336 | 2.2404 | 1.9069 | 0.6869 | 0.7604 | 3.9838 |

## Honest caveats
1. **The merger–ringdown branch is under-engaged.** The fit leaves the switch weak (`S ≤ 0.18`),
   so the QNM `β_r` / `α_mr` remnant anchor barely contributes; the merger is largely shaped by
   the inspiral polynomial's large residuals (`b2=8.55, a2=−22.9`) + the `x=0.26` clip. Symptoms:
   `α_mr(q=2) ≈ 1.0` (unphysical vs truth ringdown α ≈ 0.34); MR β overshoots truth (0.71 vs 0.66);
   a small slope kink where `x` hits the clip. So the **inspiral is genuinely PN-anchored; the MR
   sector is not yet a physical remnant anchor** — it clears the mismatch gate but for the wrong
   structural reason. This is the open item (constrain the switch to engage so the QNM branch
   carries the ringdown).
2. **In-range is ~3× the even-ν global** (6.1e-4 vs 1.8e-4): the price of ν-flat residuals + the
   physics anchor, in exchange for the free, notch-free q<3 extrapolation.
3. Off the (2,2): the note's payoff (higher-mode backbone) requires a physically-correct (2,2) MR,
   which caveat 1 does not yet provide.

## Files & usage
- `BHPTNRPNAnchored.py` — generator:
  ```python
  from BHPTNRPNAnchored import generate_pn_anchored_calibrated
  t, h = generate_pn_anchored_calibrated(q_input=5.0)   # merger at t=0, dt=0.1, no NR needed
  ```
  `get_alpha_beta(q)` returns `(t, α(t), β(t))`.
- `fit_scaling_pn_anchored.py` — global joint fit (`--refit` rewrites coeffs.json; default validates).
- `pn_anchored_results/coeffs.json` — coefficients.
- `peaks_results/prototypes/validate_pn_anchored.py` — q=2/3/5/8 validation (combined PNG).
- `peaks_results/prototypes/perq_plots_pn_anchored.py` — per-q PDF set (wf_nu_hybrid_global style).
- Plots: `Agentic_plots/pn_anchored/` — per-q PDFs `pn_anchored_q{8,5,3}` and `pn_anchored_q2_extrap`
  × `{waveform, zoomed, params, loss_coords}` (mathcalE in each title); plus PNGs
  `pn_anchored_validation`, `pn_anchored_alpha_2358`, `pn_anchored_alpha_fine_2to3`,
  `pn_anchored_q2_vs_perq_truth`.

Note: `alpha_beta_pn_scaling_note_revised.pdf` (the derivation) and memory `[[pn-anchors-scaling-note]]`.

## Verification on TACC (2026-08-29)

The shipped coefficients were re-validated on Stampede3 (fresh conda env, freshly
downloaded NRSur3dq8Remnant data, different machine from the original fit) ahead of the
second-order self-force comparison (`wardell_ideation.md`), job 3448537:

| | documented | reproduced 2026-08-29 |
|:--|--:|--:|
| in-range median | 6.16e-4 | 6.148e-04 |
| in-range max | 7.67e-4 | 7.652e-04 |
| q=2.75 | 6.79e-4 | 6.774e-04 |
| q=2.5 | 9.04e-4 | 9.026e-04 |
| q=2.25 | 1.42e-3 | 1.415e-03 |
| q=2.0 | 2.55e-3 | 2.551e-03 |

3-4 digit agreement everywhere, held-out q<3 included: **the shipped
`theta = [b1=0.352, b2=8.553, a2=-22.9193, ...]` is NOT stale.**

A fresh `--refit` from scratch (independent Powell optimisation, same objective) was
also run to test whether the shipped theta is a well-determined minimum; its output is
`pn_anchored_results/coeffs_refit_20260826.json`, with the shipped `coeffs.json` kept
canonical (`coeffs_shipped_backup.json` is the pre-refit copy).  The parameter-by-
parameter comparison is at the end of `pn_verify.3448537.log`.
**Refit outcome (job 3448537, completed 2026-08-29):** the fresh Powell fit converged
to a DIFFERENT and WORSE basin — in-range median 7.632e-4 (vs shipped 6.148e-4), max
1.047e-3 (vs 7.652e-4), q2 4.522e-3 (vs 2.551e-3) — with large individual parameter
moves (b1 -42%, b2 -32%, a2 -15%, MR/switch params up to -264%).  Two conclusions:
(1) the shipped theta is the better minimum and REMAINS CANONICAL (verified restored);
(2) individual parameters are only loosely pinned by the objective — sloppy directions —
though the combinations relevant downstream move far less (e.g. 2*a2+(4/3)*b2 shifts
only ~9% between basins).  Treat single-parameter values as ~tens-of-percent uncertain.

Definition caveat recorded during this check (relevant to the Wardell comparison):
`get_x` smooths the PHASE with a fixed 401-SAMPLE savgol window (= 80 M on the ppBHPT
dt=0.2 M grid) and uses gradient-of-smoothed, not savgol's analytic derivative — a
DIFFERENT x definition from `x_drive`/`alpha_beta_of_omega_v2` (20 M, analytic).
One-sided (ppBHPT only), so the v1 two-sided asymmetry bug does not apply, but
`a2, b1, b2` are defined with respect to THIS x.
