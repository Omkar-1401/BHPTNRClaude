# gw_remnant energy model, physically anchored (gwr_energy_anchored)

**6 coefficients**, globally jointly fitted on q in [3, 8] only.  A reduction of
`gwr_energy_stiff` from 9 coefficients that puts the derived Newtonian chirp factor
X1^(6/5) into both prefactors instead of fitting around it.

## Model

```python
E(t) = gw_remnant Eoft          # radiated energy, units of M, E(t_start) = 0
nu   = q/(1+q)**2               X1 = q/(1+q)

alpha(t) = alpha_PP(q) * (1 + alpha_E(q) * E(t))
beta (t) = beta_PP(q)  +  P(nu) * E(t)
tau(t)   = t0_nr + integral_{T_ANCHOR}^{t} beta dt'
h_model(tau) = alpha * exp(i*phi0) * h_BHPT       # phi0 analytic

# --- 6 free coefficients -----------------------------------------------------
alpha_PP(q) = X1**1.2 * (1 + c0*nu + c1*nu**2)   # 2 free, on a DERIVED base
alpha_E(q)  = A0 / nu                            # 1 free
beta_PP(q)  = X1**1.2 * (1 + b*nu)               # 1 free, on a DERIVED base
P(nu)       = P0 + P1*nu                         # 2 free

c0, c1 =  0.286438, -0.402222
A0     = -1.072443
b      =  0.012631
P0, P1 = -4.350290, 26.934566
```

The per-q model is untouched (`--form mult`), so `per_q_cache_mult.json` is reused and
the per-q optima are identical to `gwr_energy_stiff`; only the regression layer changes.
Global joint refit reuses `fit_scaling_gwr_energy_global`'s fast evaluator exactly as
the stiff model does: Powell, analytic phi0, t0_nr the only per-q nuisance, 12 training
waveforms sampled even in nu over [3, 8].  **No q < 3 data anywhere.**

## Why each order

The full argument, including three negative results, is in
`scaling_gwr_energy_stiff.md` sections "Which coefficient orders are physically
motivated" and "Three negative results".  In brief:

| function | coeffs | status |
|:---|---:|:---|
| `alpha_PP` | 2 | `X1^(6/5)` base is **derived** (Newtonian chirp rate + mass-unit conversion, exact at leading order).  Its residual expansion is well-ordered (2PA/1PA ~ 0.15), so c0 = 1PA and c1 = 2PA are **meaningful orders**. |
| `beta_PP` | 1 | Same derived base.  Residual is 0.19% in total, so 1PA suffices and the order is meaningful. |
| `alpha_E` | 1 | The `1/nu` is a **normalisation convention**, not physics: E is the ppBHPT's radiated energy in the surrogate's own units, so only `alpha_E*E` is physical.  Stopping at one coefficient is principled but negatively so -- higher orders are neither identifiable (alpha_E is the soft direction of a -0.974-correlated degeneracy with alpha_PP) nor meaningful (the 2PA term of `d ln alpha` is 1.8-3.4x the 1PA term across [3,8]). |
| `P` | 2 | **No derivable order.**  Its Bondi-mass account holds at q=3 (1.1%) and inverts by q=8.  Open question. |

So **3 of the 6 coefficients stand on derivable physics** (the two X1^(6/5) bases carry
zero fitted coefficients and pin the leading behaviour; c0 and b are meaningful 1PA
orders; c1 a meaningful 2PA order).  That is up from 3 of 9.

## Results

vs the shipped 9-coefficient `gwr_energy_stiff`.  `seeded` = per-q fit then independent
regression of the same forms; `anchored` = all 6 optimised jointly.

| q | seeded | **anchored (6 coef)** | stiff (9 coef) | |
|---:|---:|---:|---:|:---|
| 3 | 1.0452e-03 | 9.7884e-04 | 9.7359e-04 | |
| 4 | 9.3697e-04 | 9.3106e-04 | 9.3518e-04 | |
| 5 | 7.0391e-04 | 7.0149e-04 | 7.0098e-04 | |
| 6 | 5.5275e-04 | 5.4954e-04 | 5.4634e-04 | |
| 7 | 4.1703e-04 | 4.3925e-04 | 4.2661e-04 | |
| 8 | 4.7544e-04 | 3.1915e-04 | 3.5418e-04 | |
| **in-range median** | | **6.2551e-04** | 6.2366e-04 | +0.3% |
| **in-range max** | | **9.7884e-04** | 9.7359e-04 | +0.5% |

| q < 3 (held out) | seeded | **anchored (6 coef)** | stiff (9 coef) | gain |
|---:|---:|---:|---:|---:|
| 2.75 | 1.0111e-03 | **9.4774e-04** | 9.8291e-04 | 1.04x |
| 2.5 | 9.4635e-04 | **9.2793e-04** | 1.1240e-03 | 1.21x |
| 2.25 | 8.8914e-04 | **1.0163e-03** | 1.7272e-03 | 1.70x |
| **2** | 1.0223e-03 | **1.5293e-03** | 3.7850e-03 | **2.47x** |

**Better at all four held-out mass ratios, with in-range within 0.5%, on 3 fewer
coefficients.**  q=2 also now beats `pn_anchored` (2.55e-3) and `PN_opt_remnant_partial`
(3.7e-3), and sits within 1.8x of the per-q floor (8.4786e-04).

Note the `seeded` column: per-q -> regress is *better* than the joint fit at q <= 2.25
(1.02e-3 vs 1.53e-3 at q=2) but worse in-range (max 1.05e-03 vs 9.79e-04).  The joint
fit was shipped because it clears the low-q gate at all four points and holds in-range;
the seeded variant fails at q=2.75.  Worth revisiting if q=2 becomes the sole priority.

## Why fewer coefficients extrapolate better

14 -> 9 -> 6 coefficients improves q=2 by ~18x (2.8e-02 -> 3.8e-03 -> 1.5e-03) while the
in-range median never moves (6.26 / 6.24 / 6.26e-04): all three already sit at the per-q
floor, so surplus coefficients buy nothing in-range and are pure extrapolation liability.
Two effects, which multiply.

**1. The orders are degenerate in-range.**  nu spans only [0.0988, 0.1875], so 1, nu,
nu^2, nu^3 are nearly collinear.  `lever` = how far a wiggle that is *invisible in-range*
can move q=2:

| degree | cond(V) | lever |
|---:|---:|---:|
| 1 | 4.0e+01 | 0.45 |
| 2 | 1.8e+03 | 1.40 |
| 3 | 8.3e+04 | **4.35** |

Each dropped order cuts that freedom ~3x.  Worse, the higher orders grow fastest where
the extrapolation goes: q=3 -> q=2, nu^1 grows 1.19x, nu^2 1.40x, nu^3 1.66x.

**2. Anchoring shrinks what the polynomial must carry.**  Raw swing over [3,8] vs swing
after dividing by X1^(6/5): `alpha_PP` 20.80% -> 1.51% (14x), `beta_PP` 22.50% -> 0.14%
(166x).  So extrapolation error scales down with coefficient magnitude -- in `alpha_PP` at
q=3 the stiff terms are -0.214 / +0.055 / -0.104 (the nu^3 term is half the nu^1 term),
against the anchored +0.054 / -0.014.

Effect 1 is generic statistics, not physics; the physics is only in *which* base.  The
166x is real rather than relocated because X1^(6/5) is right to 0.14% unfitted -- a wrong
base would give the same variance reduction and buy a bias.

## Caveat: this is no longer a PN-free model

`gwr_energy_stiff`'s headline claim is that it reaches `pn_anchored`'s accuracy with **no
PN input**, and its value as *evidence* for the X1^(6/5) anchor comes precisely from
having discovered it rather than imposed it.  This model imposes it, so that evidentiary
claim does not transfer.  **Keep the shipped 9-coefficient `gwr_energy_stiff` frozen as
the evidence run**; this is the model.  Both should be reported.

## Higher modes

The backbone transfer (`fit_scaling_hm_backbone.py --model anchored`) inherits the
improvement exactly where the earlier `--perq` decomposition predicted it would:

| q=2, rho=1 | stiff base | anchored base | |
|:---|---:|---:|:---|
| (2,2) | 3.785e-03 | **1.529e-03** | 2.47x better |
| (3,3) | 1.735e-02 | **1.001e-02** | 1.73x better -- it is base-limited |
| (4,4) | 4.615e-02 | 4.866e-02 | 5% worse |
| (5,5) | 4.121e-02 | 4.651e-02 | worse |
| (2,1) | 3.849e-02 | 3.576e-02 | better |

(3,3) tracks the base improvement, confirming that its q<3 error is inherited rather
than intrinsic.  (4,4) and (5,5) degrade slightly -- the better base removes the
accidental compensation that was masking part of their transfer error, exactly as the
per-q-base diagnostic showed.  At q=4 (in-range) everything is unchanged to <1%.

## Files

- fit: `fit_scaling_gwr_energy_anchored.py --global --maxiter 80`
- coeffs: `gwr_energy_anchored_results/coeffs.json` (carries the reference stiff numbers
  and a `low_q_better_than_stiff9` gate count)
- plots: `python NRBHP_gwr_energy_anchored_plots.py` -> `Agentic_plots/gwr_energy_anchored/`
  (same figure set and styling as `NRBHP_gw_remnant_energy_plots.py`)
- higher-mode figures: `python NRBHP_hm_backbone_fig8_plots.py --q 2 4 --model anchored`
  -> `Agentic_plots/hm_backbone/q{2,4}_waveforms_224_anchored.png`
- `--alphaE-deg 2` gives an 8-coefficient variant (best per-function accuracy for
  `alpha_E`); not run here.
