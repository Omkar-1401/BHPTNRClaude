# inspiral_anchored — 6-coefficient inspiral-only model, E-driven alpha

**Result: beta RISES at every q from 2 to ~13.3, with no gate, no switch, no second drive
term and no post-merger branch.** That is RESUME_gwremnant.md open item 6, reached by
changing the OBJECTIVE rather than the parameterisation — five earlier attempts to fix
beta's falling drift by changing the form all failed, because the cause was never the form.

Fitted on the inspiral only: the NR comparison window is cut at `t_nr < -200 M`, so merger
and ringdown are not scored. **This is therefore not a full-waveform model** and its
mismatches are not comparable to anything in `CLAUDE.md`'s table — see "How to read the
numbers" below.

Files: `fit_scaling_inspiral_regress.py --model anchored` (master layer),
`fit_scaling_inspiral_anchored.py` (per-q layer beneath it),
`inspiral_anchored_results/coeffs.json` + `coeffs_global.json`,
`Agentic_plots/inspiral_anchored/inspiral_AN_alpha_beta_overlay_{seeded,global}.pdf`.
Sibling and control: `scaling_inspiral_fluxanchored.md`.

## Model

```python
E(t) = gw_remnant Eoft            # radiated energy, units of M, E(t_start) = 0
nu   = q/(1+q)**2                 X1 = q/(1+q)

alpha(t) = alpha_PP(q) * (1 + alpha_C(q) * E(t))
beta (t) = beta_PP(q)  * (1 + b_E(q)     * E(t))  ,  b_E = P / beta_PP
tau(t)   = t0_nr + integral_{T_ANCHOR}^{t} beta dt'        # phi0 analytic

# --- 6 free coefficients, forms taken UNCHANGED from gwr_energy_anchored -------
alpha_PP(q) = X1**1.2 * (1 + c0*nu + c1*nu**2)    # 2, derived base
alpha_C(q)  = A0 / nu                             # 1, normalisation convention
beta_PP(q)  = X1**1.2 * (1 + b*nu)                # 1, derived base
P(nu)       = P0 + P1*nu                          # 2, empirical

# seeded (regress the per-q optima)        # global (joint refit)
c0, c1 =  0.065818, -0.269234              c0, c1 = -0.024501,  0.314372
A0     =  0.004080                         A0     =  0.072862
b      = -0.008001                         b      = -0.007542
P0, P1 = -1.347431, 20.766988              P0, P1 = -1.303904, 20.072327
```

Keeping the parent's forms fixed is deliberate: it isolates the objective, which is the
entire subject of this folder.

## Results

`insp` = the scored window, the objective this model was fitted to. Per-q floor = median of
the 64 per-q optima on the same window, 9.0121e-06.

| route | coef | median | max | q=2 | master/floor |
|:---|---:|---:|---:|---:|---:|
| seeded | 6 | 1.2647e-05 | 2.3988e-05 | **3.7021e-04** | 1.40× |
| global | 6 | **1.1341e-05** | **1.5732e-05** | 4.6622e-04 | 1.26× |

Per q, seeded / global:

| q | insp (seeded) | insp (global) | P | b_E | beta rises |
|---:|---:|---:|---:|---:|:---|
| 3 | 2.3988e-05 | 1.3464e-05 | +2.4597 | +3.4787 | YES |
| 4 | 1.0060e-05 | 7.9826e-06 | +1.9077 | +2.4964 | YES |
| 5 | 8.7681e-06 | 9.3731e-06 | +1.4839 | +1.8488 | YES |
| 6 | 1.1218e-05 | 1.1081e-05 | +1.1539 | +1.3897 | YES |
| 7 | 1.4076e-05 | 1.1601e-05 | +0.8915 | +1.0473 | YES |
| 8 | 1.9046e-05 | 1.5732e-05 | +0.6785 | +0.7821 | YES |
| 2.75 | 3.5149e-05 | 2.5812e-05 | +2.6214 | +3.8089 | YES [extrap] |
| 2.5 | 6.1197e-05 | 6.0911e-05 | +2.7925 | +4.1881 | YES [extrap] |
| 2.25 | 1.3700e-04 | 1.6416e-04 | +2.9719 | +4.6277 | YES [extrap] |
| 2 | 3.7021e-04 | 4.6622e-04 | +3.1566 | +5.1435 | YES [extrap] |

(P and b_E columns are the global ones; the seeded values differ by 2–4%.)

The joint fit helps here — floor ratio 1.40× → 1.26×, max 2.40e-05 → 1.57e-05 — while
costing q=2, which is the same trade the parents show and for the same reason
(`scaling_gwr_energy_stiff.md`: the joint fit only recovers ground when the forms are too
stiff to reach the per-q floor). Powell converged in 44 s; the seed was mean 1.3497e-05 /
max 2.4992e-05 and finished at 1.0373e-05 / 1.5486e-05 on the 12 even-nu training points.

## Where beta stops rising: q ≈ 13.3

`b_E > 0` iff `P > 0`, and `P = P0 + P1*nu` is a line, so the claim has a hard boundary:

| coefficients | P = 0 at nu | q |
|:---|---:|---:|
| anchored, seeded | 0.06488 | **13.34** |
| anchored, global | 0.06496 | **13.32** |
| fluxanchored, seeded | 0.06563 | 13.16 |
| fluxanchored, global | 0.06512 | 13.28 |

So beta rises over **q ∈ [2, 13.3]** — far beyond the [3,8] training range — and the model
predicts a FALLING beta above that. All four independent coefficient sets put the boundary
within 1.4% of each other, which is reassuring but is not evidence about the physics at
q > 8: it is one line extrapolated, and nothing in the data constrains it there. Do not
quote q ≈ 13.3 as a physical scale.

## What does NOT work: the alpha coupling does not regress

`alpha_C`'s per-q values change sign inside the training range, and neither parent's
nu-form can represent that:

| q | per-q `alpha_C` | fitted `A0/nu` (seeded) | fitted (global) |
|---:|---:|---:|---:|
| 3 | **+1.8858** | +0.0218 | +0.3886 |
| 5 | **-0.1871** | +0.0294 | +0.5246 |
| 8 | **-0.1837** | +0.0413 | +0.7377 |

`A0/nu` is monotone and positive; the data crosses zero near q ≈ 4.5. The regression is
therefore not describing alpha's coupling at all — it is setting it to something small and
absorbing the difference into `alpha_PP`. **The fitted alpha is effectively constant in
time.** This costs remarkably little (floor ratio 1.26–1.40×) precisely because alpha barely
moves during the inspiral: at q=3 the per-q coupling produces ~1.6% variation across the
scored window, the fitted one ~0.02%.

Consequences: (a) **the 6th coefficient is not earning its place** — a 5-coefficient version
with `alpha_C = 0` should be tried and is likely to score the same; (b) nothing here measures
alpha's energy coupling in the inspiral, and this md must not be cited as if it did; (c) the
beta result is untouched by any of it, since beta's coefficients are fitted against a drive
that does 21–40% of its range inside the scored window.

The overlay shows this directly: **this model's alpha panel is flat across the entire range,
extrapolation included**, while the flux sibling's q=2 alpha spikes to 1.8 at merger. Flat is
not a virtue here — it is the regression having given up on the coupling.

`beta_PP` by contrast lands essentially on the derived anchor: `b = -0.0080` means
`beta_PP = X1^(6/5) * (1 - 0.008*nu)`, i.e. within 0.2% of `X1^(6/5)` across the range —
consistent with the per-q measurement of 0.108% mean deviation over all 64 q.

## How to read the numbers

The scored-window mismatches (~1e-05) look 40× better than the full-window models in
`CLAUDE.md`'s table (~5e-04). **They are not comparable.** This model is scored on 94.1% of
the NR samples but the easy 94.1% — the post-merger region it never sees carries ~2% of the
radiated energy and 32–38% of the full-window mismatch numerator. Evaluated on the whole
window the same parameters give 3.6e-03 to 2.0e-02, which is the honest statement of what
this model does NOT do.

## The per-q layer beneath it

The master coefficients are regressed from 64 per-q optima (4 free parameters each), which
are themselves the diagnostic that motivated the model. Headline numbers, from
`inspiral_anchored_results/per_q_cache.json` and `insp_anch.3431501.log`:

- Inspiral wants a rising beta at **64/64 q**; full-window vs inspiral-only sign
  disagreement at 52/64, region q ∈ [4.00, 8.00].
- Full-window `b_E` crosses zero at **q = 3.928**, against the independent multi-model
  value 3.92.
- Opposite-sign cost: median ×19.89, min ×1.37 (q=8), max ×118.93 (q=3).
- Per-q mismatch medians: inspiral 9.0121e-06 vs full-window 5.8739e-04.
- 0 seed warnings; `b_PP/X1^(6/5)` within 0.108% mean / 0.135% max over all 64 q.

**The alpha-drive control.** This model exists as the control on its flux sibling, whose
alpha coordinate is nearly inert below the cut. Swapping the drive from F to E changes
`alpha_C` completely (+1.89→-0.18 instead of +1.73→+0.09) and moves the inspiral `b_E` by
**<1% over q ∈ [3,7]**, 5.5% at q=8. At the fitted level the two models agree on `P` and
`b_E` to ~0.3% at every q. Beta's preference is independent of what alpha is doing.

## Caveats

1. **The opposite-sign cost is PROFILED**, not conditional: it re-optimises `b_PP`, and
   `beta_drift_tests/be_bpp_degeneracy.py` showed a ±0.5% move in `b_PP` swings the preferred
   `b_E` from -1.0 to +0.5…+2.0. This is the construction RESUME_gwremnant.md (2026-08-20)
   used to retract the tug-of-war story. Mitigating: the recovered `b_PP` stays within 0.135%
   of `X1^(6/5)` at all 64 q, a quarter of the band the degeneracy needs. That bounds the
   objection; it does not close it. A conditional (fixed-`b_PP`) scan would.
2. **`T_ANCHOR = -100 M` lies OUTSIDE the scored window.** The time map is still well defined
   (integrated over the whole BHPT array) but `t0_nr` is fixed by extrapolating it ~100 M past
   the scored region. It is a gauge constant so it does not bias `b_E`; it does mean `t0_nr`
   here is not comparable to the parent's.
3. **Three per-q diagnostics are mixed-gauge** — `E_frac_at_cut`, `beta_rise_to_cut` and
   `drive_span_below_cut` interpolate the NR-side `t_cut` onto the BHPT time axis. Quantified
   in `check_tcut_gauge.py`: ~7% on the energy fractions, up to 26% on the drive spans. The
   fits are unaffected. See the sibling md for the table.
4. **q < 3 is doubly extrapolated** — the coefficients are extrapolated below their training
   range, and the BHPT surrogate itself is out of domain below q=2.5
   (`BHPTNRSur1dq1e4.py:63` sets `X_min = log10(2.5)`). Shared with every model here.

## Not done

- **The 5-coefficient test** (`alpha_C = 0`), which the alpha-regression failure above
  makes the obvious next run.
- **A conditional `b_E` scan at fixed `b_PP`** on the 64-q grid, to close caveat 1 outright.
- **`t_cut = -500 M`** — dropped by decision (2026-08-23), not by finding.
- Nothing here has been folded into `CLAUDE.md`'s model table, and it should not be until
  someone decides how to present a model that is scored on a different window.
