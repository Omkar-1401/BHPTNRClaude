# inspiral_fluxanchored — 7-coefficient inspiral-only model, flux-driven alpha

The flux-drive counterpart of `inspiral_anchored`, fitted on the same truncated window
(`t_nr < -200 M`). Same conclusion on beta — **it rises at every q from 2 to ~13.2** — and
the two models agree on `P` and `b_E` to ~0.3% at every q, which is the point: beta's
behaviour does not depend on what alpha's drive coordinate is doing.

**Read `scaling_inspiral_anchored.md` first.** It is the primary writeup; the model
definition, the beta result, the q ≈ 13.3 boundary, the "how to read the numbers" warning
and caveats 1–4 are shared and are not repeated here. This file covers what is specific to
the flux drive — which is mostly a warning.

Files: `fit_scaling_inspiral_regress.py --model fluxanchored`,
`fit_scaling_inspiral_fluxanchored.py` (per-q layer, and the shared machinery for both
models), `inspiral_fluxanchored_results/coeffs.json` + `coeffs_global.json`,
`Agentic_plots/inspiral_fluxanchored/inspiral_FA_alpha_beta_overlay_{seeded,global}.pdf`.

## The one difference

```python
alpha(t) = alpha_PP(q) * (1 + alpha_C(q) * F(t))     F = Edot/max(Edot)   <- HERE
alpha(t) = alpha_PP(q) * (1 + alpha_C(q) * E(t))     E cumulative         <- anchored
beta (t) = beta_PP(q)  * (1 + b_E(q) * E(t))         E in both

alpha_C(q) = A0*nu + A1*nu**2      # 2 coefficients, PP-anchored by construction
                                   # (anchored uses A0/nu, 1 coefficient)
```

7 coefficients against the sibling's 6. Forms taken unchanged from the parent
`gwr_energy_fluxanchored`.

```python
# seeded                                    # global
c0, c1 = -0.006921,  0.239021               c0, c1 =  0.003598,  0.165091
A0, A1 = -9.368351, 80.622507               A0, A1 = -8.571051, 76.876481
b      = -0.007974                          b      = -0.007495
P0, P1 = -1.373321, 20.925776               P0, P1 = -1.305784, 20.052592
```

## Results

Per-q floor on the same window: 9.1787e-06.

| route | coef | median | max | q=2 | master/floor |
|:---|---:|---:|---:|---:|---:|
| seeded | 7 | 1.0717e-05 | 1.8757e-05 | **4.0898e-04** | 1.17× |
| global | 7 | 1.0894e-05 | **1.5000e-05** | 4.4893e-04 | 1.19× |

| q | insp (seeded) | insp (global) | P | b_E | beta rises |
|---:|---:|---:|---:|---:|:---|
| 3 | 1.8757e-05 | 1.2684e-05 | +2.4541 | +3.4708 | YES |
| 4 | 7.4966e-06 | 7.9236e-06 | +1.9026 | +2.4898 | YES |
| 5 | 8.7483e-06 | 9.1488e-06 | +1.4793 | +1.8430 | YES |
| 6 | 1.0430e-05 | 1.0783e-05 | +1.1496 | +1.3845 | YES |
| 7 | 1.1003e-05 | 1.1006e-05 | +0.8875 | +1.0426 | YES |
| 8 | 1.3926e-05 | 1.5000e-05 | +0.6747 | +0.7777 | YES |
| 2.75 | 3.2069e-05 | 2.3839e-05 | +2.6156 | +3.8006 | YES [extrap] |
| 2.5 | 6.3818e-05 | 5.6543e-05 | +2.7866 | +4.1792 | YES [extrap] |
| 2.25 | 1.5174e-04 | 1.5526e-04 | +2.9658 | +4.6182 | YES [extrap] |
| 2 | 4.0898e-04 | 4.4893e-04 | +3.1503 | +5.1332 | YES [extrap] |

Unlike the sibling, the joint fit is a **wash in the median here** (1.17× → 1.19×) while
still improving the max (1.88e-05 → 1.50e-05). Same inversion the parents show: the flux
forms already sit near the per-q floor, so there is little for a joint fit to recover.
Powell took 80 s, seed mean 1.0636e-05 → 1.0027e-05.

`P` crosses zero at nu = 0.0656 (seeded) / 0.0651 (global), i.e. **q ≈ 13.16 / 13.28** —
within 1.4% of the sibling's boundary. See `scaling_inspiral_anchored.md` for what that
does and does not mean.

## THE WARNING: alpha's coupling is not measured here

**`alpha_C` sits in a near-flat direction below the cut and must not be read physically.**
F spans only **0.0072 / 0.0135 / 0.0235** at q = 3/5/8 for `t < -200 M`, against 0.64–0.85
at merger, so the per-q fit is effectively 3-parameter and `alpha_C`'s recovered sign flips
relative to the full-window fit (+1.73 vs -0.32 at q=3). Regressing that onto
`A0*nu + A1*nu^2` is largely fitting noise, and the two coefficients come out large and
opposed (-9.37, +80.62) — a classic sign of an unconstrained direction.

The sibling has the same disease from a different cause (its per-q `alpha_C` changes sign
inside the training range, which `A0/nu` cannot represent), so **neither fitted model
measures alpha's coupling.** The difference is that here the drive itself is inert, whereas
E retains 21–40% of its range below the cut. Details and the per-q comparison table are in
`scaling_inspiral_anchored.md`, "What does NOT work".

Visible consequence: this model's **full-window** score is 2–3× worse than the sibling's
(seeded coefficients: 2.98e-02 vs 1.30e-02 at q=3, 6.98e-02 vs 2.01e-02 at q=2), because past the cut the
unconstrained flux coupling drives alpha somewhere arbitrary. **This is now visible in the
overlay**: inside the shaded (unfitted) region the q=2 alpha climbs from ~0.62 in the
inspiral to **1.8 at merger**, a 3× excursion produced entirely by `alpha_C * F(t)` with
nothing constraining it. Put the two models' alpha panels side by side — the sibling's is
flat across the whole range, this one is not — and the difference is the whole warning.

**So use this model as the cross-check on beta, not as an alpha model.** For anything about
alpha, the E drive is the one with a constrained coupling — and even there, see the
sibling's warning.

## The per-q layer beneath it

64 per-q optima, 4 free parameters each. From `inspiral_fluxanchored_results/` and
`insp_grid.3431452.log`:

- Inspiral wants a rising beta at **64/64 q**; sign disagreement with the full-window fit at
  **53/64**, region q ∈ [3.80, 8.00].
- Full-window `b_E` crosses zero at **q = 3.799** (sibling 3.928, independent multi-model
  value 3.92).
- Opposite-sign cost: median ×19.18, min ×1.34, max ×97.50.
- Per-q mismatch medians: inspiral 9.1787e-06 vs full-window 4.4415e-04.
- Validation against the 3-point `beta_drift_tests/inspiral_only_sign.py` reference passes;
  largest deviation 0.056 in full-window `b_E` at q=5 (expected — ours is refit).
- The truncated window keeps **94.1%** of the NR samples. Cutting the other 5.9% moves `b_E`
  from -1.48 to +0.36 at q=8. That asymmetry is the finding.

## Two housekeeping facts specific to this folder

1. **The 64-q cache predates the diagnostic fields.** It was written by job 3431452; the
   later refactor added `b_PP_over_X1_6_5`, `beta_rise_to_cut`, `E_frac_at_cut` and
   `drive_span_below_cut`, which exist only for the 3 rows re-forced by job 3431501.
   `report()` tolerates the gap via `.get`, and q=3/5/8 happen to be the rows it indexes
   directly, so it works — by luck. A `--force` over all 64 q would fill them in. The
   sibling's cache is complete.
2. **`summary.json` was stale and has been repaired.** The `--check --force` run at 18:08
   overwrote the 64-q summary with a 3-q one; job 3432081 re-reported over the cached grid
   **without refitting**, so it now reads `n_q = 64` and every fitted value is unchanged.
   `per_q_cache.json.presummary.bak` is the backup taken first.

## Mixed-gauge diagnostics (applies to both models)

`fit_one_q` computes three reported quantities by interpolating `t_cut` onto
`case["t_bhpt"]`, but `t_cut` is an **NR-side** threshold (`truncate()` applies it to
`t_nr`) and the axes differ by the time map. The NR cut at -200 M is `t_bhpt` = -266.3 /
-238.4 / -227.0 at q = 3/5/8. **The fits are unaffected.** `check_tcut_gauge.py` prints both:

| q | E frac as-is | E frac fixed | d beta as-is | d beta fixed | F span as-is | F span fixed |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.2310 | 0.2150 | +2.902% | +2.702% | 0.0097 | **0.0072** |
| 5 | 0.3216 | 0.3098 | +1.143% | +1.101% | 0.0159 | **0.0135** |
| 8 | 0.4136 | 0.4039 | +0.121% | +0.119% | 0.0262 | **0.0235** |

~7% on the energy fractions, up to **26% on the flux span** — the number the inert-alpha
argument rests on, and the correction makes F *more* inert, not less. `summary.json` still
carries the as-is values; the corrected spans are the ones quoted above.

## Not done

- `--force` over all 64 q to fill the missing diagnostic fields.
- A single-coefficient `alpha_C` variant, or `alpha_C = 0`, given that the two coefficients
  are not constrained.
- `t_cut = -500 M` — dropped by decision (2026-08-23), not by finding.
