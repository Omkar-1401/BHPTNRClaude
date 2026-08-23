# Complex-alpha scaling summary

## Objective

Test whether allowing complex linear-in-time amplitude scaling can reduce the q=5, `(2,2)` raw BHPT-to-NRHybSur3dq8 error below `mathcalE < 0.01`, while keeping `beta(t)` and the time map real.

The reference setup was:

- mass ratio: `q = 5`
- mode: `(2, 2)`
- BHPT model: `BHPTNRSur1dq1e4`
- BHPT setting: `calibrated=False`
- NR model: `NRHybSur3dq8`
- NR waveform: nonspinning, `dt=0.1`, `f_low=5e-3`, `t_start=-5000.1`
- error definition: `mathcalE_error` from `NRBHP_ansatz.py`, with NR as the reference waveform

## Steps followed

1. Deleted the earlier complex-`beta` markdown results.
2. Re-read `scaling.md` and used the same `x`, `t_min`, and `t_max` convention.
3. Converted the constant phase rotation in `scaling.md` into a complex linear `alpha(t)` seed.
4. Optimized complex `alpha(t)` and real linear `beta(t)` directly against the complex waveform samples.
5. Computed `mathcalE` only after restricting to common support, with no extrapolated BHPT samples.

## Errors observed

- real `scaling.md` ansatz reproduced in this script: `mathcalE = 0.0223037820262305`
- complex `alpha(t)`, real `beta(t)`: `mathcalE = 0.000929685478304248`

The final comparison support was approximately `[-4965.53774421, 94.6622557873]`.

## Caveats

- This is a complex amplitude correction, not a complex time map.
- The complex phase content in `alpha(t)` can absorb residual phase error, so it is less directly interpretable as a pure physical amplitude scale.
- `beta(t)` remains real and monotonic in the accepted fit, so no complex-time BHPT evaluation or projection is used.

## Current status

The requested `mathcalE < 0.01` target was achieved. The final complex-alpha fit reached `mathcalE = 0.000929685478304248`.
