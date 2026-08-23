# Justification for the time-dependent calibration approach

## Is the order-of-magnitude improvement real, or overfitting?

Mostly real, for several reasons.

**The improvement is physically motivated, not merely parametric.** The single ingredient
that halved the error relative to the best constant-parameter model was `beta_L` — the
linear drift of the inspiral time-stretch factor along the PN-loss coordinate. This
captures a genuine physical effect: BHPT and NR evolve at slightly different rates
during the inspiral, and that rate difference is not constant. The additional `alpha_E`
and `alpha_J` terms similarly encode real amplitude evolution driven by the energy and
angular momentum radiated since the logistic switch point. None of these are arbitrary
free parameters — each has a direct physical interpretation.

**The model generalises well across q.** If the per-q fits were overfitting, the
polynomial master model would degrade sharply relative to the independent fits.
Instead the median master error (7.7e-5) is only ~35% above the median per-q error
(5.7e-5). That ratio is healthy and consistent with a smooth, well-constrained
q-dependence.

**10 parameters against ~51,000 NR waveform points** is vastly underdetermined, so
classical overfitting in the statistical sense is not the operative concern.

The honest caveat is different: **NRHybSur3dq8 itself carries surrogate modelling
errors**, likely at the ~1e-4 level. Below that floor we may be fitting surrogate
artefacts rather than real BHPT-NR physics. Our final errors (7e-5 to 2e-4) sit right
in that regime. To determine whether the improvement is genuine at this level would
require validation against actual NR waveforms rather than the surrogate — something
that cannot be resolved from the surrogate-to-surrogate comparison alone.

## Time-independent vs time-dependent: which is more ideal?

Time-independent models (constant α and β per q, then a quartic polynomial in 1/q)
are used by other groups and typically achieve mathcalE ~ 1e-3, degrading to ~1e-2 at
low mass ratios. The present time-dependent creative model achieves a median master
error of 7.7e-5 and a maximum of 2.1e-4 (at q = 3) across the full range 3 ≤ q ≤ 8.

| | Time-independent (constant α, β) | Time-dependent (this work) |
|:---|:---:|:---:|
| Total q-dep parameters | ~8 | 40 |
| Typical mathcalE | ~1e-3 (worse at low q) | ~1e-4 |
| Evaluation cost | Trivial (multiply + rotate) | PN-loss coordinate + β integral |
| Physical motivation | Minimal | Strong (beta drift, PN-loss switch) |
| Calibration time per q | Fast | ~minutes |

The evaluation overhead of the time-dependent model is real but moderate. Given that
the BHPT waveform must already be in hand, the additional work is: smooth phase
differentiation to get ω_gw, two cumulative PN-flux integrals to form p_loss, one
sigmoid evaluation, and one cumulative trapezoid for the β time map. None of these
are expensive in absolute terms.

**The time-dependent model is the right direction.** The constant-α/β model leaves
real accuracy on the table for a reason that is now understood — the inspiral beta
drift — not simply due to insufficient free parameters. The 40-coefficient polynomial
expansion is not extravagant: it is 10 smooth functions of a single variable, each
well-constrained by 40 calibration points spaced uniformly in q.

## Runtime cost at waveform evaluation time

A common concern with time-dependent models is evaluation speed. It is worth being
precise about what "time-dependent" actually costs once the calibration is done.

The 40 polynomial coefficients are computed once during calibration and stored. At
waveform generation time, evaluating the model for a new q requires:

1. Evaluate 10 cubic polynomials in 1/q — trivial arithmetic, negligible cost.
2. Compute the PN-loss coordinate from the BHPT waveform — one Savitzky-Golay
   smoothing, one numerical derivative, two cumulative trapezoid integrals.
3. Evaluate the sigmoid and the β time-map integral — one vectorised pass over the
   BHPT time grid.
4. Interpolate the scaled waveform onto the NR time axis — one `np.interp` call.

Steps 2–4 scale with the length of the BHPT waveform, exactly as any waveform
post-processing would. The time-independent model also requires step 4 (or equivalent
rescaling); the real overhead here is steps 2–3, which amount to a few vectorised
NumPy operations over an array already in memory.

This is a small constant-factor overhead, not a scaling difference. There is no
optimisation loop, no iterative solver, and no surrogate evaluation at query time —
just array arithmetic. The label "time-dependent" can suggest something expensive;
the precomputed polynomial means it is not. In practice the evaluation cost of this
model is dominated by the BHPT surrogate call itself, not the calibration layer on
top of it.

## The practical choice depends on the application:

- **High-accuracy surrogate construction or BHPT-NR systematic studies**: the
  time-dependent model is worth the complexity. The order-of-magnitude accuracy
  improvement is physically motivated and generalises cleanly across q.

- **Large-scale parameter estimation (millions of waveform evaluations, MCMC)**:
  if mathcalE ~ 1e-3 is acceptable for the science goal, the simpler constant model
  wins on speed and interpretability. The time-dependent model would be the better
  choice for building the underlying surrogate that a fast emulator is then trained on.
