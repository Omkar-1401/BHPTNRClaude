"""
inspiral_anchored -- the `gwr_energy_anchored` per-q model with the NR comparison window
restricted to t < -200 M.

Identical to `inspiral_fluxanchored` except for ALPHA's drive coordinate:

    alpha(t) = a_PP * (1 + a_E * E(t))      <- E, cumulative radiated energy  (here)
    alpha(t) = a_PP * (1 + a_F * F(t))      <- F = Edot/max(Edot)   (fluxanchored)
    beta (t) = b_PP * (1 + b_E * E(t))      <- E in both

WHY THIS IS THE RIGHT SECOND MODEL.  In the flux run, alpha's coordinate turned out to
be nearly INERT below the cut -- F spans only 0.0097 / 0.0159 / 0.0263 at q = 3/5/8 for
t < -200 M against 0.64-0.85 at merger -- so `a_F` sat in a near-flat direction, its
recovered sign flipped relative to the full-window fit, and the fit was effectively
3-parameter.  E(t) retains 23.1/32.2/41.4% of its range below the cut, so here alpha's
coupling is genuinely constrained.

That makes this a control on the flux result rather than a repeat of it.  Two outcomes,
both informative:

  * b_E comes out essentially the same as the flux run -> the "inspiral wants a rising
    beta" conclusion is independent of what alpha is doing, i.e. beta's preference is not
    an artefact of alpha absorbing (or failing to absorb) the amplitude in the inspiral.
  * b_E differs materially -> alpha and beta are coupled in the inspiral, and the flux
    run's b_E was partly compensating for its inert alpha.  That would weaken the
    single-model conclusion and would need reporting.

Note this is a PER-Q diagnostic: neither model has a regression layer here, so
"anchored" refers only to the parent whose per-q form is reproduced (the `mult` form,
alpha and beta both on E), NOT to the parent's 6-coefficient X1^(6/5)-anchored nu
regression, which is deliberately absent.  The X1^(6/5) relation is instead REPORTED as
a diagnostic (`b_PP_over_X1_6_5`) rather than imposed.

Usage:
    python fit_scaling_inspiral_anchored.py --nproc 24
    python fit_scaling_inspiral_anchored.py --q 3 5 8        # spot check
"""
from fit_scaling_inspiral_fluxanchored import main

if __name__ == "__main__":
    main(default_model="anchored")
