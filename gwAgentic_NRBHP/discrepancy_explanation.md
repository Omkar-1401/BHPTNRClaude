# Discrepancy explanation

The `scaling.md` and `scaling_rerun.md` fits use the same functional family:

```python
alpha(t) = alpha0 + alpha1 * x
beta(t) = beta0 + beta1 * x
```

with `tau(t)` obtained by integrating the real `beta(t)` over raw BHPT source
time. The difference in error is not mainly a different comparison window. I
checked both fits on their shared NR support:

```text
old scaling.md fit on shared support:      mathcalE = 0.0223453983
new scaling_rerun.md fit on shared support: mathcalE = 0.00093931495
```

So the `scaling.md` coefficients were a valid low-order fit, but they were not
the best fit found within that ansatz.

The coefficients changed because this is a nonconvex waveform-alignment problem.
Small changes in `tau0`, `beta(t)`, and `phi0` shift the waveform by fractions
of a cycle, and the optimizer can land in different basins. The original
`scaling.md` result beat the then-requested loose target, but the later rerun
used multiple independent constant-scaling seeds and optimized `alpha0`,
`alpha1`, `beta0`, `beta1`, `tau0`, and `phi0` together.

The two fits are:

```text
old scaling.md:
alpha(t) = 0.77716 + 0.03838*x
beta(t)  = 0.80759 + 0.004335*x
tau0-ish = -4998.58
phi0     = 1.97531
mathcalE ~= 0.0223

new scaling_rerun.md:
alpha(t) = 0.80673 - 0.008930*x
beta(t)  = 0.80363 - 0.000811*x
tau0     = -4970.71
phi0     = 1.03012
mathcalE ~= 0.000939
```

In short: `scaling.md` should be understood as an earlier acceptable low-order
fit, not as the optimized minimum for that ansatz. The rerun found a better
alignment basin in the same real linear model class.
