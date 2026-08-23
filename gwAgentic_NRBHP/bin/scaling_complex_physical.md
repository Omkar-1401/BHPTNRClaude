Using a physically reparameterized complex-alpha, real-beta fit for `q=5`,
`(2,2)`, I get:

The BHPT waveform is the raw surrogate output with `calibrated=False`; the
NR reference is `NRHybSur3dq8`.

```python
q = 5
nu = q / (1 + q)**2
nu = 0.138888888888889

t_bhpt_merger = 1.10958353616297e-07
t_nr_merger = -0.837744212698453
```

The dimensionless physical time variable is merger-centered and scaled by the
symmetric mass ratio:

```python
theta = nu * (t_bhpt - t_bhpt_merger)
```

Complex amplitude scaling:

```python
alpha(theta) = (
    0.417769339738713 + 0.681562307267425j
    + (-0.000147473797901047 + 8.17440857493892e-05j) * theta
)
```

Real time-scaling ansatz:

```python
beta0 = 0.802325568998225
beta1 = -1.12150395523135e-06
beta(theta) = beta0 + beta1 * theta
```

The real time map is written relative to the two merger times:

```python
Delta_tau = 3.42403748806521
dt_bhpt = t_bhpt - t_bhpt_merger
tau(t_bhpt) = (
    t_nr_merger
    + Delta_tau
    + beta0 * dt_bhpt
    + 0.5 * beta1 * nu * dt_bhpt**2
)
```

This satisfies:

```python
d tau / d t_bhpt = beta(theta)
```

Applied as:

```python
h_model(tau(t_bhpt)) = alpha(theta) * h_BHPT(t_bhpt)
```

With common-support interpolation, this gave:

```text
mathcalE = 0.00092820968526494
```

on NR support approximately `[-5000.03774421, 94.6622557873]`.

The previous arbitrary `t_min`/`t_max` normalization is not used in this
final formula. The only time origin information entering the formula is
through the physical merger times `t_bhpt_merger` and `t_nr_merger`. The
fitted `Delta_tau` is a merger-relative alignment offset, not an absolute
simulation start/end time. The time map remains real; only
`alpha(theta)` is complex.
