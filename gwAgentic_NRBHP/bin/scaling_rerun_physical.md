Using a physically reparameterized real low-order time-dependent fit for
`q=5`, `(2,2)`, I get:

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

Amplitude scaling:

```python
alpha(theta) = 0.798154429795784 + (-2.02506994142429e-05) * theta
```

Real time-scaling ansatz:

```python
beta0 = 0.802859390180896
beta1 = -1.81652710845685e-06
beta(theta) = beta0 + beta1 * theta
```

The real time map is written relative to the two merger times:

```python
Delta_tau = 3.39415214622104
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

I also included a constant phase rotation:

```python
phi0 = 1.03070932117571  # radians
```

Applied as:

```python
h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)
```

With common-support interpolation, this gave:

```text
mathcalE = 0.000938183072751145
```

on NR support approximately `[-5000.03774421, 94.6622557873]`.

The previous arbitrary `t_min`/`t_max` normalization is not used in this
final formula. The only time origin information entering the formula is
through the physical merger times `t_bhpt_merger` and `t_nr_merger`. The
fitted `Delta_tau` is a merger-relative alignment offset, not an absolute
simulation start/end time.
