Using a fresh rerun of the real low-order time-dependent fit for `q=5`, `(2,2)`, I get:

```python
x = 2 * (t - t_min) / (t_max - t_min) - 1
t_min = -6188.39999991155
t_max = 114.800000111376
```

Amplitude scaling:

```python
alpha(t) = 0.806734255287305 + (-0.00893045394034559) * x
```

Real time-scaling ansatz:

```python
beta0 = 0.803628877212988
beta1 = -0.000811064523098718
beta(t) = beta0 + beta1 * x
```

The real time map is the integral of `beta(t) = d tau / dt`:

```python
tau0 = -4970.7102530274
u = t - t_min
tau(t) = tau0 + beta0 * u + beta1 * (u**2 / (t_max - t_min) - u)
```

I also included a constant phase rotation:

```python
phi0 = 1.03012031084274  # radians
```

Applied as:

```python
h_model(tau(t)) = alpha(t) * exp(1j * phi0) * h_BHPT(t)
```

With common-support interpolation, this gave:

```text
mathcalE = 0.000939315602402435
```

on NR support approximately `[-4970.53774421, 94.6622557873]`.

The requested `mathcalE < 1e-05` target was not reached. This is the best result found in this rerun with real linear `alpha(t)` and real linear `beta(t)`, using generic starts rather than previous fitted coefficients.
