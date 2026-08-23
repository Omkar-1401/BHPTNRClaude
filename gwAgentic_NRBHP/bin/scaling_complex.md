Using a fresh rerun of the complex-alpha, real-beta fit for `q=5`, `(2,2)`, I get:

```python
x = 2 * (t - t_min) / (t_max - t_min) - 1
t_min = -6188.39999991155
t_max = 114.800000111376
```

Complex amplitude scaling:

```python
alpha(t) = (0.480784114296956 + 0.64655982091471j) + (-0.064666656149051 + 0.0358265083065535j) * x
```

Real time-scaling ansatz:

```python
beta0 = 0.802799878766632
beta1 = -0.000510785785122325
beta(t) = beta0 + beta1 * x
```

The real time map is:

```python
tau0 = -4965.51400425426
u = t - t_min
tau(t) = tau0 + beta0 * u + beta1 * (u**2 / (t_max - t_min) - u)
```

Applied as:

```python
h_model(tau(t)) = alpha(t) * h_BHPT(t)
```

With common-support interpolation, this gave:

```text
mathcalE = 0.000929658906555302
```

on NR support approximately `[-4965.33774421, 94.6622557873]`.

The requested `mathcalE < 1e-05` target was not reached. This is the best result found in this rerun with complex linear `alpha(t)` and real linear `beta(t)`, using generic starts and a least-squares solve for the complex alpha coefficients at each real time map.
