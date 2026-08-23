Using the low-order time-dependent fit for `q=5`, `(2,2)`, I get:

```python
x = 2 * (t - t_min) / (t_max - t_min) - 1
t_min = -6188.39999991155
t_max = 114.800000111376
```

Amplitude scaling:

```python
alpha(t) = 0.777159886604501 + 0.0383836831997672 * x
```

Time map:

```python
tau(t) = 6.83151902970822*x**2 + 2545.20004488191*x - 2460.3692183605
```

So the corresponding time-dependent beta is:

```python
beta(t) = d tau / dt
        = 0.807589809897401 + 0.0043352703577125 * x
```

I also included a constant phase rotation:

```python
phi0 = 1.97530886542818  # radians
```

Applied as:

```python
h_model(tau(t)) = alpha(t) * exp(1j * phi0) * h_BHPT(t)
```

With common-support interpolation, this gave:

```text
mathcalE = 0.0223913001494002
```

on NR support approximately `[-4998.76, 86.97]`, so it is comfortably below the updated `< 1` target.
