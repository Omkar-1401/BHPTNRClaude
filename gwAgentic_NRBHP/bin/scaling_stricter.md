# Stricter q=5 (2,2) piecewise-linear scaling fit

Using a two-segment linear fit for `alpha(s)` and `beta(s)` in raw BHPT source time `s`, I get:

```python
t_start = -5000.0377442127
t_end = 100
t_cut = -172.554851971848  # optimized cutoff, constrained to be < -100M
s_min = -6211.57614051161
s_cut = -213.396540026457
s_max = 114.800000108869
```

For `s <= s_cut`:

```python
x_left = (s - s_min) / (s_cut - s_min)
alpha_left(s) = 0.80542112672677 + (0.0103701734041557) * x_left
tau_left(s) = t_start + 0.804824665778662 * (s - s_min) + (1.06942324622101e-07) * (s - s_min) * (s - s_cut)
beta_left(s) = 0.804824665778662 + (1.06942324622101e-07) * (2*s - s_min - s_cut)
```

For `s > s_cut`:

```python
x_right = (s - s_cut) / (s_max - s_cut)
alpha_right(s) = 0.858522261052373 + (-0.171022361144058) * x_right
tau_right(s) = t_cut + 0.830462295121895 * (s - s_cut) + (0.000135410789512007) * (s - s_cut) * (s - s_max)
beta_right(s) = 0.830462295121895 + (0.000135410789512007) * (2*s - s_cut - s_max)
```

Phase rotation:

```python
phi0 = 1.43742544198764  # radians, principal value
phi0_unwrapped = 510.375435323534
```

Applied as:

```python
h_model(tau(s)) = alpha(s) * exp(1j * phi0) * h_BHPT(s)
```

The fit uses raw `BHPTNRSur1dq1e4` with `calibrated=False`, `NRHybSur3dq8` with `dt=0.1`, `f_low=5e-3`, and the `mathcalE_error` definition in `NRBHP_ansatz.py` with NR as the reference waveform.

Diagnostics:

- q: `5`
- mode: `(2, 2)`
- optimized cutoff in NR time: `-172.554851971848`
- corresponding BHPT source split: `-213.396540026457`
- common NR support: `[-5000.0377442127, 99.9622557873008]`
- beta endpoint values `(left_min, left_cut, right_cut, right_max)`: `(0.804183206508686, 0.805466125048639, 0.786020942507061, 0.874903647736728)`
- optimizer downsampled `mathcalE`: `0.000270031466600323`
- full-window `mathcalE`: `0.000269988438387273`
- `t <= t_cut` local `mathcalE`: `8.54560576524473e-05`
- `t > t_cut` local `mathcalE`: `0.0013492641787664`

No extrapolated BHPT samples are used. The source interval is inside the raw BHPT support and endpoint BHPT values are obtained by interpolation within that support.
