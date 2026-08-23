# PN-loss q=5 (2,2) scaling fit

This fit follows the radiated-energy/radiated-angular-momentum intuition of
arXiv:2301.07215, but uses the PN flux expressions in `PN_exprs.py` as
time-dependent IMR coordinates rather than only as a single merger-ringdown
rescaling factor.

The BHPT waveform is the raw surrogate output with `calibrated=False`; the
NR reference is `NRHybSur3dq8`.

```python
q = 5
nu = q / (1 + q)**2
nu = 0.138888888888889
t_bhpt_merger = 1.10958353616297e-07
t_nr_merger = -0.837744212698453
t_ref = -100
```

I estimate a PN frequency parameter from the raw BHPT `(2,2)` phase:

```python
omega_gw = abs(d unwrap(arg(h_BHPT_22)) / dt_bhpt)
x_pn = (omega_gw / 2)**(2/3)
F_E = PN_exprs.dEdt(nu, x_pn)
F_J = PN_exprs.dJdt(1.0, x_pn, nu)
DeltaE_PN(t) = integral_from_merger_to_t F_E dt
DeltaJ_PN(t) = integral_from_merger_to_t F_J dt
Ehat = DeltaE_PN / abs(DeltaE_PN(t_ref))
Jhat = DeltaJ_PN / abs(DeltaJ_PN(t_ref))
p_loss = 0.5 * (Ehat + Jhat)
```

The optimized PN-loss split is:

```python
p_cut = -1.04912911600967
Ehat_cut = -1.03377368472509
Jhat_cut = -1.06448454729426
t_bhpt_cut = -123.160864418004
t_cut_nr = -99.715505147549
DeltaE_PN(t_ref) = -0.0216330274770254
DeltaJ_PN(t_ref) = -0.0443145705120832
```

For `p_loss <= p_cut`:

```python
dE = Ehat - Ehat_cut
dJ = Jhat - Jhat_cut
alpha_left = 0.814083118052137 + (-0.00507316934927739) * dE + (0.00512047726983142) * dJ
beta_left = 0.803047080047792 + (-0.0186120035858063) * dE + (0.00465002149479355) * dJ
```

For `p_loss > p_cut`:

```python
dE = Ehat - Ehat_cut
dJ = Jhat - Jhat_cut
alpha_right = 0.831569035085684 + (0.237376525271462) * dE + (-0.346031085814184) * dJ
beta_right = 0.812858430627061 + (-0.0330118271886764) * dE + (0.0552310304927563) * dJ
```

The time map is the numerical integral of `beta` anchored at the PN-loss split:

```python
tau(t_bhpt) = t_cut_nr + integral_from_t_bhpt_cut_to_t_bhpt beta(t') dt'
```

Phase rotation:

```python
phi0 = 1.3729217458051  # radians, principal value
phi0_unwrapped = 1.3729217458051
```

Applied as:

```python
h_model(tau(t_bhpt)) = alpha(Ehat, Jhat) * exp(1j * phi0) * h_BHPT(t_bhpt)
```

Diagnostics:

- q: `5`
- mode: `(2, 2)`
- common NR support: `[-4999.9377442127, 94.1622557873015]`
- common-support coverage of requested NR window: `0.99884315993804`
- optimizer downsampled `mathcalE`: `8.17864436145218e-05`
- full-window `mathcalE`: `8.21468646687417e-05`
- `t <= t_cut_nr` local `mathcalE`: `3.96680857281551e-05`
- `t > t_cut_nr` local `mathcalE`: `0.000430916058920405`
- alpha endpoint values `(left_start, left_cut, right_cut, right_end)`: `(0.805988290186194, 0.814083118052137, 0.831569035085684, 1.77885127451811)`
- beta endpoint values `(left_start, left_cut, right_cut, right_end)`: `(0.803801686301886, 0.803047080047792, 0.812858430627061, 0.763683153677214)`

The requested `mathcalE ~ 1e-4` target is reached, and the fit is
comfortably below `1e-3`. The main caveat is that the 2PN flux expressions are
being used as an IMR coordinate even through merger-ringdown, where they should
be interpreted as a physically motivated proxy rather than a controlled PN
approximation.
