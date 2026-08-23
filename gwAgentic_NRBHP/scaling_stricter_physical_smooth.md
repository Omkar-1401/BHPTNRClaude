# Reduced stricter physical smooth q=5 (2,2) scaling fit

This run replaces the previous smooth ansatz with a lower-parameter
version motivated by the nearly constant pre-cutoff behavior. Before
`theta_cut`, both `alpha` and `beta` are constants. After a fitted smooth
transition interval, the post-cutoff branches are linear in `theta`.

The BHPT waveform is the raw surrogate output with `calibrated=False`; the
NR reference is `NRHybSur3dq8`.

```python
q = 5
mode = (2, 2)
nu = q / (1 + q)**2
nu = 0.138888888888889

t_bhpt_merger = 1.10958353616297e-07
t_nr_merger = -0.837744212698453
theta = nu * (t_bhpt - t_bhpt_merger)
Theta_nr = nu * (t_nr - t_nr_merger)
```

The 10 fitted parameters are:

```python
theta_cut = -40.211386551818
t_cut_nr = -232.029085330417
theta_transition_width = 26.9556318610348
beta_left = 0.804278517755289
beta_right_edge = 0.808357178154863
beta_right_slope = 0.00161139010664821
alpha_left = 0.808778897989962
alpha_right_edge = 0.813372710961577
alpha_right_slope = -0.00680018833178102
phi0 = 1.24976456081864
```

The optimized source cutoff and transition are:

```python
t_bhpt_cut = t_bhpt_merger + theta_cut / nu  # -289.521983062131
theta_transition_end = theta_cut + theta_transition_width  # -13.2557546907833
t_bhpt_transition_end = t_bhpt_merger + theta_transition_end / nu  # -95.4414336626812
Theta_cut_nr = -32.109908488572
t_cut_nr = t_nr_merger + Theta_cut_nr / nu  # -232.029085330417
```

Functional form:

```python
# theta <= theta_cut
alpha(theta) = alpha_left
beta(theta) = beta_left

# theta_cut < theta < theta_cut + theta_transition_width
z = (theta - theta_cut) / theta_transition_width
H00 = 2*z**3 - 3*z**2 + 1
H01 = -2*z**3 + 3*z**2
H11 = z**3 - z**2
y_smooth = H00*y_left + H01*y_right_edge + H11*theta_transition_width*y_right_slope

# theta >= theta_cut + theta_transition_width
alpha(theta) = alpha_right_edge + alpha_right_slope * (theta - theta_transition_end)
beta(theta) = beta_right_edge + beta_right_slope * (theta - theta_transition_end)
```

The time map is the numerical integral of the smooth positive `beta(theta)`,
shifted so that `tau(t_bhpt_cut) = t_cut_nr`:

```python
d tau / d t_bhpt = beta(theta)
tau(t_bhpt_cut) = t_cut_nr
h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)
```

Transition values:

| location | theta | t_bhpt | mapped tau | alpha | beta |
| --- | ---: | ---: | ---: | ---: | ---: |
| cutoff/start | -40.211386551818 | -289.521983062131 | -232.029085330418 | 0.808778897989962 | 0.804278517755289 |
| transition end | -13.2557546907833 | -95.4414336626812 | -76.2409804458191 | 0.813372710961577 | 0.808357178154863 |

Diagnostics:

- q: `5`
- mode: `(2, 2)`
- time window: NR restricted to `-5000.1` through `100.0`, then common support only
- BHPT `calibrated`: `False`
- optimized phase rotation: `True`
- common NR support: `[-4999.9377442127, 98.5622557873012]`
- optimizer downsampled `mathcalE`: `0.000149235734978704`
- full-window `mathcalE`: `0.000149607512601296`
- pre-transition local `mathcalE`: `5.97737475151136e-05`
- smooth-transition local `mathcalE`: `0.000180926074469447`
- post-transition local `mathcalE`: `0.000914034223340297`
- alpha endpoint values `(used_start, used_end)`: `(0.808778897989962, 0.614805857516124)`
- beta endpoint values `(used_start, used_end)`: `(0.804278517755289, 0.85541009015362)`
- minimum alpha on used source support: `0.614805857516124`
- minimum beta on used source support: `0.800663287644468`
- pre-cutoff alpha and beta slopes: `0` by construction
- transition edge value jumps: `0` by construction for both alpha and beta

No extrapolated BHPT samples are used. The evaluated source samples are the
raw BHPT samples whose fitted `tau(t_bhpt)` lies inside the NR comparison
support, and the NR grid is then restricted to that common support.
