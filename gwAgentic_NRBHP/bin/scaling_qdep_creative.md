# Creative q-dependent scaling fit

This repeats the q-dependent scalar scaling experiment using 40 uniformly spaced q values from 3 to 10 and modes `(l,l)` for `l = 2, 3, 4, 5`. The waveform extraction and `mathcalE` definition follow `NRBHP_ansatz.py`: raw `BHPTNRSur1dq1e4` with `calibrated=False`, `NRHybSur3dq8` with `dt=0.1`, `f_low=5e-3`, and `t_start=-5000.1`.

For each q and mode I treated a constant time shift and phase rotation as nuisance alignment parameters when evaluating `mathcalE`; they are not part of the reported alpha/beta functions. `NRHybSur3dq8` warns above q about 8.01, so the q=8.01..10 part uses NR surrogate extrapolation.

## candidate families tested

| family | form | worst max mathcalE | mean median mathcalE |
|---|---|---:|---:|
| poly2 | `1 + A/q + B/q^2` | 0.11122335 | 0.0089386124 |
| poly3 | `1 + A/q + B/q^2 + C/q^3` | 0.080825229 | 0.0082088236 |
| log_poly2 | `1 + A/q + B/q^2 + C log(q)/q^2` | 0.080864272 | 0.0083774342 |
| exp_decay | `1 + A/q + B/q^2 + C exp[-D(q-3)]` | 0.080927318 | 0.0083412719 |
| exp_log_mix | `1 + (A/q) exp[B(q-3)] + (C/q^2) log(D q)` | 0.080913569 | 0.0083353282 |

For reference, the previous four-term `1/q` polynomial had these validated max errors with the same nuisance-alignment evaluation:

| l | min mathcalE | median mathcalE | max mathcalE | q at max |
|---:|---:|---:|---:|---:|
| 2 | 0.00051275989 | 0.0010217173 | 0.0022646289 | 3 |
| 3 | 0.0014174235 | 0.0032100239 | 0.0088387899 | 3 |
| 4 | 0.0054982148 | 0.010743526 | 0.049278533 | 3 |
| 5 | 0.010581142 | 0.018282525 | 0.080872676 | 3 |

## selected family

Selected family: `poly3`

```python
f(q; params) = 1 + A/q + B/q^2 + C/q^3
alpha(q, l) = f(q; alpha_params[l])
beta(q) = f(q; beta_params)
```

## beta parameters

| p0 | p1 | p2 |
| ---: | ---: | ---: |
| -1.22941907641 | 1.45479451154 | -1.08626488941 |

## alpha parameters

| l | p0 | p1 | p2 |
|---:|---:|---:|---:|
| 2 | -1.27834964328 | 1.92419866472 | -2.1399012335 |
| 3 | -2.99152320358 | 5.16843150682 | -5.17153099083 |
| 4 | -3.81270403574 | 7.97936337666 | -7.93697980013 |
| 5 | -4.89152105181 | 11.2621394245 | -10.9809045722 |

## achieved errors for selected family

| l | min mathcalE | median mathcalE | max mathcalE | q at max |
|---:|---:|---:|---:|---:|
| 2 | 0.00050388728 | 0.0010312108 | 0.0023169845 | 3 |
| 3 | 0.0014012228 | 0.0031593748 | 0.0090084386 | 3 |
| 4 | 0.0055410195 | 0.010616641 | 0.05007371 | 3 |
| 5 | 0.010672873 | 0.018028068 | 0.080825229 | 3 |

## lower-bound diagnostic

The requested target of `< 1e-3` for every mode is not reachable with any scalar-in-time alpha(q,l) and beta(q) family on the full window, because the per-q optimized scalar fits already exceed that for higher modes at low q. These values are the per-q scalar lower-ish errors before fitting a global q-function:

| l | min lower-ish mathcalE | median lower-ish mathcalE | max lower-ish mathcalE | q at max |
|---:|---:|---:|---:|---:|
| 2 | 0.00049311686 | 0.0010209659 | 0.0022645856 | 3 |
| 3 | 0.0013868707 | 0.0031771936 | 0.0088393118 | 3 |
| 4 | 0.0055052781 | 0.010526782 | 0.049286419 | 3 |
| 5 | 0.010385715 | 0.017806157 | 0.080789395 | 3 |

The creative forms reduce functional complexity and slightly change the q-trends, but they do not remove the residual higher-mode time-dependent mismatch. Reaching `< 1e-3` for `l=4` and `l=5` likely requires mode-dependent time reparameterization, time-dependent alpha/beta, or a shorter/weighted comparison window.
