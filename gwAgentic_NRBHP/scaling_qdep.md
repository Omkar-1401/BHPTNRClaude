# q-dependent scaling fit

Fit form:

```python
alpha(q, l) = 1 + A_alpha(l)/q + B_alpha(l)/q**2 + C_alpha(l)/q**3 + D_alpha(l)/q**4
beta(q) = 1 + A_beta/q + B_beta/q**2 + C_beta/q**3 + D_beta/q**4
```

The fit used 40 uniformly spaced q values from 3 to 10 and modes `(l,l)` for `l = 2, 3, 4, 5`. The `mathcalE` definition matches `NRBHP_ansatz.py`. A constant time shift and phase rotation were treated as nuisance alignment parameters when measuring the final errors; they are not included in the alpha/beta ansatz coefficients.

`NRHybSur3dq8` warns that q values above about 8.01 are outside its training range, so the high-q part of this q=3..10 sweep uses NR surrogate extrapolation.

## beta coefficients

| A_beta | B_beta | C_beta | D_beta |
|---:|---:|---:|---:|
| -1.24116450042 | 1.63823509893 | -1.97154177912 | 1.33749010735 |

## alpha coefficients

| l | A_alpha | B_alpha | C_alpha | D_alpha |
|---:|---:|---:|---:|---:|
| 2 | -1.34005326034 | 2.88788867946 | -6.79063043803 | 7.02639406495 |
| 3 | -3.10607054305 | 6.94779267852 | -13.7319640012 | 12.9087168665 |
| 4 | -3.98079244161 | 10.5911609772 | -20.5071088134 | 18.9621851423 |
| 5 | -4.74037229115 | 8.87898607451 | 0.581593859672 | -17.5244928804 |

## achieved errors

| l | min mathcalE | median mathcalE | max mathcalE | q at max |
|---:|---:|---:|---:|---:|
| 2 | 0.00051275989 | 0.0010217173 | 0.0022646289 | 3 |
| 3 | 0.0014174235 | 0.0032100239 | 0.0088387899 | 3 |
| 4 | 0.0054982148 | 0.010743526 | 0.049278533 | 3 |
| 5 | 0.010581142 | 0.018282525 | 0.080872676 | 3 |

## notes

The earlier hard error thresholds were relaxed. The table above reports the achieved errors from the fitted constant-in-time q-dependent ansatz. The higher modes, especially l=4 and l=5 at low q, retain larger errors because a single constant beta(q) and scalar alpha(q,l) cannot fully absorb their residual time-dependent structure over the full `t >= -5000.1` window.
