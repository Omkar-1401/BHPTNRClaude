# Physical-smooth q-dependent scaling fit

This run generalizes the q=5 `stricter_physical_smooth` ansatz over
mass ratios between 3 and 8. The ansatz is unchanged in structure:
`alpha` and `beta` are constant before a physical cutoff, pass through
a cubic Hermite transition, and are linear in the merger-centered
physical source time coordinate after the transition.

The physical coordinates are:

```python
nu = q / (1 + q)**2
theta = nu * (t_bhpt - t_bhpt_merger)
Theta_nr = nu * (t_nr - t_nr_merger)
Theta_cut_nr = nu * (t_cut_nr - t_nr_merger)
```

The q-dependence is represented by polynomials in `y = 1/q`:

```python
C(q) = c0 + c1*y + c2*y**2 [+ c3*y**3]
```

For a given q, the coefficients define:

```python
theta_end = theta_cut + theta_transition_width

# theta <= theta_cut
alpha(theta) = alpha_left
beta(theta) = beta_left

# theta_cut < theta < theta_end
z = (theta - theta_cut) / theta_transition_width
H00 = 2*z**3 - 3*z**2 + 1
H01 = -2*z**3 + 3*z**2
H11 = z**3 - z**2
y_smooth = H00*y_left + H01*y_right_edge + H11*theta_transition_width*y_right_slope

# theta >= theta_end
alpha(theta) = alpha_right_edge + alpha_right_slope * (theta - theta_end)
beta(theta) = beta_right_edge + beta_right_slope * (theta - theta_end)

t_cut_nr = t_nr_merger + Theta_cut_nr / nu
d tau / d t_bhpt = beta(theta)
tau(t_bhpt_merger + theta_cut / nu) = t_cut_nr
h_model(tau(t_bhpt)) = alpha(theta) * exp(1j * phi0) * h_BHPT(t_bhpt)
```

Calibration grid: `41` q values from `3` to `8`.
Rows used in the polynomial coefficient regression: `41`. Rows with invalid support or `mathcalE >= 0.001` are held out of the regression.
Fit settings: source stride `3`, NR stride `8`, top starts `3`, maxiter `850`, full-refine threshold `0.00025`.
The transition-width safety bound used by this script is `130` in `theta`.
For the saved cache used here, the low-q rows were also given a targeted
denser cleanup pass after the wider transition bound was introduced;
the per-q table below is the authoritative record of the final cached
independent fits.

The master q-polynomial describes the physical shape coefficients.
The merger-centered time anchor `Theta_cut_nr` and phase `phi0` have
formal q-polynomial values, but they are also treated as nuisance
alignment parameters and re-optimized when reporting the master-model
error statistics. This mirrors the previous PN-opt q-dependent workflow
and keeps the alpha/beta calibration from absorbing arbitrary absolute
time and phase offsets.

The selected polynomial degree is:

```python
degree = 3
```

## Selected Master Coefficients

| coefficient | c0 | c1 | c2 | c3 |
|:---|---:|---:|---:|---:|
| theta_cut | -92.3338429543 | 694.831047699 | -2964.18641159 | 3861.56089336 |
| theta_transition_width | 261.308902047 | -3057.53093177 | 12860.8258821 | -16835.66783 |
| Theta_cut_nr | -78.1974449399 | 542.110799213 | -2121.91284552 | 2728.44834907 |
| beta_left | 0.998719022642 | -1.19162345007 | 1.2599900999 | -0.820543932492 |
| beta_right_edge | 1.08481143444 | -1.75875274987 | 1.97124858648 | 0.105700786237 |
| beta_right_slope | -0.0208696818128 | 0.368022924913 | -1.85040322933 | 2.89579123498 |
| alpha_left | 0.989819129984 | -1.08218689572 | 1.01961340477 | -0.665242423261 |
| alpha_right_edge | -0.805339752253 | 24.1245295682 | -111.72625556 | 155.852746602 |
| alpha_right_slope | -0.0750088402908 | 0.973745600613 | -4.36264346743 | 5.91033162067 |
| phi0 | -201.528082028 | 3270.66653597 | -15860.40223 | 23648.6296399 |

## Error Summary

| model | min mathcalE | median mathcalE | mean mathcalE | max mathcalE | q at selected max |
|:---|---:|---:|---:|---:|---:|
| independent per-q physical-smooth, valid-support rows | 0.00010451168 | 0.0001504883 | 0.00021536918 | 0.00068641537 |  |
| independent per-q rows used for q-regression | 0.00010451168 | 0.0001504883 | 0.00021536918 | 0.00068641537 |  |
| quadratic q-polynomial, optimized `Theta_cut_nr`/`phi0` | 0.00013292221 | 0.00019818835 | 0.00030931364 | 0.0010839933 |  |
| cubic q-polynomial, optimized `Theta_cut_nr`/`phi0` | 0.00012855061 | 0.00034149525 | 0.00038024779 | 0.00080742577 |  |
| selected q-polynomial | 0.00012855061 | 0.00034149525 | 0.00038024779 | 0.00080742577 | 3.75 |

The extrapolation request to q=2 and q=10 is treated as formal polynomial
extrapolation only. The calibration and error statistics above use q in [3, 8].

## Formal Extrapolated Parameter Values

| q | theta_cut | theta_width | Theta_cut_nr | beta_left | beta_edge | beta_slope | alpha_left | alpha_edge | alpha_slope | note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 2 | -21.111111 | 2 | -2 | 0.61533683 | 0.7114598 | 0.03 | 0.62047373 | 2.8 | 0.05 | clipped by bounds |
| 10 | -48.631041 | 67.3284 | -42.477045 | 0.89133603 | 0.92875435 | 0.00032436962 | 0.89113133 | 0.6457034 | -0.015350383 |  |

## Per-q Results

| q | per-q mathcalE | selected master mathcalE | coverage | used in q-fit | theta_cut | theta_width | beta_left | alpha_left | master phi0 |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 3 | 0.00068641537 | 0.00071426112 | 0.99849 | True | -45.20567 | 42.818009 | 0.71123499 | 0.71776525 | -0.35513225 |
| 3.125 | 0.00059985785 | 0.00063685341 | 0.998216 | True | -45.547602 | 43.758389 | 0.71951857 | 0.72633316 | -2.4078523 |
| 3.25 | 0.00051975955 | 0.00065093741 | 0.998137 | True | -48.859237 | 48.497849 | 0.72736573 | 0.73387775 | 1.5053821 |
| 3.375 | 0.0004584879 | 0.00069283198 | 0.998098 | True | -48.602136 | 48.495722 | 0.73482196 | 0.74130691 | -1.1644594 |
| 3.5 | 0.00040739683 | 0.00074025615 | 0.998157 | True | -53.3114 | 52.534067 | 0.74181895 | 0.74773635 | 2.1740613 |
| 3.625 | 0.0005095886 | 0.00078236677 | 0.998216 | True | -16.077175 | 36.634829 | 0.74901843 | 0.7557365 | -1.0240528 |
| 3.75 | 0.00032470426 | 0.00080742577 | 0.998314 | True | -54.369295 | 51.934396 | 0.75487214 | 0.76081409 | 1.8208278 |
| 3.875 | 0.00029569769 | 0.0008025299 | 0.998431 | True | -58.854419 | 57.350673 | 0.7608969 | 0.76741866 | -1.8369459 |
| 4 | 0.00026250693 | 0.00075948803 | 0.998569 | True | -51.810058 | 50.545362 | 0.76665717 | 0.77211835 | 0.58229495 |
| 4.125 | 0.00024776407 | 0.0006791378 | 0.998706 | True | -44.841529 | 32.859425 | 0.77220215 | 0.77781005 | 2.8112799 |
| 4.25 | 0.00023641656 | 0.00057205299 | 0.998902 | True | -38.859337 | 19.718997 | 0.77743603 | 0.78291655 | -1.4239311 |
| 4.375 | 0.00020468282 | 0.00045539682 | 0.999059 | True | -51.892499 | 44.610248 | 0.7823519 | 0.78743799 | 0.45515446 |
| 4.5 | 0.00019073591 | 0.00034792571 | 0.999235 | True | -45.787569 | 33.73827 | 0.78716799 | 0.79234682 | 2.1668008 |
| 4.625 | 0.00018033223 | 0.00026541939 | 0.999392 | True | -35.37733 | 18.379218 | 0.79174362 | 0.7969639 | -2.5540303 |
| 4.75 | 0.00017021227 | 0.00021772878 | 0.999588 | True | -35.325292 | 17.575417 | 0.7960881 | 0.80117335 | -1.1427063 |
| 4.875 | 0.00015856994 | 0.00020768181 | 0.999706 | True | -35.314855 | 20.412208 | 0.80030718 | 0.80509706 | 0.12975839 |
| 5 | 0.0001504883 | 0.00023142006 | 0.999863 | True | -35.118386 | 19.677508 | 0.80430177 | 0.80902117 | 1.268069 |
| 5.125 | 0.00014435054 | 0.00028011411 | 0.999961 | True | -35.366986 | 19.963128 | 0.80813578 | 0.8128781 | 2.2752111 |
| 5.25 | 0.00013888534 | 0.00034149525 | 0.999941 | True | -35.029376 | 20.19133 | 0.81186342 | 0.81612942 | -3.1266648 |
| 5.375 | 0.00013572221 | 0.00040246617 | 0.999941 | True | -34.864712 | 20.621051 | 0.81538886 | 0.81958209 | -2.3622037 |
| 5.5 | 0.00013082472 | 0.0004514422 | 0.999941 | True | -35.077361 | 21.369817 | 0.81884772 | 0.82283367 | -1.7132166 |
| 5.625 | 0.00013209694 | 0.00047981938 | 0.999922 | True | -34.276812 | 20.036585 | 0.82213364 | 0.82592197 | -1.1754049 |
| 5.75 | 0.00012959754 | 0.00048289719 | 0.999961 | True | -41.342986 | 27.146113 | 0.82530519 | 0.82897215 | -0.74170019 |
| 5.875 | 0.00012443241 | 0.00046032572 | 0.999941 | True | -41.118842 | 29.768557 | 0.82836669 | 0.83167735 | -0.408485 |
| 6 | 0.00013777237 | 0.00041536562 | 0.999961 | True | -41.613958 | 34.832583 | 0.83130009 | 0.8343252 | -0.17674075 |
| 6.125 | 0.00017022203 | 0.00035543107 | 0.999941 | True | -42.980396 | 48.42433 | 0.8341512 | 0.83835564 | -0.038018545 |
| 6.25 | 0.00020932585 | 0.00029188885 | 0.999922 | True | -38.021378 | 35.501331 | 0.83657702 | 0.84078096 | 0.014402143 |
| 6.375 | 0.00015852285 | 0.00023468758 | 0.999961 | True | -43.369933 | 43.357901 | 0.83958307 | 0.84254668 | -0.023232774 |
| 6.5 | 0.00015512584 | 0.00018986671 | 0.999961 | True | -47.096371 | 54.493669 | 0.84216038 | 0.84416928 | -0.14552958 |
| 6.625 | 0.00014000394 | 0.00015973671 | 0.999961 | True | -52.182028 | 53.539081 | 0.84467235 | 0.84712183 | -0.34862391 |
| 6.75 | 0.00013787175 | 0.00014310737 | 0.999961 | True | -57.260255 | 52.953773 | 0.84705861 | 0.8492654 | -0.63001169 |
| 6.875 | 0.00013752528 | 0.00013639045 | 0.999922 | True | -52.261819 | 49.019104 | 0.84949075 | 0.85186775 | -0.98829927 |
| 7 | 0.00013072243 | 0.00013475558 | 0.999961 | True | -51.856328 | 47.706246 | 0.85176441 | 0.85373043 | -1.4166952 |
| 7.125 | 0.00012854772 | 0.00013406618 | 0.999961 | True | -52.008365 | 47.909313 | 0.85395395 | 0.85577856 | -1.9161642 |
| 7.25 | 0.00012355629 | 0.00013247884 | 0.999941 | True | -46.397202 | 42.977703 | 0.85621474 | 0.85799269 | -2.4809511 |
| 7.375 | 0.00011586142 | 0.0001302903 | 0.999961 | True | -40.641843 | 35.200234 | 0.85834074 | 0.85998822 | -3.1151908 |
| 7.5 | 0.00011213136 | 0.00012855061 | 0.999922 | True | -40.636842 | 35.35596 | 0.86034299 | 0.86218921 | 2.4721018 |
| 7.625 | 0.00010451168 | 0.00012856972 | 0.999941 | True | -34.369814 | 26.593069 | 0.86234535 | 0.8639747 | 1.7172515 |
| 7.75 | 0.00010485655 | 0.0001311977 | 0.999941 | True | -34.081933 | 27.688477 | 0.86422441 | 0.8658077 | 0.90212799 |
| 7.875 | 0.00011447571 | 0.00013668972 | 0.999941 | True | -40.12256 | 41.287295 | 0.86607975 | 0.86825693 | 0.028372736 |
| 8 | 0.00010957637 | 0.00014481228 | 0.999941 | True | -35.216185 | 37.343302 | 0.86791507 | 0.86936817 | -0.89902305 |

No extrapolated BHPT samples are used in the error calculation; each
evaluation masks the NR waveform to the common support of the transformed
BHPT time array before interpolation.

No plotting files were generated in this run, per the current instruction
to focus on the calibration script and markdown first.
