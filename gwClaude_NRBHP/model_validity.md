# BHPT-NR Calibration Model Validity

Mismatch metric: `mathcalE = (‖h₁‖² + ‖h₂‖² − 2Re⟨h₁,h₂⟩) / (2‖h₁‖²)`.  
Reference: NRHybSur3dq8 (2,2) mode, nonspinning.  
At evaluation, `t0_nr` and `phi0` are re-polished (Nelder-Mead, 2 DOF) as alignment
nuisances; all other parameters are fixed by the regression.  
**invalid** = time map non-monotone or coverage < 0.88 after nuisance polish.

---

## Model Summaries

### PN_opt_creative_q_dep
10-parameter logistic-switch model (creative) with all parameters fit as cubic
polynomials in `1/q` across a 40-q training grid [3, 8].  Median in-range
mismatch 7.66e-5.  The regression coordinate `1/q` is unbounded and
ill-conditioned at low q: the `phi0` coefficient extrapolates to ~369 rad at
q=2, producing a completely wrong phase and mismatch of 2.14e-01.  This model
motivated the shift to bounded coordinates (chi_f, nu) in later variants.

### PN_opt_remnant_partial
Same 10-parameter per-q model.  `beta_r` is factorised as
`r_beta(chi_f) × beta_r_physical(M_f, chi_f, q)` where the physical part uses
the exact Berti et al. (2006) (2,2,0) QNM formula evaluated via surfinBH at
any q.  The remaining 9 parameters are cubic polynomials in `chi_f(q)` from
NRSur3dq8Remnant.  Since `chi_f` is bounded and monotone (0.3 to 0.54 in
[3,8]), extrapolation stays finite at all tested q.  Accuracy is sensitive to
training grid density: 40-q gives best in-range median (6.21e-05) but 5.84e-02
at q=2, whereas the 74-q extended grid recovers 3.00e-03 at q=2 by anchoring
the chi_f polynomial over a wider range.

### PN_opt_nu_Pade  *(selected form: poly_nu)*
PP-anchored deg-4 polynomial in `nu = q/(1+q)²` for the 6 physical parameters
(`alpha_i, beta_i, beta_r → 1`; `alpha_E, alpha_J, beta_L → 0` as `nu → 0`).
`p0`, `w`, and the alignment nuisances are fit unanchored.  Reuses the 40-q
per-q cache from PN_opt_creative_q_dep.  Best in-range median (7.05e-5) and
best extrapolation at q=2.5 (3.87e-4); PP anchoring costs essentially nothing
in-range while pinning physical limits.  Breaks at q=1.5 because unanchored
`p0` and `w` roll over and make the time map non-monotone.

### q_dep_classic
Constant amplitude scaling `alpha` and time-stretch `beta` per q (Islam et al.
2204.01972 model class), calibrated here against NRHybSur3dq8.  Only 2 physical
DOF fit as PP-anchored quartic polynomials in `1/q`.  In-range median ~1e-3,
roughly 15× worse than the logistic-switch models.  The low dimensionality makes
extrapolation stable: time map remains well-formed at all tested q, and finite
mismatches are recovered even at q=1.5 (3.00e-01).

---

## Statistics

Each model is fitted twice: once on a **40-q [3,8]** training grid
(`np.linspace(3,8,40)`) and once on the **64-q [3,8]** training grid (all 64
waveforms cached in [3,8]: the 40-q linspace points plus 24 intermediate
round-number q values).  Each fit is evaluated at the training grid plus four
single extrapolation points; single-point columns (q = 3, 2.5, 2.0, 1.5)
differ between rows because different training grids produce different
polynomial coefficients.  The 64-q `med [3,10]` column is omitted (training
does not extend past q=8).

| model | train | med [3,8] | med [3,10] | q=3 | q=2.5 | q=2 | q=1.5 |
|:---|:---|---:|---:|---:|---:|---:|---:|
| PN_opt_creative_q_dep | 40-q | 7.66e-05 | 7.98e-05 | 2.13e-04 | 1.23e-02 | 2.14e-01 | invalid |
| PN_opt_creative_q_dep | 64-q | 8.07e-05 | — | 4.82e-04 | 4.51e-02 | 6.20e-02 | invalid |
| PN_opt_remnant_partial | 40-q | **6.21e-05** | **6.31e-05** | 1.83e-04 | 2.27e-03 | 5.84e-02 | 4.02e-01 |
| PN_opt_remnant_partial | 64-q | 7.48e-05 | — | 2.85e-04 | 1.43e-03 | **3.66e-03** | 3.42e-01 |
| PN_opt_nu_Pade | 40-q | 7.05e-05 | 7.61e-05 | **1.68e-04** | **3.87e-04** | 6.46e-02 | invalid |
| PN_opt_nu_Pade | 64-q | 8.31e-05 | — | 2.56e-04 | 2.30e-02 | invalid | invalid |
| q_dep_classic | 40-q | 1.09e-03 | 1.06e-03 | 2.27e-03 | 3.93e-03 | 8.02e-03 | **2.96e-01** |
| q_dep_classic | 64-q | 1.07e-03 | — | 2.27e-03 | 3.91e-03 | 9.04e-03 | 3.61e-01 |

Bold = best in that column.  All values are `mathcalE` in scientific notation;
**invalid** = time map non-monotone after nuisance polish (coverage=NaN,
error sentinel=50); — = not evaluated.

### Notes

- **64-q vs 40-q in-range**: adding the 24 intermediate training points barely
  changes the median (within ~15%) for all models.  The extra density does not
  meaningfully improve in-range accuracy.

- **Remnant training-grid sensitivity**: the 40-q fit collapses at q=2
  (5.84e-02); the 64-q fit recovers 3.66e-03 at q=2, which is the original
  accepted fit result.  The chi_f polynomial is better anchored by the denser
  training grid.

- **PN_opt_nu_Pade regression at 64-q**: the 24 extra intermediate training
  points shift the degree-4 poly-in-nu enough that the time map loses
  monotonicity at q=2 (valid at 6.46e-02 with 40-q).  The 40-q grid is
  preferred for nu_Pade extrapolation.

- **At q=1.5** the only models that stay finite are remnant and classic; none
  is accurate.  The BHPT surrogate itself limits q=1.5 accuracy — a direct
  per-q fit bottoms out near 3–4e-01 independent of the q-regression.

- **Recommended use (training-grid aware):**
  - q ∈ [3, 8]: remnant/40-q or pade/40-q (6–7e-05 median); 40-q training
    preferred.
  - q ∈ [2.5, 3): pade/40-q (3.87e-04 at q=2.5).
  - q ∈ [2, 2.5): remnant/64-q (3.66e-03 at q=2).
  - q < 2: no model usable without direct calibration anchor points.
