# Summary

This note summarizes the setup and checks used to define the current target
error for the BHPT-to-NR waveform scaling task.

## Objective

The working goal is to find a simple theoretical scaling between a raw BHPT
waveform and an NR hybrid waveform such that the normalized time-domain error

```python
mathcalE = np.sum(np.abs(h_ref - h_model) ** 2) / (2 * np.sum(np.abs(h_ref) ** 2))
```

is reduced to `mathcalE <= 0.1`.

The immediate reference case is:

- mass ratio: `q = 5`
- mode: `(2, 2)`
- BHPT model: `BHPTNRSur1dq1e4`
- BHPT setting: `calibrated=False`
- NR model: `NRHybSur3dq8`
- NR reference waveform: nonspinning, `dt=0.1`, `f_low=5e-3`, `t_start=-5000.1`

## Steps Followed

1. Inspected the project root and confirmed the main reference file is
   `Untitled1.py`.
2. Read `Untitled1.py` to identify the waveform sources, the `(2, 2)` mode
   comparison, the `mass_norm = 1 / (1 + 1/q)` scaling, and the `mathcalE_error`
   function.
3. Read the `BHPTNRSurrogate` and `BHPTutils` documentation to confirm required
   dependencies and the role of the local surrogate data.
4. Created a local virtual environment at `.venv`.
5. Verified that the scientific Python dependencies are available through the
   local environment.
6. Found that `gwsurrogate` requires native BLAS/GSL libraries from the existing
   `ut_codex` conda environment, so the workflow needs:

   ```bash
   export LD_LIBRARY_PATH=/home/omkarnm1401/miniconda3/envs/ut_codex/lib:${LD_LIBRARY_PATH:-}
   ```

7. Set a local matplotlib cache directory to avoid writes under the home config
   directory:

   ```bash
   mkdir -p .cache/matplotlib
   export MPLCONFIGDIR=.cache/matplotlib
   ```

8. Applied the requested 4-core limit through BLAS/threading environment
   variables:

   ```bash
   export OMP_NUM_THREADS=4
   export OPENBLAS_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   export VECLIB_MAXIMUM_THREADS=4
   export NUMEXPR_NUM_THREADS=4
   ```

9. Verified that `gwsurrogate.LoadSurrogate("NRHybSur3dq8")` loads successfully
   from the local environment.
10. Ran a quick baseline comparison for `q=5` using the same basic interpolation
    behavior as `Untitled1.py`.

## Baseline Errors Observed

Using the current reference-script behavior, the errors were:

- raw BHPT: `mathcalE ~= 1.40`
- BHPT time scaled by `mass_norm`: `mathcalE ~= 1.36`
- BHPT time and amplitude scaled by `mass_norm`: `mathcalE ~= 1.11`

These values are above the target `mathcalE <= 0.1`.

## Important Caveat

The quick baseline produced interpolation-domain warnings because some requested
NR samples were outside the scaled BHPT time support. Future accepted results
should first restrict the NR time grid to the common support of the scaled BHPT
time array, interpolate only on that common support, and then compute
`mathcalE`.

Also, `Untitled1.py` plots BHPT strain multiplied by `mass_norm`, but its final
metric call only applies `mass_norm` to the BHPT time array. Any future result
should state exactly which time, amplitude, and phase scalings were used in the
metric calculation.

## Current Status

The new target error is documented as `mathcalE <= 0.1`. That target has not yet
been achieved by the simple baseline checks above; the next step is to optimize
a low-dimensional scaling, likely involving time scaling, amplitude scaling,
time shift, and constant phase rotation, while enforcing common-support
interpolation.
