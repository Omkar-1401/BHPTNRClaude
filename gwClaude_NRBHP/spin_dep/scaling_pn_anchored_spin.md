> **STATUS 2026-09-27 -- INSTALLED: switch + merger fitted.**  `coeffs_spin.json` now holds p00, p01, w0, r0, m0, m1 fitted on the 2d spin grid together with the 4 spin terms (all still spin-independent; b1, b2, a2 = nonspinning pn_anchored): **mean 1.1722e-03 / median 8.649e-04 / max 6.577e-03** on the 96-point grid (published 1.2808e-03 / 7.6858e-04 / 9.6416e-03), same point from 3 of 4 starts; self-test B.pn_anchored_spin_mean reproduces it exactly (job 3474054).  Code: `fit_pn_anchored_spin.model9()` reads the installed 9; `--refit` fits the 10.  At chi1 = 0 the model no longer reduces exactly to nonspinning pn_anchored.  Previous (9 frozen, mean 1.6788e-03 / median 8.932e-04): `coeffs_spin_frozen9_20260926.json`.  Details and the full q-chi1 grid (incl. q = 2, 2.5): the 2026-09-27 sections at the END.  The Model block and constants C1-C7 below are current; the Results / Provenance numbers in the body predate the 09-24 units correction and the 09-26 fixes, and the body's "9 frozen" protocol is superseded.


# pn_anchored + spin — 4 spin coefficients on a frozen 9-coefficient nonspinning base

*(Title and opening result as of 2026-09-09. Since 2026-09-27 six of the nine non-spin coefficients
-- p00, p01, w0, r0, m0, m1 -- are fitted on the spin grid; only b1, b2, a2 come from the nonspinning fit.)*

**Result: aligned primary spin over the whole range chi1 in [-0.6, +0.8] costs 4 fitted
coefficients, and the model reaches median 7.686e-04 across a 96-point (q, chi1) grid
against 5.543e-04 for the frozen nonspinning model at chi1 = 0.** So the entire spin
extension is a ~1.4x degradation of the median, for 4 parameters, with alpha untouched
and the ringdown anchor's spin dependence *derived* rather than fitted.

The cost is not evenly spread. Accuracy is U-shaped in chi1 with its minimum near
chi1 = -0.2 (4.86e-04, *better* than the nonspinning column) and degrades ~6x toward
each end — 1.28e-03 at chi1 = -0.6 and 3.50e-03 at chi1 = +0.8. Diagnosing that U-shape
is the open work; two candidate causes are identified at the end, and one competing
explanation (a chi1 < 0 / chi1 >= 0 domain split) has already been tested and falsified.

This is a FULL-WAVEFORM model scored by waveform mismatch, the same objective and the
same relative-L2 definition as the nonspinning models, so its numbers ARE comparable to
`CLAUDE.md`'s table. The chi1 = 0 column is a null check, not a fit.

Files: `fit_pn_anchored_spin.py`, `pn_anchored_spin_results/coeffs_spin.json`,
`Agentic_spin_plots/pn_anchored_spin/pn_anchored_spin_alpha_beta_overlay_*.pdf`.
Design and step-1 measurement: `spin_extension_plan.md`.
Sibling (tested, did not pay): `fit_pn_anchored_spin_split.py`.

## Model

```python
X1   = q/(1+q)                nu = q/(1+q)**2           base = X1**1.2
x_pp(t) = (m1*omega_GW/2)**(2/3)  from the ppBHPT phase (m1 clock), smoothed over 80 m1
xc = X1**(-4/5) * min(x_pp, XCLIP=0.26)                 # the note's x = (M*Omega)**(2/3), Eq. 6; C3
p_loss(t) = 0.5*(e_hat + j_hat)                         # wf-flux coordinate, measured
S(t) = sigmoid( (p_loss - (p00 + p01*nu)) / w0 )        # merger blend

# --- the two branches -----------------------------------------------------------
alpha_insp = base * ( 1 + (55/42)*nu*xc + nu*a2*xc**2 )        # NO spin
alpha_mr   = base * ( m0 + m1*nu )                             # NO spin

beta_insp  = base * ( 1 + nu*( b1c*xc + b2c*xc**2 ) )
beta_mr    = beta_r(q, chi1) * r0

alpha = (1-S)*alpha_insp + S*alpha_mr
beta  = (1-S)*beta_insp  + S*beta_mr
tau(t) = t0_nr + integral_{T_ANCHOR=-100}^{t} beta dt'         # phi0 analytic; T_ANCHOR is ppBHPT time (m1 units), C10

# --- spin enters in EXACTLY two places ------------------------------------------
b1c = b1 + s11*chi1 + s12*chi1**2          # 2 fitted
b2c = b2 + s21*chi1 + s22*chi1**2          # 2 fitted

beta_r(q, chi1) = w220(chi1) * M_f / w220(chi_f)                 # 0 fitted, DERIVED, C6
    # pp side w220(chi1): the primary rings down, in m1 units (the ppBHPT clock)
    # NR side w220(chi_f)/M_f: the real remnant, in M units (the NR clock)
    M_f, chi_f = NRSur3dq8Remnant(q, [0,0,chi1], [0,0,0])        # C7
    w220(chi)  = exact Kerr (2,2,0) QNM frequency, qnm package      # C5; counter-rotating for chi < 0
    # history: (1+q)/q error until 09-24 (UNITS_AUDIT.md A1); BCW fit + 0.3737 rescale until 09-26
```

Every number in the block above is explained in **Constants and their sources** (after
the Coefficients section): `1.2` = 6/5 (C1), `55/42` (C2), `x` (C3), `XCLIP = 0.26` (C9),
`T_ANCHOR = -100` (C10), `p_loss` normalisation (C11), `X1**(-4/5)` (C3), `w220` (C5),
`beta_r` (C6), NRSur3dq8Remnant (C7).

At chi1 = 0 every spin term vanishes identically (each carries a factor of chi1, and
`beta_r` reduces to the shipped nonspinning expression), so the model is bit-identical to
nonspinning `pn_anchored` there.  [SUPERSEDED 2026-09-27: the FORM still reduces at chi1 = 0, but
p00, p01, w0, r0, m0, m1 are now fitted on the 2d spin grid, so the coefficient VALUES differ from
nonspinning pn_anchored and the model is no longer bit-identical to it there -- see the status note
at the top and "Anchors (derived, not fitted)" below.]

## Anchors (derived, not fitted)

*Written 2026-09-27 against the current code, `spin_dep/fit_pn_anchored_spin.py` (below "F"), and
`physics_common.py` (shared definitions). Equation numbers are those of
`../alpha_beta_pn_scaling_note_revised.pdf` ("the note"). C1-C7 below give the same facts in short
form.*

**The two clocks.** The ppBHPT input (BHPTNRSur2dq1e3, `calibrated=False`) has time `t` in units of
the primary mass m1; the NR target (NRHybSur3dq8) has time `τ` in units of the total mass M. The
time map is `dτ/dt = β(t)`, so every anchor for β is an NR-clock rate over a ppBHPT-clock rate. The
bare change of mass unit would be `β = m1/M = X1`.

**Coefficient status (2026-09-27, `pn_anchored_spin_results/coeffs_spin.json`, key `spin_base` =
F `SPIN_BASE`, F:145-155).** `b1, b2, a2` are the nonspinning pn_anchored values
(`../pn_anchored_results/coeffs.json`; F `model9()` refuses a file where they differ, F:158-171);
`p00, p01, w0, r0, m0, m1` and the four spin terms `s11, s12, s21, s22` are fitted on the 96-point
(q, χ1) grid. All nine non-spin coefficients are spin-independent (functions of ν only). No anchor
below carries a fitted parameter.

### A1. Inspiral anchor: β_LO = α₂,LO = X1^(6/5)

`base = X1^(6/5) = (q/(1+q))^(6/5) = (m1/M)^(6/5)`, F:233 (`base = X1 ** 1.2`).

*What it is.* The leading-order (Newtonian) time map between the ppBHPT and NR clocks and the
leading-order (2,2) amplitude ratio; the base of both β and α.

*Derivation (note Sec. 3).* At Newtonian order `dx_pp/dt = (64/5) η_pp x_pp^5` with `η_pp = m2/m1 = 1/q`,
and `dx_NR/dτ = (64/5) ν x_NR^5` (Eq. 7). With `τ = βt` the orbital phase is shared, so
`Ω_NR = Ω_pp/β`, `x_NR = β^(−2/3) x_pp` (Eq. 8), and the mapped chirp rate is
`(64/5) η_pp β^(5/3) x_NR^5` (Eq. 9). Equality with the binary's requires `η_pp β^(5/3) = ν` (Eq. 10);
with `ν/η_pp = X1²` (Eq. 11), `β_LO = (ν/η_pp)^(3/5) = X1^(6/5)` (Eq. 12). Matching the Newtonian (2,2)
amplitudes `η_pp x_pp` and `ν x_NR` (Eqs. 13-14) gives `α₂,LO = X1² β^(−2/3) = X1^(6/5) = β_LO`
(Eqs. 15-16).

*Why derived.* The exponent follows from the linearity of the Newtonian chirp rate in the mass ratio
and the `β^(−2/3)` scaling of x, so the leading q-dependence is known and is not fitted. It tends to 1
as q → ∞. It is spin-independent: the spin corrections to the chirp rate start at 1.5PN and are
carried by the fitted terms below.

*Use in this model* (F:238-241):
- `β_insp = X1^(6/5) [1 + ν(b1c xc + b2c xc²)]`, `b1c = b1 + s11 χ1 + s12 χ1²`,
  `b2c = b2 + s21 χ1 + s22 χ1²` (F:234-235). The note's Eq. 58 with a ν-suppressed residual
  (Eqs. 61-62): `b1, b2` (nonspinning fit) and `s11, s12, s21, s22` (spin fit) all multiply
  `ν xc` or `ν xc²`, so `β_insp → X1^(6/5)` as ν → 0.
- `α_insp = X1^(6/5) [1 + (55/42) ν xc + a2 ν xc²]` — note Eq. 57 (A3); no spin term.
- `α_mr = X1^(6/5) (m0 + m1 ν)` — `X1^(6/5)` is only the normalisation carried over from the inspiral;
  `m0, m1` (fitted on the spin grid) set the merger level. The note derives `X1^(6/5)` for the inspiral
  and says merger–ringdown needs its own anchor (Sec. 8, after Eq. 62).

### A2. Ringdown anchor: β_r(q, χ1) = w220(χ1) · M_f / w220(χ_f)

Code: `beta_r_phys(q, chi1) = w_pp_qnm(chi1) * mf / w220(chif)` (F:208-227), with
`w_pp_qnm = w220 = physics_common.w220_exact` (F:90-95, 203-205).
- `w220(χ)` = `M ω_220`, real part of the Kerr l = m = 2, n = 0 quasinormal-mode frequency in units of
  the ringing hole's own mass, **exact**: Leaver's continued fraction via the `qnm` package
  (L. C. Stein 2019, JOSS 4, 1683), `physics_common.py:54-72`. For χ < 0 the (2,2) mode is the
  counter-rotating branch `|ω_{2,−2}(|χ|)|` (`physics_common.py:59`). This matters on both sides here:
  χ1 < 0 on the ppBHPT side, and χ_f < 0 on the NR side at large q with χ1 = −0.6 (χ_f = −0.125 at q = 8).
- ppBHPT side: the ringing hole is the primary itself, mass m1 and spin χ1 (the secondary's
  contribution is neglected), so `ω_pp = w220(χ1)/m1`, i.e. `w220(χ1)` per unit of the ppBHPT clock —
  no mass conversion.
- NR side: the remnant, `ω_NR = w220(χ_f)/(M_f M)`, i.e. `w220(χ_f)/M_f` per unit of the NR clock, with
  `M_f` (units of M) and `χ_f` from surfinBH `NRSur3dq8Remnant` (Varma et al. 2019, PRL 122, 011101) at
  `(q, [0,0,χ1], [0,0,0])` (F:60, 225-226).

*Why it is the frequency ratio.* The mapped ppBHPT waveform has frequency `(dφ_pp/dt)/(dτ/dt) = ω_pp/β`.
For the two ringdowns to stay in phase it must equal `ω_NR`, so at ringdown
`β = dτ_NR/dt_pp = ω_pp/ω_NR = w220(χ1) M_f / w220(χ_f)`. Both sides move with spin; the spin
dependence of the anchor costs no parameter.

*No free parameter; `r0` multiplies it.* `β_MR = r0 · β_r(q, χ1)` (F:239), with `r0` fitted on the spin
grid (spin-independent; 0.7987 in the installed file). At χ1 = 0 the anchor reduces to the nonspinning
expression `0.373672 · M_f / w220(χ_f)` (w220(0) = 0.373672). Values (evaluated 2026-09-27 with the
functions above): q = 3: 0.748 / 0.763 / 0.859 at χ1 = −0.6 / 0 / +0.8; q = 8: 0.876 / 0.879 / 0.906.

*What it replaces (2026-09-26).* Until 2026-09-26 w220 was the Berti–Cardoso–Will prograde fit
`f1 + f2(1−χ)^f3` (1.5251, −1.1568, 0.1292; PRD 73, 064030, Table VIII), which (a) is 1.5% low at χ = 0
(0.3683 vs 0.37367), (b) was used on the ppBHPT side rescaled to 0.3737 (`W_SCHW`, `RESCALE_QNM`,
F:86-87, now unused) but raw on the NR side, and (c) was extrapolated to χ < 0, where it is 3% low at
χ = −0.2 and 7% low at χ = −0.6. The exact values agree with the measured ppBHPT ringdown frequency at
q = 5 to +0.09 / +0.26 / −0.67 / +0.18 % for χ1 = −0.2 / 0 / 0.2 / 0.6 (`physics_common.py:6-15`). Before
2026-09-24 the ppBHPT side also carried a spurious `(1+q)/q` (`UNITS_AUDIT.md` A1). `QF1..QF3` (F:61) are
unused.

### A3. Fixed 1PN amplitude slope 55/42

`PN_SLOPE = 55/42` (F:97), the ν x coefficient of `α_insp`. The 1PN (2,2) amplitude is
`h22 = h22^N [1 + (−107/42 + (55/42)ν) x + 2π x^(3/2) + …]` (note Eq. 55); the `−107/42` and `2π x^(3/2)`
terms are present in the test-particle limit and hence already in the ppBHPT waveform, leaving
`1 + (55/42) ν x` in the ratio at fixed x (Eq. 56), which is the ν x term of Eq. 57. Not fitted. Defined
for the note's x (A4). (This is the nonspinning coefficient; α carries no spin term because its spin
dependence measured consistent with zero, `spin_extension_plan.md` sec 7.)

### A4. The note's x: x = x_pp · X1^(−4/5), and XCLIP

`get_x` (F:195-200) returns `x_pp = (ω_22/2)^(2/3) = (m1 Ω)^(2/3)` on the m1 clock (phase Savitzky–Golay
smoothed over 80 m1 of time via `physics_common.savgol_window`, polyorder 3; clipped to [1e-8, 0.6]).
The note's PN variable is `x = (M Ω)^(2/3)` (Eq. 6); Eq. 8 with `β_LO = X1^(6/5)` (Eq. 12) gives
`x = X1^(−4/5) x_pp`, `physics_common.x_scale` (`physics_common.py:88-90`), applied at F:236 as
`xc = X1^(−4/5) · min(x_pp, XCLIP)`. This is the binary's x at the same phase under the leading-order map,
not the bare unit change `X1^(−2/3)` (they differ by `X1^(−2/15)`, 3.9% at q = 3, 1.6% at q = 8). Until
2026-09-26 x_pp was used directly (9–21% too small for the 55/42 slope). `XCLIP = 0.26` (F:97) is a chosen
cap on x_pp, not an anchor (0.327 to 0.286 on the note's x for q = 3 to 8).

### A5. T_ANCHOR = −100 (a convention)

`T_ANCHOR = −100.0` (F:97), ppBHPT time in m1 units. `mismatch` sets `ts = ∫β dt − [∫β dt]_{t = T_ANCHOR}`
(F:263) and `τ = t0 + ts` (F `_err_at_t0`, F:246), so `τ(T_ANCHOR) = t0`, the per-case nuisance offset
optimised with the mismatch. It fixes only what t0 means; it is not a physical anchor.

### Which fitted coefficient multiplies which anchor

| anchor | enters as | coefficients attached (source) |
|:--|:--|:--|
| A1 `X1^(6/5)` | base of `β_insp` | `b1`, `b2` (nonspinning fit); `s11, s12, s21, s22` (spin grid) |
| A1 `X1^(6/5)` | base of `α_insp` | `a2` (nonspinning fit) |
| A1 `X1^(6/5)` | base of `α_mr` | `m0`, `m1` (spin grid) |
| A2 `β_r(q, χ1)` | `β_MR = r0 · β_r` | `r0` (spin grid) |
| A3 `55/42`, A4 `X1^(−4/5)` | fixed | none |
| switch (not an anchor) | `S = sigmoid((p_loss − (p00 + p01 ν))/w0)` | `p00`, `p01`, `w0` (spin grid) |

## Coefficients

*History (2026-09-09, pre-units-fix, 9-frozen protocol; superseded -- current values in
`coeffs_spin.json` `theta9` / `theta_spin` and the 2026-09-27 sections at the end).*

```python
# FROZEN — the shipped nonspinning 9, never re-opened
b1, b2, a2 = 0.352, 8.553, -22.9193
p00, p01, w0, r0, m0, m1 = 3.2336, 2.2404, 1.9069, 0.6869, 0.7604, 3.9838

# FITTED — the 4 spin coefficients   (job 3483880, 2026-09-09)
s11, s12, s21, s22 = -0.0856, -0.2769, +13.3217, +8.1237
```

**Why the 9 are frozen.** An independent refit of them (job 3448537) reached a *better
training objective* (mean 5.044e-04 vs 6.148e-04) but a *worse model*: in-range median
7.632e-04, q=2 4.522e-03 vs 2.551e-03, with b1 -42%, b2 -32%, a2 -15%, p0_1 -264%.
They are basin-degenerate at the 10-40% level, so re-opening them would entangle the
spin answer with basin choice. The known merger-contamination of b2 (~2x the
second-order-self-force x^2 content, `wardell_ideation.md` 4h) is chi1-INDEPENDENT while
the spin terms are strictly odd/even in chi1, so over the symmetric grid they are close
to orthogonal.

## Constants and their sources

Added 2026-09-26 so that no number in this file stands without an explanation.  Code
references are to `spin_dep/fit_pn_anchored_spin.py` unless another file is named; line
numbers updated to the current code 2026-09-27; the named constant or function is the
reliable pointer.  The anchors are explained in full in "Anchors (derived, not fitted)" above.  **Units:** ppBHPT times (the input waveform, `t_bhpt`) are in units of
the PRIMARY mass m1; NR times (the target, `t_nr`) are in units of the TOTAL mass M.

**Derived from theory (not fitted)**

* **C1 — `base = X1**1.2`, i.e. X1^(6/5), with X1 = q/(1+q).**  The leading-order value
  of both beta and alpha.  Derivation: match the Newtonian chirp rate
  dx/dt = (64/5) eta x^5 on the two sides under the time map tau = beta*t (x scales as
  beta^(-2/3)); this gives eta_pp * beta^(5/3) = nu, so beta_LO = (nu/eta_pp)^(3/5) =
  X1^(6/5), where eta_pp = m2/m1 = 1/q and nu/eta_pp = X1^2.  Matching the Newtonian
  (2,2) amplitude as well gives alpha_LO = X1^(6/5) too.  Source:
  `../alpha_beta_pn_scaling_note_revised.pdf`, Eqs. (7)-(12) for beta, (13)-(16) for alpha.
* **C2 — `55/42` (`PN_SLOPE`, line 97).**  The mass-ratio part of the 1PN coefficient of
  the (2,2) amplitude: h22 = h22_Newtonian * [1 + (-107/42 + (55/42) nu) x + 2 pi x^(3/2)
  + ...] (note Eq. 55).  The -107/42 part is the same for a test particle, so in the ratio
  h_NR/h_pp at the same x only (55/42) nu x survives (note Eq. 56); this is the 1PN term
  of alpha_insp in note Eq. (57).  Fixed, not fitted.
* **C3 — `x`.**  The model uses the note's PN frequency variable x = (M*Omega)^(2/3)
  (note Eq. 6), on the TOTAL-mass clock.  What the ppBHPT phase gives directly (`get_x`) is
  x_pp = (m1*omega_22/2)^(2/3), on the primary-mass clock, because the ppBHPT time axis is in
  m1 units.  The two are related by the same leading-order time map as C1: with
  beta_LO = X1^(6/5), x = x_pp * X1^(-4/5) (note Eqs. 8, 12; `../physics_common.x_scale`).
  The fixed 1PN slope 55/42 (C2) is defined for this x.  Until 2026-09-26 the code used x_pp
  directly, i.e. an x 9-21% too small (audit C1).  Numerical settings (chosen): phase
  Savitzky-Golay smoothed over 80 m1 in time (polyorder 3; was a fixed 401 samples = 40 m1
  on this surrogate), x_pp clipped to [1e-8, 0.6], and XCLIP applied to x_pp.
* **C4 — `nu = q/(1+q)**2`, `X1 = q/(1+q)`.**  Symmetric mass ratio and primary mass
  fraction.  Over the training range q in [3, 8]: nu from 0.1875 down to 0.0988, X1 from
  0.750 up to 0.889 (these are the numbers behind "nu falls 1.9x, X1 rises 1.18x" below).

**External fits and constants (published, not fitted here)**

* **C5 — `w220(chi)`, the Kerr (2,2,0) ringdown frequency.**  Real part of the l = m = 2,
  n = 0 quasinormal-mode frequency, M*omega_220, in units of the hole's own mass, EXACT:
  Leaver's continued-fraction solution of the Teukolsky equation via the `qnm` package
  (L. C. Stein 2019, JOSS 4, 1683; `../physics_common.w220_exact`).  For chi < 0 (spin
  anti-aligned with the orbit) the (2,2) mode is the counter-rotating one,
  |omega_{2,-2}(|chi|)|.  w220(0) = 0.37367.  Replaced on 2026-09-26 the Berti-Cardoso-Will
  fit f1 + f2*(1-chi)^f3 (Phys. Rev. D 73, 064030, Table VIII; f1 = 1.5251, f2 = -1.1568,
  f3 = 0.1292), which is 1.5% low at chi = 0, was fitted for chi >= 0 only, and is 3% / 7% low
  when extrapolated to chi = -0.2 / -0.6.  Checked against the measured ppBHPT ringdown
  frequency at q = 5: chi1 = -0.2 / 0 / 0.2 / 0.6 -> +0.09 / +0.26 / -0.67 / +0.18 %.
* **C6 — `beta_r(q, chi1) = w220(chi1) * M_f / w220(chi_f)`, the merger-ringdown anchor.**
  Late in the ringdown each waveform rings at its own QNM frequency, so the time map must
  stretch ppBHPT time by exactly their ratio.  pp side: in ppBHPT the ringing hole IS the
  primary (spin chi1; the secondary is negligible), so its frequency is w220(chi1) in units
  of m1 -- the ppBHPT clock, no mass conversion.  NR side: the real remnant, w220(chi_f)/M_f
  in units of M -- the NR clock.  Zero fitted parameters; the fitted `r0` only scales it.
  At chi1 = 0 it reduces to the nonspinning pn_anchored expression.  History: until 09-24 the
  pp side carried a spurious (1+q)/q (17-29% too large, UNITS_AUDIT.md A1); until 09-26 it
  was 0.3737*w220_BCW(chi1)/w220_BCW(0) (`W_SCHW`, `RESCALE_QNM`, now unused) against the raw
  BCW fit on the NR side -- two sources on the two sides of one ratio (audit C10).
* **C7 — `M_f, chi_f` from NRSur3dq8Remnant.**  Remnant mass (in units of M) and spin of
  the real binary, from the surfinBH package (`surfinBH.LoadFits("NRSur3dq8Remnant")`,
  line 60), evaluated at (q, [0,0,chi1], [0,0,0]).  NRSur3dq8Remnant is Varma et al. 2019,
  Phys. Rev. Lett. 122, 011101 (arXiv:1809.09125).  This is the model's only NR-based input.
* **C8 — waveforms.**  Input: BHPTNRSur2dq1e3 (2,2) mode with `calibrated=False` (the raw
  ppBHPT waveform, time in m1 units).  Target: NRHybSur3dq8 (2,2) mode at
  (q, [0,0,chi1], [0,0,0]) with `dt = 0.1` and `f_low = 5e-3` (`F_LOW`, line 140), both
  in `build_case`.

**Chosen settings (picked by hand, not fitted, not derived)**

* **C9 — `XCLIP = 0.26` (line 97).**  Upper cap on x_pp before it is converted to the note's x and
  enters the PN polynomials (`xc = X1^(-4/5) min(x_pp, 0.26)`, line 236), so the x and x^2 terms stop growing through merger.  Inherited
  from nonspinning pn_anchored (`../fit_scaling_pn_anchored.py:72`, stored as `x_clip` in
  `../pn_anchored_results/coeffs.json`).
* **C10 — `T_ANCHOR = -100` (line 97).**  The ppBHPT time (m1 units) at which the running
  integral of beta is set to zero, i.e. tau(T_ANCHOR) = t0.  Same value as
  `../fit_scaling_PN_opt_creative.py:24`.  It only fixes where t0 is measured from.
* **C11 — `T_REF = -100` (`../fit_scaling_PN_opt_creative.py:23`), the p_loss
  normalisation.**  In `../fit_scaling_wf_nu_q_dep.wf_loss_coordinates`: e_hat =
  (E(t) - E(t_peak)) / |E(T_REF) - E(t_peak)|, and the same for j_hat, where E and J are the
  cumulative radiated energy and angular momentum of the ppBHPT (2,2) mode and t_peak is
  the |h| peak (ppBHPT time, m1 units).  So p_loss = -1 at t = T_REF, 0 at the peak, and
  positive after it; the switch centre `p00 + p01*nu` (about 3.5 to 3.7 on this grid) is
  therefore well past the peak.
* **C12 — scored NR window `[-5000.1, +100.0]` (NR time, M units).**  `NR_T_START`,
  `NR_T_END` in `../fit_scaling_PN_opt_creative.py:21-22`; the NR waveform is cut to this
  span in `build_case`.
* **C13 — `MIN_COVERAGE = 0.88` (`../fit_scaling_PN_opt_creative_q_dep.py:34`).**  If the
  mapped ppBHPT waveform covers less than 88% of the NR window's samples, the case scores
  a penalty of 50.0 instead of a mismatch (`_err_at_t0`).  A non-monotone time map scores
  50 + 1000*(size of the backwards step) (`mismatch`).
* **C14 — t0 search.**  t0 is a per-case time shift (a nuisance, not a coefficient):
  81-point scan over seed +- 200 M, then a bounded 1-D minimisation within +- 6 M of the
  best scan point (`mismatch`, lines 267-273); first seed -50 (`full_mm`).
* **C15 — `SRC_STRIDE = 6`, `NR_STRIDE = 10` (line 140).**  Every 6th ppBHPT and every
  10th NR sample are used to find t0 quickly; the mismatch is then recomputed at FULL
  resolution at that t0 (`full_mm`).  All mismatches quoted here are full resolution.
* **C16 — grid.**  `TRAIN_Q` (lines 125-126): 12 values q = 3.0, 3.2564, 3.5128, 3.8974, 4.2,
  4.6, 5.0, 5.4359, 5.9487, 6.5897, 7.2308, 8.0 (the nonspinning fit's grid);
  `CHIS` (line 127): chi1 = -0.6 ... +0.8 in steps of 0.2.  12 x 8 = 96.  chi1 below -0.6
  is left out on purpose (retrograde ringdown regime, see the FREEZE_XC note, lines 99-118).
* **C17 — optimiser.**  Powell (lines 465-466), xtol 1e-4, ftol 1e-6, `--maxiter` default 60,
  `--budget` default 95 minutes (line 341).  Since 2026-09-27 `--refit` fits 10 parameters,
  p00, p01, w0, r0, m0, m1 (`FREE9`, line 153) + the 4 spin terms, started from the installed
  values (line 453).  Historical starting point for the spin terms, `SEED = [1.107, 2.882,
  0, 0]` (line 139): s11 and s12 taken from the measured-ratio fit's a2 = +1.107 and
  a4 = +2.882 (`spin_extension_plan.md` sec 8), s21 = s22 = 0.  The fit minimises the MEAN
  mismatch over the grid.
* Small numerical guards: the switch argument is clipped to +-60 inside `sigmoid`
  (`../fit_scaling_PN_opt_creative.py:88`) and w0 is floored at 1e-3 (`model_ab`).

**Fitted coefficients of this model**

* **Current (2026-09-27, `coeffs_spin.json`):** `theta9` = b1 0.5298, b2 7.3741, a2 -20.1478
  (nonspinning pn_anchored), p00 2.0952, p01 6.8474, w0 1.9222, r0 0.7987, m0 0.9634, m1 2.2968
  (fitted on the spin grid); `theta_spin` = s11 0.0524, s12 -0.0614, s21 10.0201, s22 1.6660.
  What each multiplies: see the table at the end of "Anchors (derived, not fitted)".
* *History (superseded 2026-09-27):* **The frozen 9** (`b1, b2, a2, p00, p01, w0, r0, m0, m1`) were the nonspinning
  pn_anchored fit, read from `../pn_anchored_results/coeffs.json` `theta` and copied into
  `pn_anchored_spin_results/coeffs_spin.json` `theta9_frozen`.  What each multiplies:
  b1 -> nu*xc and b2 -> nu*xc^2 in beta_insp; a2 -> nu*xc^2 in alpha_insp; p00, p01 ->
  switch centre p00 + p01*nu (in p_loss units, C11); w0 -> switch width (p_loss units);
  r0 -> scale on beta_r in beta_mr; m0, m1 -> alpha_mr = base*(m0 + m1*nu).
* **The 4 spin coefficients**, fitted here (`coeffs_spin.json` `theta_spin`, job 3483880,
  log `spinnu.3483880.log`): s11 -> chi1*nu*xc, s12 -> chi1^2*nu*xc, s21 -> chi1*nu*xc^2,
  s22 -> chi1^2*nu*xc^2 (all inside beta_insp).  Note: job 3483880 ran
  `fit_pn_anchored_spin_eps.py --weight nu`, which the json records as algebraically
  identical to this script.
* **After the 2026-09-24 units correction** the coefficients in use are in
  `redo_properly_results/spin13.json`: `frozen9_corrected` (the 9, from the corrected
  nonspinning fit) and `steps["2"]["best"]["theta_spin"]` (the 4 spin terms).

**Where the result numbers come from**

* Every mismatch is the relative L2 difference (n1 + n2 - 2|Z|)/(2 n1) between the NR
  (2,2) mode and the mapped, rescaled ppBHPT mode over the window C12, with the overall
  phase solved exactly and t0 optimised per case (C14), at full resolution (C15).
* "median" and "max" = over all 96 grid cases; a column value = over the 12 q at that
  chi1.  The fit itself minimised the MEAN (C17); median is what is reported.
* Pre-correction body: per-case values, median, max in
  `pn_anchored_spin_results/coeffs_spin.json` (identical copy:
  `pn_anchored_spin_eps_results/coeffs_nu_50min.json`).  The eps = 1/q arm:
  `pn_anchored_spin_eps_results/coeffs_eps_50min.json`.  The 2026-08-31 run (median
  7.863e-04, drive quarantine ON): `coeffs_spin_20260831_quarantine.json`.  The split
  variant: `coeffs_spin_split.json`.  The step-1 R_alpha / R_beta / exponent-k numbers are
  measurements from `spin_measure_alpha_beta.py` (`spin_measure_results/`,
  `spin_extension_plan.md` sec 7-8), not fits of this model.
* Correction section at the end: `redo_properly_results/spin13.json` and log
  `spin13.3535558.log` (6 starts).

**Numbers in this file that do not match the current json/log (left unchanged, flagged
2026-09-26)**

* Opening paragraph "4.86e-04" (chi1 = -0.2), "1.28e-03" (chi1 = -0.6), "3.50e-03"
  (chi1 = +0.8), and the q-structure paragraph "1.09e-02 (q=3) -> 2.40e-03 (q=5) ->
  7.43e-03 (q=8)" are from the 2026-08-31 quarantine run
  (`coeffs_spin_20260831_quarantine.json`: 4.856e-04, 1.280e-03, 3.496e-03; 1.091e-02,
  2.400e-03, 7.428e-03), not from the shipped run in the tables (4.8125e-04, 1.3218e-03,
  3.4447e-03; 9.64e-03, 2.53e-03, 7.03e-03).
* eps section "9.44e-3 vs 5.57e-3 at q=8": the eps value matches `coeffs_eps_50min.json`;
  the nu value at q = 8, chi1 = +0.8 is 7.03e-03 in `coeffs_nu_50min.json` and in the full
  grid above.
* Caveat "coeffs_spin.json was written by the quarantine run": the current
  `coeffs_spin.json` records `freeze_xc: false` and job 3483880 (quarantine OFF); the
  quarantine run's file is `coeffs_spin_20260831_quarantine.json`.
* The "hardcoded Schwarzschild 0.3683" in "Why beta_r is derived" is the BCW fit's value at
  chi = 0 (C5), not the exact Schwarzschild value, which is 0.3737 (C6).

## Results

Grid: 12 even-nu q in [3, 8] x 8 chi1 in [-0.6, +0.8] = 96 points. Objective: waveform
mismatch (relative L2, analytic phi0, t0 a per-case nuisance).

**ALL 96: median 7.6858e-04, max 9.6416e-03.**

| chi1 | median | max | vs chi1=0 |
|---:|---:|---:|---:|
| -0.6 | 1.3218e-03 | 2.2585e-03 | 2.38x |
| -0.4 | 6.1146e-04 | 1.0991e-03 | 1.10x |
| -0.2 | 4.8125e-04 | 7.5425e-04 | 0.87x |
| +0.0 | 5.5425e-04 | 6.6171e-04 | 1.00x | (null check)
| +0.2 | 5.8838e-04 | 6.6897e-04 | 1.06x |
| +0.4 | 9.2897e-04 | 1.1120e-03 | 1.68x |
| +0.6 | 1.1721e-03 | 2.4904e-03 | 2.11x |
| +0.8 | 3.4447e-03 | 9.6416e-03 | 6.22x |

Full grid:

| q | -0.6 | -0.4 | -0.2 | +0.0 | +0.2 | +0.4 | +0.6 | +0.8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.000 | 1.86e-03 | 7.72e-04 | 6.50e-04 | 6.62e-04 | 6.36e-04 | 1.09e-03 | 2.49e-03 | 9.64e-03 |
| 3.256 | 1.63e-03 | 6.90e-04 | 5.69e-04 | 6.51e-04 | 5.44e-04 | 8.31e-04 | 1.62e-03 | 6.27e-03 |
| 3.513 | 1.46e-03 | 6.38e-04 | 5.14e-04 | 6.58e-04 | 5.31e-04 | 7.35e-04 | 1.19e-03 | 4.44e-03 |
| 3.897 | 1.27e-03 | 5.85e-04 | 4.62e-04 | 6.50e-04 | 5.72e-04 | 7.21e-04 | 9.22e-04 | 3.11e-03 |
| 4.200 | 1.16e-03 | 5.60e-04 | 4.36e-04 | 6.30e-04 | 6.14e-04 | 7.65e-04 | 8.67e-04 | 2.67e-03 |
| 4.600 | 1.07e-03 | 5.33e-04 | 4.16e-04 | 5.87e-04 | 6.55e-04 | 8.31e-04 | 8.85e-04 | 2.49e-03 |
| 5.000 | 1.02e-03 | 5.21e-04 | 4.07e-04 | 5.22e-04 | 6.69e-04 | 9.02e-04 | 9.50e-04 | 2.53e-03 |
| 5.436 | 1.04e-03 | 5.28e-04 | 4.12e-04 | 4.43e-04 | 6.53e-04 | 9.56e-04 | 1.04e-03 | 2.74e-03 |
| 5.949 | 1.14e-03 | 5.67e-04 | 4.38e-04 | 3.55e-04 | 6.05e-04 | 1.00e-03 | 1.16e-03 | 3.10e-03 |
| 6.590 | 1.37e-03 | 6.68e-04 | 5.01e-04 | 2.67e-04 | 5.22e-04 | 1.04e-03 | 1.30e-03 | 3.78e-03 |
| 7.231 | 1.72e-03 | 8.31e-04 | 5.98e-04 | 2.10e-04 | 4.40e-04 | 1.07e-03 | 1.44e-03 | 4.86e-03 |
| 8.000 | 2.26e-03 | 1.10e-03 | 7.54e-04 | 1.97e-04 | 3.79e-04 | 1.11e-03 | 1.59e-03 | 7.03e-03 |

**The q-structure is the diagnostic.** The chi1 = 0 column improves MONOTONICALLY with q
(6.62e-04 -> 1.97e-04, a factor 3.4). Every spinning column is U-shaped in q with its
minimum near q ~ 4.6-5.0: at chi1 = +0.8 it runs 1.09e-02 (q=3) -> 2.40e-03 (q=5) ->
7.43e-03 (q=8), a 4.5x U. Since the only difference between the chi1 = 0 column and the
others is the spin term, **the mass-ratio weighting of the spin correction is wrong at
both ends of the q range.**

That weighting was never measured. The spin terms sit INSIDE the nonspinning drift's
`nu*(...)` bracket and so inherited its `nu` prefactor by construction. The natural PN
spin variable for a non-spinning secondary is chi_eff = X1*chi1, and over q in [3,8] the
two prefactors move in OPPOSITE directions: nu falls 1.9x (0.188 -> 0.099) while X1
rises 1.18x (0.750 -> 0.889). A 2.2x relative error across the training range is the
right size to produce the observed U.

## Provenance

| job | date | what | outcome |
|:---|:---|:---|:---|
| 3451839 / 3451846 | 08-30 | step-1 parameter-free measurement, 30 points | alpha: no signal; beta: real, 2-3.7x Islam on prograde |
| 3452029, 3452038, 3452054 | 08-30 | first fit attempts | cancelled — wrong seed mapping, then the `beta_r` bug (below) |
| 3452082 | 08-30/31 | the fit, 197 evals, stopped on budget | theta = (-0.7440, +0.4149, +25.1318, -3.6317), median 9.387e-04 |
| 3453596 | 08-31 | A/B on the drive + resume, 197 more evals | median 7.863e-04, quarantine ON |
| 3483880 | 09-09 | rerun, quarantine OFF, seeded from the above, 105 evals | **median 7.686e-04** — the shipped coefficients |

Neither fit converged; both stopped on a 95-minute budget. The coefficients moved
enormously between them (s11 -0.744 -> +0.009, s21 25.1 -> 11.7, s22 -3.6 -> +5.9) at
similar objective, so **the 4 spin coefficients are themselves basin-degenerate**, the
same pathology the nonspinning 9 show.

## Why alpha has no spin terms

Measured, not assumed. Step 1 read alpha and beta off the two waveforms parameter-free
(paired by equal accumulated phase back from merger; `alpha = |h_NR|/|h_pp|`,
`beta = omega_pp/omega_NR`) and formed `R_alpha(x;q,chi1) = alpha(x;q,chi1)/alpha(x;q,0)`.

| q | chi1 | mean R_alpha | x-spread | Islam predicts | (meas-1)/(Islam-1) |
|---:|---:|---:|---:|---:|---:|
| 3 | +0.80 | 0.99341 | 2.38% | 1.02227 | **-0.30** |
| 5 | +0.80 | 0.99694 | 1.04% | 1.01290 | -0.24 |
| 8 | +0.80 | 1.00024 | 1.53% | 1.00790 | 0.03 |
| 3 | +0.60 | 0.99545 | 1.13% | 1.01345 | -0.34 |
| 8 | +0.60 | 1.00229 | 0.88% | 1.00477 | 0.48 |

Two reasons this became "no spin terms":
1. **The signal is below the scatter.** |R_alpha - 1| <= 0.7% at all 25 points, while the
   x-spread WITHIN a single case is 0.66-2.4%. It is a non-detection at ~1% resolution.
2. **Where Islam predicts the most, we measure the opposite sign** — at chi1 = +0.8 their
   form wants +1.3% to +2.2%; we measure -0.66% to +0.02%.

By contrast R_beta is unambiguous (up to 1.0235 at q=3/chi1=+0.8, ordered consistently
across all q and both signs). Since alpha and beta are degenerate through the time map,
giving alpha spurious spin freedom would let it absorb beta's real effect.

There is also a physical reason to expect this split: alpha is the amplitude ratio at
FIXED x, where the quadrupole formula dominates and spin-orbit enters only weakly at
1.5PN; beta is a RATE, tied to the flux, where spin-orbit enters at 1.5PN with a large
coefficient. Spin changes how fast you sweep through x far more than how loud you are at
a given x.

**Caveat, and it is a live one:** the measurement covers x in [0.06, 0.12], the INSPIRAL
ONLY. alpha's spin dependence through merger and ringdown was never measured, and
`alpha_mr = base*(m0 + m1*nu)` is spin-blind by fiat at exactly the point where the
remnant differs most between chi1 = -0.6 and +0.8. chi1 = +0.8 is simultaneously the
noisiest measurement, the largest disagreement with Islam, and the worst fitted column.

## Why beta_r is derived, not fitted

`beta_r` is the ratio of the two sides' ringdown frequencies, and BOTH sides move with
spin. In ppBHPT the "remnant" IS the primary, of mass m1 and spin chi1 (the secondary is
a test particle), so its QNM is `w_pp(chi1)` in the primary-mass units the ppBHPT data already uses (no
`(1+q)/q` factor -- that was the 2026-09-24 units bug, UNITS_AUDIT.md A1); the NR
side is `w220(chi_f)/M_f` with M_f, chi_f from NRSur3dq8Remnant. w220 is exact Kerr (since
2026-09-26; before, the BCW fit) and the remnant fit already takes spin, so the spin dependence
of the ringdown anchor costs ZERO parameters (full statement: "Anchors (derived, not fitted)" A2).

**A bug here was caught before it reached a result** (`diag_beta_r.py`, 08-30). The first
version gave spin to the NR side only, leaving the pp side at the hardcoded Schwarzschild
0.3683. At q=3, chi1=+0.8 that drove `beta_r*r0` to 0.498 against `beta_insp = 0.708` — a
30% discontinuity between the branches, in the wrong direction. Corrected it is 0.791,
straddling the inspiral value. At chi1 = 0 both forms agree exactly, so no nonspinning
model was ever affected.

## The retrograde x artefact — real, and NOT the problem

At chi1 = -0.6 the alpha/beta overlays spike violently for t in (0, +80) M at every q.
Chased to ground (08-31):

* the post-merger `|h|` envelope carries 15-16 genuine local minima at chi1 = -0.6, 1-5
  at -0.2, and **ZERO at chi1 > 0** (where it decays monotonically);
* each null wraps the phase, so `x = (omega/2)^(2/3)` spikes to 1.4-1.8x the pp-side QNM
  frequency — physically impossible;
* `XCLIP = 0.26` caps the spikes but not the dips, so `xc` oscillates;
* the spikes reach alpha and beta only because **S saturates at 0.18-0.19**, so the
  x-driven inspiral branch still supplies ~83% of both throughout the ringdown. This is
  pn_anchored's known weak-switch caveat (07-28) costing something for the first time.

Rink et al. document the cause: retrograde ringdown modes at chi1 <~ -0.6 defeat a global
surrogate fit, so they partition the time domain at merger, `[-13500,0] U [0,125]`, and
train the subdomains separately. The modulation switches on at t = 0 (first null at
+11.4/+11.6/+11.9 M for q = 3/5/8) and its frequency is q-INDEPENDENT (omega_beat = 0.806
at every q), which no QNM can be — consistent with a separately-trained subdomain's own
reconstruction.

**But fixing it does not help.** Applying their H(t) to our drive (freeze x at its merger
value for t > 0) was A/B-tested at fixed coefficients, job 3453596:

| chi1 | -0.6 | -0.4 | -0.2 | 0 | +0.2 | +0.4 | +0.6 | +0.8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gain | **0.97x** | **0.97x** | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

Slightly WORSE on the two retrograde columns, exactly neutral elsewhere. The spikes sit
where `|h|` is 5e-5..3e-4 of peak and therefore carry no weight in an L2 mismatch: **they
are cosmetic.** Note `XCLIP` already implements the quarantine for chi1 >= 0 by accident
— x at merger there (0.27-0.33) sits above the clip — which is why the fix is a no-op on
that half.

## What was tried and did NOT pay: the chi1 domain split

Rink et al.'s other strategy is two separate models either side of chi1 = 0. Implemented
as an 8-coefficient variant (`fit_pn_anchored_spin_split.py`, jobs 3458360/3458363):
**median 7.245e-04, max 1.015e-02 — 8% and 7% better for double the coefficients**, and
two columns (chi1 = -0.4, +0.6) got WORSE. The U-shape survived intact.

The negative result is solid, not a budget artefact: the chi1 < 0 half CONVERGED (292
evals, "terminated successfully") and still left chi1 = -0.6 at 1.21e-03. And the two
halves declined to diverge much when given the freedom (s21 = 9.60 vs 11.98), so the
U-shape was never "one quadratic spanning two regimes".

## The mass-ratio weighting of the spin term — measured, then TESTED AND REJECTED

The spin terms sit inside the nonspinning drift's `nu*(...)` bracket, so they inherited a
`nu` prefactor by construction -- never chosen, never tested.  That is the natural suspect
for the q-U-shape, since the chi1 = 0 column (which has no spin term) falls monotonically
while every spinning column turns up at both ends.

**Measured** from the parameter-free step-1 grid, fitting `|R_beta - 1| ~ V^k` across
q = 3..8 at fixed chi1, over x <= 0.09 (above that the 0.08-0.11 envelope-beating band
makes the exponents unusable -- they scatter -1.39 to +2.30):

| V | k | \|k-1\| |
|:---|---:|---:|
| nu = X1*X2  (the current choice) | 1.45 +- 0.37 | 0.45 |
| X2 = 1/(1+q) | 1.15 +- 0.29 | 0.15 |
| **eps = 1/q** | **0.95 +- 0.24** | **0.05** |
| nu*X1  (chi_eff) | 1.94 +- 0.49 | 0.94 |

Decline over q = 3 -> 8: measured 2.626 +- 0.633 against 2.667 for 1/q, 1.898 for nu.
So the measurement points at **eps = 1/q, an INTEGER power of the self-force expansion
parameter** -- not a fitted exponent.  It also rules out chi_eff = X1*chi1, the obvious
PN guess, which needs k = 1.94 and moves the weighting the WRONG WAY (X1 rises with q).

**But the calibration disagrees.**  Like-for-like, same script, same seed, same 50-minute
budget, ~105 evals each (jobs 3482648 / 3483880):

| W | median | max |
|:---|---:|---:|
| **nu** | **7.6858e-04** | 9.6416e-03 |
| eps = 1/q | 8.1511e-04 | 1.1128e-02 |

and the split by column is not uniform -- eps is BETTER across the whole retrograde side
and at chi1 = +0.4, and loses only at chi1 = +0.8 (4.13e-03 vs 3.44e-03), which carries
the max and pulls the median:

| chi1 | -0.6 | -0.4 | -0.2 | 0 | +0.2 | +0.4 | +0.6 | +0.8 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| eps | **1.12e-3** | **5.62e-4** | **4.58e-4** | 5.54e-4 | 6.13e-4 | **8.89e-4** | 1.18e-3 | 4.13e-3 |
| nu | 1.32e-3 | 6.11e-4 | 4.81e-4 | 5.54e-4 | **5.88e-4** | 9.29e-4 | **1.17e-3** | **3.44e-3** |

**The q-U-shape did NOT flatten under eps** -- at chi1 = +0.8 it still minimises near
q ~ 4.6 and the high-q end got worse (9.44e-3 vs 5.57e-3 at q=8).  So the weighting was
not the cause, and `nu` stays.

The disagreement is itself the finding: the parameter-free measurement is INSPIRAL ONLY
(x <= 0.09) and says eps; the full-window mismatch is merger-weighted and says nu, purely
because of the strong-prograde merger.  Neither settles the question, because they are
answering about different regions -- the same inspiral-vs-full-window split that recurs
throughout this workspace.  That a single global weight cannot serve both ends of the
chi1 range is a hint that something else varies with chi1.

**A note on the two U-shapes, which are not the same thing.**  The U in *chi1* at fixed q
is EXPECTED and should stay: at chi1 = 0 every spin term vanishes and the model reduces to
the frozen nonspinning one, so error must grow in both directions as the four coefficients
take on more work.  It is also roughly symmetric in |chi1| (at 0.4: 6.11e-4 retrograde vs
9.29e-4 prograde; at 0.6: 1.32e-3 vs 1.17e-3).  The anomaly is the U in *q* at fixed chi1,
because the chi1 = 0 column falls monotonically (6.62e-4 -> 1.97e-4) while every spinning
column does not.  Only the latter is a defect.

## Caveats

* **Neither fit converged.** Both stopped on a 95-minute budget, and the coefficients are
  basin-degenerate — treat the 4 numbers as one point in a valley, not as measurements.
* **The chi1 = 0 column reads 5.543e-04, not the shipped 6.148e-04**, because the SOURCE
  changed: BHPTNRSur2dq1e3 at zero spin is a different waveform from BHPTNRSur1dq1e4.
  The model is algebraically identical there. This is not a bug.
* **`coeffs_spin.json` was written by the quarantine run**, so its coefficients assume
  `--freeze-xc` — a <=3% inconsistency on two retrograde columns and exactly zero
  elsewhere. Pre-quarantine values `(-0.7440, +0.4149, +25.1318, -3.6317)` are in
  `history.md` and the 3452082 log.
* **q < 3 is a DOUBLE extrapolation under spin.** BHPTNRSur2dq1e3's own domain starts at
  q = 3 (BHPTNRSur1dq1e4 reached 2.5), so the SOURCE waveform is extrapolated too — it
  only warns. At q = 2: 1.45e-02 (chi1=-0.6), 3.75e-03 (-0.2), 4.55e-03 (+0.2),
  2.73e-02 (+0.6), an order of magnitude worse than q = 3 where the nonspinning model
  degraded only ~4x. The good q<3 behaviour of nonspinning pn_anchored does NOT carry
  over.
* **chi1 = +0.8 is at the edge of BHPTNRSur2dq1e3's training range** (|chi| <= 0.8), which
  is a candidate cause of that column on its own.
* Only the (2,2) mode. Only aligned primary spin; the secondary is a non-spinning test
  particle by construction.

## Not done

1. ~~Measure the mass-ratio weighting~~ — DONE 2026-09-09, and rejected: the
   measurement says `eps = 1/q` (k = 0.95 +- 0.24) but the calibration prefers `nu`, and
   eps does NOT flatten the q-U-shape. See the section above. The q-U-shape remains the
   largest structure in the residuals and is now UNEXPLAINED — the drive quarantine, the
   chi1 domain split, chi_eff and eps have all been tested and none of them is the cause.
2. **Test alpha spin terms near merger.** The only region where alpha's spin-blindness
   was never checked, and it competes with (1) as an explanation for chi1 = +0.8.
3. **The five frozen merger/ringdown coefficients** (p00, p01, w0, r0, m0, m1) are all
   nu-only. `beta_r` got derived spin dependence; the switch POSITION, its WIDTH, and both
   alpha merger constants did not.
4. **Run the optimiser to convergence**, once (1)-(3) settle the form. Re-tuning 4
   coefficients inside a form known to be mis-weighted is not worth the node-hours.
5. Higher modes; a spin-aware `p_loss`; chi1 < -0.6 (retrograde QNM regime, deliberately
   excluded).

---

## 2026-09-24 units correction

**Root cause (all models):** the ppBHPT waveform's time axis is in units of the PRIMARY
mass (`calibrated=False` returns the raw training grid); the NR waveform's is in TOTAL
mass.  Anything crossing between them needs an `X1 = q/(1+q)` conversion.
Mechanical check: `python units_selftest.py --q 3 5 8` (all pass).

**What affected this model:** the `beta_r` `(1+q)/q` error (see scaling_pn_anchored.md),
plus its 9 frozen coefficients come from the nonspinning fit, which also changed.

**Baseline reproduced exactly** (median 7.6858e-04, worst per-case ratio 1.00002).

**Corrected fit:** 9 frozen from the corrected nonspinning multistart (round 2), 4 spin
terms fitted, both corrections on, 6 starts all within 0.01% (converged):
median **1.0993e-03 (+43.0%)**, mean 2.0620e-03, max 2.0286e-02.

Column medians (x1e-3, q in [3,8]):
chi1 -0.6 1.41 | -0.4 0.72 | -0.2 0.53 | 0.0 0.66 | +0.2 0.79 | +0.4 1.47 | +0.6 2.58 | **+0.8 6.64**
q=2 row: median 5.15e-03, worst 0.216 at chi1=+0.8.

**Diagnosis:** the chi1=0 column (0.66e-03) is BETTER than the published overall median, so
the frozen base is fine.  The damage is at large |chi1|, dominated by chi1=+0.8.  Small-grid
tests (18 points): a cubic chi1 term does NOT fix it; freeing the merger amplitudes does NOT;
freeing the SWITCH (p00,p01,w0) cuts the chi1=+0.8 column 2.4x but breaks chi1=0.  One
spin-blind switch cannot serve every spin.  Candidate (parked for discussion):
p0 = p00 + p01*nu + p02*chi1.  Details: scaling_junction_spin.md sec 15.
Most of the +43% is inherited from the nonspinning base (+21.7%); the spin fit adds ~17%.

**Do not quote** the step-1 figure (+168.8%): it scored frozen coefficients against a
constant they were not fitted for.

**Status: open.**  Coefficients: `redo_properly_results/spin13.json`.  Plots REDRAWN on the
corrected coefficients: `Agentic_spin_plots/pn_anchored_spin/` (28 figures, via
`plot_pn_family_corrected.py`); pre-correction in `pn_anchored_spin_PREFIX_BACKUP_20260924/`.


## 2026-09-26 fix round -- coefficients of that date (superseded 2026-09-27 by the switch + merger fit below)

Refit on the fixed code (x on the note's clock, exact Kerr QNMs including the counter-rotating branch for chi1 < 0 -- the BCW fit used before was 3-7% low there -- and the new nonspinning 9).  The 9 nonspinning coefficients are frozen at the new
`pn_anchored_results/coeffs.json`; the 4 spin coefficients were refit from the previous values
(job 3540979, bounded budget).  Stored in `coeffs_spin.json`, stamped `model_form`.
Full-resolution mathcalE over the 96-point grid (12 q x 8 chi1):

| | published | 2026-09-26 |
|:--|--:|--:|
| mean | 1.2808e-03 | 1.6788e-03 |
| median | 7.6858e-04 | 8.9320e-04 |
| max | 9.6416e-03 | 1.7054e-02 |

**Worse than published (+31% mean, +16% median).**  It stopped on its 95-minute budget, but had
nearly stalled (best mean 1.747e-3 -> 1.681e-3 over the last 30 minutes), so this looks structural
rather than unconverged: with the new nonspinning 9 frozen (their switch
leaks more into the inspiral), 4 spin terms cannot absorb the difference, worst at chi1 = +0.8.
Next step (needs compute): refit with the 9 also free.  Plots in `Agentic_spin_plots/pn_anchored_spin/`
predate this refit.  Backup of the previous files: `SPIN_PN_RESULTS_pre20260926v2.tar.gz`.


## 2026-09-27 test -- switch freed (Lonestar6, job 3473309; test only, nothing installed)

`test_spin_free_switch.py`: p00, p01, w0 freed along with the 4 spin terms (7 free); b1, b2, a2,
r0, m0, m1 stay at the nonspinning `pn_anchored_results/coeffs.json`.  Control arm = the 4 spin
terms only, same subset, same start (current `coeffs_spin.json`, start reproduced to ratio
1.00000).  Subset = sec 15 of scaling_junction_spin.md: q 3, 5, 8 x chi1 -0.6, -0.2, 0, +0.4,
+0.6, +0.8 (18 points).  Both CONVERGED (Powell; 1594 / 479 evals).  Results in
`test_free_switch_results/{switch,spin}.json`.

| arm | free | mean | median | max |
|:--|--:|--:|--:|--:|
| start (stored) | -- | 2.7335e-03 | 1.2315e-03 | 1.7054e-02 |
| control | 4 | 2.5604e-03 | 1.2394e-03 | 1.1870e-02 |
| switch freed | 7 | 2.3224e-03 | 1.2312e-03 | 1.0854e-02 |

Against the control: mean -9.3%, median -0.7%, max -8.6%.  The switch barely moved (p00 0.889 ->
0.935, p01 2.861 -> 3.843, w0 1.705 -> 1.713) -- nowhere near the late switch the physical fit
chose (p00 4.08).  Gains are at the large cells (chi1 = +0.8 at q = 3: 1.14e-2 -> 9.01e-3;
chi1 = -0.6 at q = 3); the cost is the chi1 = 0 column, which the control leaves untouched
(q = 3: 4.52e-4 -> 6.68e-4, q = 5: 4.45e-4 -> 7.74e-4, +48% / +74%), and chi1 = +0.4 (q = 8
+60%).  Same trade as sec 15 on the pre-fix code: one spin-blind switch cannot serve every chi1.
With the switch freed the model no longer reduces to nonspinning pn_anchored at chi1 = 0.

**Late-switch start (job 3473669).**  Same switch-freed arm, started at the physical fit's switch
(p00 4.0806, p01 11.1030, w0 2.3957) with r0, m0, m1 still frozen: the start scores mean 2.24e-2,
8x worse than the stored coefficients, and Powell returns to the SAME early-switch point (p00
0.935, p01 3.847, w0 1.713; mean 2.3224e-03, median 1.2304e-03).  So the early switch is not a
local trap: with the merger branch frozen, a late switch is simply bad.  The late switch the
physical fit chose only works together with a refit merger branch (r0, m0, m1), consistent with
sec 15's 10-free variant.

**Switch + merger freed (job 3473675).**  p00, p01, w0, r0, m0, m1 + 4 spin free (10; only the
inspiral shape b1, b2, a2 frozen), same 18-point subset, two starts: the stored values, and the
physical fit's late switch + merger (start mean 1.03e-2).  BOTH converged (4252 / 4196 evals) to
the SAME point: p00 2.275, p01 7.33, w0 1.946, r0 0.786, m0 0.964, m1 2.55 (1.397 frozen);
s11 +0.153, s12 -0.091, s21 +8.62, s22 +2.75 -- a switch between the nonspinning (0.89) and the
physical (4.08) one.

| arm | free | mean | median | max |
|:--|--:|--:|--:|--:|
| control | 4 | 2.5604e-03 | 1.2394e-03 | 1.1870e-02 |
| switch | 7 | 2.3224e-03 | 1.2312e-03 | 1.0854e-02 |
| switch + merger | 10 | 1.6824e-03 | 1.1645e-03 | 6.0998e-03 |

Against the control: mean -34%, median -6%, max -49%.  chi1 = +0.8 falls at every q (q = 3:
1.14e-2 -> 6.10e-3; q = 8: 1.19e-2 -> 5.39e-3).  The cost is the q = 3 row at low spin
(chi1 = -0.2: 5.6e-4 -> 1.12e-3; chi1 = 0: 4.52e-4 -> 9.66e-4) and chi1 = 0 at q = 5 (+54%);
chi1 = 0 at q = 8 halves (4.96e-4 -> 2.33e-4).  q = 3 is the edge of both the training grid and
BHPTNRSur2dq1e3 -- the same row where pn_anc_spin_physical pays.  With these freed the model no
longer reduces to nonspinning pn_anchored at chi1 = 0.  Subset only: needs the full grid.

*Note on the late start.*  The physical fit is a different model; its switch/merger values were
used ONLY as a second starting point in this subset check.  No result depends on them: the run
from the stored values reached the same point by itself.  The full-grid multistart (job below)
uses starts from this model only: the stored values, that subset answer, and two +-30%
perturbations of the stored values (seeds 1, 2).

**What freeing the switch + merger means (for the paper).**  The model's form is unchanged
(same equations, same 4 polynomial spin terms; the X1^(6/5) inspiral anchor and beta_r(q, chi1)
stay derived).  p00, p01, w0, r0, m0, m1 remain spin-INDEPENDENT (functions of nu only, one
value shared by every chi1) but are fitted on the 2d spin grid instead of copied from the
nonspinning fit; b1, b2, a2 stay at the nonspinning values.  Consequence: the model no longer
reduces exactly to nonspinning pn_anchored at chi1 = 0, and the chi1 = 0 column is fitted data,
not an independent check -- the same treatment as fluxanchored_spin, energy_anchored_spin and
pn_anc_spin_physical.

**Full-grid multistart, switch + merger freed (job 3473724, 96 points, 85 min).**  Four starts,
all from this model; all four CONVERGED.  Files `test_free_switch_results/merger_full_*.json`.

| start | evals | mean | median | max | p00 | p01 | w0 | r0 | m0 | m1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| stored | 3435 | 1.1722e-03 | 8.6491e-04 | 6.5772e-03 | 2.095 | 6.847 | 1.922 | 0.799 | 0.963 | 2.297 |
| 18-pt answer | 2058 | 1.1722e-03 | 8.6209e-04 | 6.5804e-03 | 2.123 | 6.840 | 1.930 | 0.799 | 0.963 | 2.310 |
| stored +-30%, seed 2 | 4264 | 1.1722e-03 | 8.6482e-04 | 6.5772e-03 | 2.095 | 6.849 | 1.922 | 0.799 | 0.964 | 2.290 |
| stored +-30%, seed 1 | 869 | 3.0409e-02 | 2.1376e-02 | 8.9862e-02 | 8.695 | 3.634 | **-6.504** | 1.111 | 0.827 | 1.332 |

Three starts agree to ~1% in every coefficient (spin terms s11 +0.05, s12 -0.06, s21 +10.02,
s22 +1.67; s22 the loosest, 1.53-1.68).  Seed 1 started far off (mean 0.14) and ran into an
unphysical region (NEGATIVE switch width w0 = -6.5, which model_ab floors to 1e-3, i.e. a step
switch; s22 = -67.8): a failed start, not a competing minimum -- it is 26x worse.

Against the stored (9 frozen) coefficients: mean -30%, median -3%, max -61%; 75 of 96 cases
better.  Against published (1.2808e-03 / 7.6858e-04 / 9.6416e-03): mean -8.5%, median +12.5%,
max -32%.  Column medians (x1e-3, new/stored): chi1 -0.6 1.10/1.23, -0.4 0.60/0.60, -0.2
0.47/0.51, 0 0.50/0.46, +0.2 0.58/0.63, +0.4 0.89/0.95, +0.6 1.18/1.96, +0.8 2.96/5.50.  The
cost is concentrated in the q = 3 row at low/mid spin (chi1 = 0: 0.45 -> 1.04; +0.2: 0.44 ->
1.25), the edge of the training grid and of BHPTNRSur2dq1e3.  Test only -- nothing installed.

**Below the grid, q = 2 and 2.5 (2026-09-27, `score_lowq_grid.py`, job 3474032; extrapolation --
below TRAIN_Q and below BHPTNRSur2dq1e3's q_min = 3; the surrogate warns "outside bounds").**
q = 3 rescored to ratio 1.00000 for both models.  mathcalE x 1e-3, chi1 = -0.6 ... +0.8:

| model | q | -0.6 | -0.4 | -0.2 | 0 | +0.2 | +0.4 | +0.6 | +0.8 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pn_anc_spin_physical | 2.0 | 14.76 | 26.35 | 29.38 | 36.29 | 29.90 | 26.37 | 13.85 | 20.72 |
| pn_anc_spin_physical | 2.5 | 2.98 | 6.10 | 6.59 | 7.54 | 6.89 | 5.96 | 4.16 | 7.19 |
| pn_anchored_spin switch+merger (test) | 2.0 | 7.78 | 14.84 | 16.42 | 22.44 | 17.69 | 13.69 | 9.05 | 25.97 |
| pn_anchored_spin switch+merger (test) | 2.5 | 1.88 | 3.74 | 4.08 | 4.86 | 4.64 | 3.82 | 4.10 | 11.27 |

Worst at chi1 = 0 for both, where nonspinning pn_anchored (on BHPTNRSur1dq1e4, valid there)
scores 1.16e-3 at q = 2 -- so part of this is likely the 2d surrogate's own extrapolation, not
the model.  Not yet separated.  Values: `test_free_switch_results/lowq_grid.json`.

**Full q-chi1 grid, INSTALLED coefficients (switch + merger fitted)** -- mathcalE x 1e-3
(full resolution, vs NRHybSur3dq8; coefficients `test_free_switch_results/merger_full_stored.json`).
q = 2 and 2.5 (bold) are extrapolation: below TRAIN_Q and below BHPTNRSur2dq1e3's q_min = 3.

| q \ chi1 | -0.6 | -0.4 | -0.2 | +0.0 | +0.2 | +0.4 | +0.6 | +0.8 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **2.000** | 7.78 | 14.84 | 16.42 | 22.44 | 17.69 | 13.69 | 9.05 | 25.97 |
| **2.500** | 1.88 | 3.74 | 4.08 | 4.86 | 4.64 | 3.82 | 4.10 | 11.27 |
| 3.000 | 0.93 | 1.05 | 1.17 | 1.04 | 1.25 | 1.18 | 2.07 | 6.12 |
| 3.256 | 0.90 | 0.67 | 0.69 | 0.60 | 0.76 | 0.81 | 1.59 | 4.80 |
| 3.513 | 0.95 | 0.53 | 0.50 | 0.48 | 0.58 | 0.69 | 1.31 | 3.90 |
| 3.897 | 1.04 | 0.50 | 0.42 | 0.52 | 0.55 | 0.73 | 1.13 | 3.06 |
| 4.200 | 1.07 | 0.53 | 0.42 | 0.56 | 0.58 | 0.81 | 1.09 | 2.66 |
| 4.600 | 1.09 | 0.56 | 0.42 | 0.58 | 0.63 | 0.91 | 1.10 | 2.33 |
| 5.000 | 1.10 | 0.57 | 0.43 | 0.54 | 0.64 | 1.00 | 1.14 | 2.17 |
| 5.436 | 1.12 | 0.58 | 0.43 | 0.46 | 0.62 | 1.02 | 1.16 | 2.14 |
| 5.949 | 1.22 | 0.61 | 0.44 | 0.36 | 0.55 | 1.01 | 1.18 | 2.30 |
| 6.590 | 1.46 | 0.69 | 0.49 | 0.25 | 0.45 | 0.95 | 1.18 | 2.86 |
| 7.231 | 1.86 | 0.86 | 0.60 | 0.21 | 0.36 | 0.87 | 1.18 | 4.00 |
| 8.000 | 2.53 | 1.18 | 0.81 | 0.22 | 0.29 | 0.77 | 1.22 | 6.58 |

Ratio pn_anc_spin_physical / this candidate is tabulated at the end of
scaling_pn_anc_spin_physical.md.  This candidate wins chi1 = 0 and +0.2 at every q and the low/mid
spin cells for q <= 5.4; physical wins chi1 = +0.8 at every q.

**Figures redrawn 2026-09-27 on the current coefficients** (job 3474058): chi_const / q_const
alpha-beta overlays, coarse + dense mismatch heatmaps, in `Agentic_spin_plots/pn_anchored_spin/`; previous
versions in `Agentic_spin_plots_pn_pre20260927.tar.gz`.  **New paper-style figures** (after Islam+
arXiv:2407.18319v2 Figs 5, 7, 9a; `plot_spin_paper_figs.py`) in `Agentic_spin_plots/pn_anchored_spin/paper/`:
* `pn_anchored_spin_hist_train_vs_heldout` -- log10 mathcalE, training (96) vs held-out (270 in-range points
  off the training grid: 3 < q < 8, chi1 in (-0.6, 0.8), 15 x 18).  Held-out scores in
  `paper/heldout_mismatch.json`.
* `pn_anchored_spin_calibration_chi+0.6` -- Re h22 before / after calibration, q = 3, 5, 8 at chi1 = +0.6.
* `pn_anchored_spin_mismatch_vs_chi1` -- mathcalE vs chi1 for q = 3, 5, 8.

Held-out (270 in-range points): mean 8.27e-4 / median 7.10e-4 / max 2.91e-3 -- below the training numbers because the held-out set is interior (no q = 3 or 8 row, no chi1 = -0.6 or +0.8 column), where this model's errors are largest; it is interpolation, not a harder test.
