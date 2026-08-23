"""Is the POST-MERGER beta the QNM ratio?  Parameter-free test, for the E-alpha /
E-beta model (gwr_energy_anchored: alpha = alpha_PP(1+alpha_E*E), beta = beta_PP+P*E).

Motivation (2026-08-11).  The measured conflict is that beta's inspiral wants a rising
slope at every q while the merger-ringdown wants a falling one, and a single coefficient
on E(t) cannot serve both.  One way to free the inspiral coefficient is to PIN the
post-merger beta to the remnant, which costs zero parameters if the physics holds.

beta is the time-rescaling dtau/dt, and matching phases means omega_NR*(dtau/dt) =
omega_pp, so in the ringdown -- where both waveforms sit on a QNM plateau --

    beta_ringdown  =  omega_pp^QNM / omega_NR^QNM

`BHPTNRPNAnchored._beta_r_phys` implements exactly that ratio:
    numerator   0.3683*(1+q)/q                       Schwarzschild (2,2,0) of the primary
                                                     (m1 = q/(1+q), so /m1 = *(1+q)/q)
    denominator (F1 + F2*(1-chif)^F3)/mf              Berti Kerr (2,2,0) fit for the
                                                     remnant, converted to total-mass units
with (M_f, chi_f) from surfinBH NRSur3dq8Remnant.

This script tests all three legs separately, so a disagreement can be localised:
  1. measured omega_pp plateau  vs  0.3683*(1+q)/q
  2. measured omega_NR plateau  vs  Berti/mf
  3. measured ratio             vs  beta_r_phys(q)          <- the claim being tested
  4. and what the E-model's own beta actually does post-merger, from its fitted coeffs.

Leg 3 is the one that decides whether the post-merger branch can be pinned for free.
Recall pn_anchored needs r0 = 0.6869 -- a 31% correction -- on top of beta_r_phys, so
the expectation going in is that the literal ratio does NOT hold; this locates why.

No fitting anywhere: omegas are read off the waveforms, the QNM values are formulas.
"""
import sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_anchored as AN
import BHPTNRPNAnchored as PNA

QS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0]
RING = (30.0, 90.0)          # ringdown plateau window, M, relative to each merger
F_SCHW = 0.3683              # the constant pn_anchored uses for the primary's (2,2,0)


def omega_plateau(t, h, lo, hi):
    """Median |d phase/dt| over [merger+lo, merger+hi], plus the spread there."""
    t = np.asarray(t, float); h = np.asarray(h)
    im = int(np.argmax(np.abs(h)))
    ph = np.unwrap(np.angle(h))
    w = np.abs(np.gradient(ph, t))
    m = (t >= t[im] + lo) & (t <= t[im] + hi)
    if m.sum() < 5:
        return np.nan, np.nan, np.nan
    ww = w[m]
    return float(np.median(ww)), float(ww.std() / abs(np.median(ww)) * 100), float(t[im])


def qnm_nr(q):
    """Berti (2,2,0) of the remnant, in TOTAL-mass units -- the beta_r_phys denominator."""
    mf, chif = PNA._remnant(q)
    qn = PNA._cfg()["qnm"]
    return (qn["F1"] + qn["F2"] * (1.0 - chif) ** qn["F3"]) / mf, mf, chif


# the shipped E-alpha / E-beta model (gwr_energy_anchored, global)
d = json.loads((AN.RESULTS / "coeffs.json").read_text())
th = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                     [float(d["b"])], np.asarray(d["P"], float)])
aE_deg = int(d["alphaE_degree"])

print("Post-merger beta: measured omega ratio vs the QNM prediction "
      f"(plateau window merger+{RING[0]:.0f}..{RING[1]:.0f} M)\n")
print(f"{'q':>4s} | {'w_pp meas':>10s} {'w_pp QNM':>9s} {'diff':>7s} "
      f"| {'w_NR meas':>10s} {'w_NR QNM':>9s} {'diff':>7s} "
      f"| {'beta meas':>9s} {'beta_r_phys':>11s} {'diff':>8s} | {'model beta':>10s} {'diff':>8s}")
rows = {}
for q in QS:
    case = FL.add_flux(G.load_case(q, 6, 10))
    wpp, spp, tm_pp = omega_plateau(case["t_bhpt"], case["h_bhpt"], *RING)
    wnr, snr, tm_nr = omega_plateau(case["t_nr"], case["h_nr"], *RING)
    wpp_q = F_SCHW * (1.0 + q) / q
    wnr_q, mf, chif = qnm_nr(q)
    beta_meas = wpp / wnr
    beta_phys = PNA._beta_r_phys(q)

    # what the E-model's beta actually is at the end of the window
    p = AN.params_at(q, th, aE_deg)
    e = np.asarray(case["losses"]["e_oft"], float)
    beta_model_end = float(p[2] * (1.0 + p[3] * e[-1]))

    f = lambda a, b: (a - b) / b * 100.0
    print(f"{q:4.1f} | {wpp:10.5f} {wpp_q:9.5f} {f(wpp, wpp_q):+6.2f}% "
          f"| {wnr:10.5f} {wnr_q:9.5f} {f(wnr, wnr_q):+6.2f}% "
          f"| {beta_meas:9.5f} {beta_phys:11.5f} {f(beta_meas, beta_phys):+7.2f}% "
          f"| {beta_model_end:10.5f} {f(beta_model_end, beta_meas):+7.2f}%")
    rows[q] = dict(w_pp=wpp, w_pp_qnm=wpp_q, w_pp_spread_pct=spp,
                   w_NR=wnr, w_NR_qnm=wnr_q, w_NR_spread_pct=snr,
                   beta_meas=beta_meas, beta_r_phys=beta_phys,
                   beta_model_end=beta_model_end, mf=mf, chif=chif)

print(f"\nplateau spreads (should be small if these really are QNM plateaus):")
for q in QS:
    print(f"   q={q:g}: omega_pp {rows[q]['w_pp_spread_pct']:5.2f}% , "
          f"omega_NR {rows[q]['w_NR_spread_pct']:5.2f}%   "
          f"(M_f={rows[q]['mf']:.4f}, chi_f={rows[q]['chif']:.4f})")

inr = [q for q in QS if 3.0 <= q <= 8.0]
r = np.array([rows[q]["beta_meas"] / rows[q]["beta_r_phys"] for q in inr])
print(f"\nmeasured/predicted over q in [3,8]: mean {r.mean():.4f}  "
      f"spread {r.std()/r.mean()*100:.2f}%  range [{r.min():.4f}, {r.max():.4f}]")
print(f"pn_anchored's fitted r0 = {PNA._cfg()['theta'][6]:.4f}  <- compare to the mean above")
print("\nverdict: a q-FLAT ratio means the QNM anchor is real up to one constant (which is "
      "what r0 is); a q-DEPENDENT ratio means it cannot be pinned with zero parameters.")
json.dump({str(k): v for k, v in rows.items()},
          open(str(HERE / "ringdown_beta_vs_qnm.json"), "w"), indent=1, default=float)
