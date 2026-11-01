"""Frequency-domain mismatch vs NR across mass ratio (Zhong et al. 2025 Fig. 14 style), one model per process.

Points: q in linspace(3, 8, 21) x chi1 in CHIS.  Total masses MTOT [Msun].  aLIGO zero-detuned high-power PSD,
f_low = 20 Hz (raised per case if the waveform starts too late; flagged), maximised over time shift and template
polarisation / phase, reported for four signal polarisations (fd_core.py).  The NR and model waveforms are the ones
the time-domain score uses (same window, same alignment), so E and the FD mismatch see identical data.
Gates before anything is written: the model reproduces its stored T96 scores on T96[::8]; self-tests on the first point:
  (1) match(NR, NR) = 0;  (2) an explicit detector strain F+ h+ + Fx hx (arbitrary iota, phi, psi, sky) gives the same
  mismatch as the Re[e^{i delta} h22] form at delta = arg K (the single-mode degeneracy); (3) white-PSD mismatch vs E.
    python fdm/fd_run.py MODEL     MODEL in zhong | rink | flux_orbit | p40 | p40lean | p49b12 | p49b9
    -> fdm/results/fd_<MODEL>.json
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

import fd_route                                 # noqa: F401  (first: cache routing)
import fd_core as FC
S = fd_route.S
HERE = Path(__file__).resolve().parent; EXPL = HERE.parent; REPO = EXPL.parent
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
MODEL = sys.argv[1]
QS = [round(float(q), 6) for q in np.linspace(3.0, 8.0, 21)]
CHIS = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.8]
PTS = [(q, c) for c in CHIS for q in QS]
MTOT = [100.0, 200.0]
T96 = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]; CHK = T96[::8]
k6 = lambda q, c: f"{q:.6f}_{c:+.4f}"
k4 = lambda q, c: f"{q:.4f}_{c:+.1f}"

if MODEL in ("zhong", "rink"):
    import fd_models as FM
    calibrated = lambda k: FM.calibrated(MODEL, *k)
    if MODEL == "rink":
        R = json.loads((REPO / "spin_dep_split" / "results" / "rink_scores.json").read_text())
        _r = {(r["q"], r["chi1"]): r["full_E_full"] for r in R["T96"]}; REF = {k: _r[k] for k in CHK}
    else:
        z = json.loads((OUT / "zhong_scores.json").read_text())["zhong"]["T96_cases"]; REF = {k: z[k6(*k)] for k in CHK}
    LABEL = {"zhong": "Zhong et al. 2025 extracted (8 coef, spin-blind)", "rink": "Rink et al. 2024 (12 coef)"}[MODEL]
elif MODEL == "flux_orbit":
    import plot_spin_paper_figs as PF
    NAME = "flux_anc_spin_with_orbit"
    PF.MODELS[NAME] = PF.LOADERS[NAME]()
    calibrated = lambda k: PF.calibrated(NAME, *k)
    REF = {k: PF.MODELS[NAME]["stored"][k4(*k)] for k in CHK}
    LABEL = "flux_anc_spin_with_orbit (9 coef)"
else:
    import modelsetup
    _M = modelsetup.load(MODEL); P6 = _M["P6"]; TH = np.asarray(_M["TH"]); R = _M["R"]
    REF = {k: R["T96_cases"][k6(*k)] for k in CHK}
    LABEL = _M["label"]

    def calibrated(k):                          # plot_model.calibrated, verbatim logic
        _, t0 = P6.err(TH, k, "fit")
        e, t1 = P6.err(TH, k, "full", seed=t0)
        c = P6.get(k); case = c["case"]; t = case["t_bhpt"]
        alpha, beta = P6.ab(TH, k, "full")
        bc = P6.creative.cumulative_trapezoid(beta, t); tau = t1 + bc - np.interp(S.T_ANCHOR, t, bc)
        h, tn, hn = case["h_bhpt"], case["t_nr"], case["h_nr"]
        m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
        tc, r = tn[m], hn[m]
        g = np.interp(tc, tau, alpha) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
        Z = np.sum(r * np.conj(g)); n1, n2 = np.sum(np.abs(r) ** 2), np.sum(np.abs(g) ** 2)
        e2 = float((n1 + n2 - 2 * abs(Z)) / (2 * n1))
        assert abs(e2 / e - 1) < 1e-6, (k, e, e2)
        return dict(tc=tc, hnc=r, hcal=g * Z / abs(Z), E=e)


def _E(k):
    return calibrated(k)["E"]


def _work(k):
    try:
        w = calibrated(k)
        row = dict(q=k[0], chi1=k[1], E=w["E"])
        for M in MTOT:
            row[f"M{int(M)}"] = FC.match(w["tc"], w["hnc"], w["hcal"], M)
        row["white_M100"] = FC.match(w["tc"], w["hnc"], w["hcal"], 100.0, psd="white", f_low=1.0)["mm"]
        return row
    except Exception as exc:
        print(f"  [fail] {k}: {exc!r}", flush=True); return dict(q=k[0], chi1=k[1], error=repr(exc))


def selftest(k):
    w = calibrated(k); tc, hn, hm = w["tc"], w["hnc"], w["hcal"]
    z = FC.match(tc, hn, hn, 100.0)["mm_worst"]
    th, ps, io, ph, va = 0.9, 0.7, 1.0, 0.4, 2.1
    s = FC.detector_strain_check(tc, hn, io, ph, ps, th, va).astype(complex)
    c = np.cos(io); Y22 = np.sqrt(5 / (64 * np.pi)) * (1 + c) ** 2 * np.exp(2j * ph)
    Y2m2 = np.sqrt(5 / (64 * np.pi)) * (1 - c) ** 2 * np.exp(-2j * ph)
    Fp = 0.5 * (1 + np.cos(th) ** 2) * np.cos(2 * va) * np.cos(2 * ps) - np.cos(th) * np.sin(2 * va) * np.sin(2 * ps)
    Fx = 0.5 * (1 + np.cos(th) ** 2) * np.cos(2 * va) * np.sin(2 * ps) + np.cos(th) * np.sin(2 * va) * np.cos(2 * ps)
    K = (Fp + 1j * Fx) * Y22 + np.conj((Fp + 1j * Fx) * Y2m2)
    assert np.allclose(s.real, np.real(K * hn), atol=1e-12 * np.abs(hn).max()), "strain != Re[K h22]"
    a = FC.match(tc, s, hm, 100.0, deltas=(0.0,))["mm"]
    b = FC.match(tc, hn, hm, 100.0, deltas=(float(np.angle(K)),))["mm"]
    wm = FC.match(tc, hn, hm, 100.0, psd="white", f_low=1.0)["mm"]
    print(f"selftest {k}: match(NR,NR) mismatch {z:.1e}; explicit strain {a:.6e} vs delta=argK {b:.6e}; "
          f"white-PSD mismatch {wm:.3e} vs time-domain E {w['E']:.3e}", flush=True)
    assert z < 1e-8 and abs(a / b - 1) < 1e-6, (z, a, b)


def main():
    t0 = time.time()
    with mp.get_context("fork").Pool(96) as pool:
        e = np.array(pool.map(_E, CHK))
        r = e / np.array([REF[k] for k in CHK])
        print(f"{MODEL}: reproduce stored T96 on {len(CHK)} points: {r.min():.5f}..{r.max():.5f}", flush=True)
        assert np.allclose(r, 1, atol=2e-3), r
        selftest(PTS[len(QS) * 2 + 8])                                 # chi1 = 0, q = 5
        rows = pool.map(_work, PTS, chunksize=1)
    ok = [x for x in rows if "error" not in x]
    for M in MTOT:
        v = np.array([x[f"M{int(M)}"]["mm"] for x in ok]); fl = sum(x[f"M{int(M)}"]["flagged"] for x in ok)
        print(f"  M = {M:.0f}: FD mismatch median {np.median(v):.3e} max {v.max():.2e}; f_low raised on {fl}/{len(ok)}", flush=True)
    (OUT / f"fd_{MODEL}.json").write_text(json.dumps(dict(
        model=MODEL, label=LABEL, psd="aLIGOZeroDetHighPower", f_low=FC.F_LOW, fs=FC.FS_HZ, taper_M=[FC.T_ON, FC.T_OFF],
        deltas=list(FC.DELTAS), MTOT=MTOT, reproduce_ratio=[float(r.min()), float(r.max())], rows=rows), indent=1))
    print(f"{MODEL}: {len(ok)}/{len(rows)} points, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
