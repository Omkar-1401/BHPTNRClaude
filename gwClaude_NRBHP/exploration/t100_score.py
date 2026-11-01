"""Score one final candidate on the untouched TEST set (t100_set.py: R100 interior + L20 low q).  No fitting.

    python t100_score.py MODEL        MODEL in v7s3 | p32 | p36 | p38n2 | flux_orbit | pn13 | rink | zhong

Each model is set up exactly as its pass / install script does (one model per process: the exploration modules
monkeypatch P6.ab / P6.get on import).  Before the test set is touched, the stored per-case T96 scores are reproduced on
T96[::8] (12 points) and the run stops unless every ratio is 1 within 2e-3 (CLAUDE.md due diligence 1).
Metric everywhere: full-window mathcalE vs NRHybSur3dq8 (2,2), t0 searched, phi0 analytic, full resolution.
-> exploration/t100/<MODEL>.json  (per-case + median / mean / max for R100 and L20)
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(HERE))
MODEL = sys.argv[1]
import t100_set as TS                          # noqa: E402  (patches FS.build_case for TEST keys; before any model import)
S, FS = TS.S, TS.FS
OUT = HERE / "t100"
T96 = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
CHK = T96[::8]
k6 = lambda q, c: f"{q:.6f}_{c:+.4f}"
k4 = lambda q, c: f"{q:.4f}_{c:+.1f}"

if MODEL in ("v7s3", "p32", "p36", "p38n2", "p39", "p40", "k1"):
    argv0 = sys.argv[0]
    if MODEL == "k1":                           # lean variant: pass 40 + Ori–Thorne onset T_start = -1 (pass 42 K1, 16c)
        argv0 = sys.argv[0]; sys.argv = [argv0, "K1"]
        import p42_otonset as K42; P6 = K42.P6
        P6.get = K42.L.get; P6.ab = K42.L.ab; K42.C.use("O2n"); K42.M.S1FUN = K42.onset
        R = json.loads((HERE / "p42" / "p42_K1.json").read_text())
    elif MODEL == "p40":                          # pass 40: pass 39 L4 coefficients + closed-form R2 (O2, natural range)
        import p40_r2closed as C40; P6 = C40.P6
        P6.get = C40.L.get; P6.ab = C40.L.ab; C40.use("O2n")
        R = json.loads((HERE / "p39" / "p39_L4.json").read_text())
        R = dict(theta=R["theta"], n_coef=R["n_coef"],
                 T96_cases=json.loads((HERE / "p40" / "p40.json").read_text())["O2n"]["T96_cases"])
    elif MODEL == "p39":                        # pass 39 L4: D1 + g(nu) on the q-flat terms (nu_c = 0.04)
        sys.argv = [argv0, "L4"]; import p39_numlimit as L39; P6 = L39.P6
        P6.get = L39.get; P6.ab = L39.ab
        R = json.loads((HERE / "p39" / "p39_L4.json").read_text())
    elif MODEL == "v7s3":
        sys.argv = [argv0, "Z"]; import p23_s3 as Z; P6 = Z.P6
        R = json.loads((HERE / "p23" / "p23_Z.json").read_text())
    elif MODEL == "p32":
        sys.argv = [argv0, "G2"]; import p31_spinq as G; P6 = G.P6
        R = json.loads((HERE / "p32" / "p32_G2.json").read_text())["round2"]
    elif MODEL == "p36":
        sys.argv = [argv0, "D1"]; import p36_rdslope as D; P6 = D.P6
        P6.get = D.get
        R = json.loads((HERE / "p36" / "p36_D1.json").read_text())
    else:
        sys.argv = [argv0, "N2"]; import p38_newshape as N; P6 = N.P6
        P6.get = N.D.get; P6.ab = N.ab
        R = json.loads((HERE / "p38" / "p38_N2.json").read_text())
    TH = list(map(float, R["theta"])); NC = int(R["n_coef"])
    REF = {k: R["T96_cases"][k6(*k)] for k in CHK}

    def err(k):
        return P6._full((TH, k))

elif MODEL == "flux_orbit":
    import fit_flux_anc_spin_with_orbit as FO
    d = json.loads(FO.COEFFS.read_text()); TH = FO.load_theta(); NC = int(d["n_coefficients"])
    REF = {k: d["mismatches"][k4(*k)] for k in CHK}

    def err(k):
        return FO.mismatch(TH, k[0], k[1], FS.build_case(*k))[0]

elif MODEL == "pn13":
    d = json.loads(S.COEFFS.read_text()); TH9 = S.model9(); THS = np.array(d["theta_spin"], float); NC = 13
    REF = {k: d["mismatches"][k4(*k)] for k in CHK}

    def err(k):
        return S.full_mm(TH9, THS, k[0], k[1], FS.build_case(*k))

elif MODEL == "rink":
    # The published calibrated BHPTNRSur2dq1e3 (Rink et al. 2024): calibrated=True already applies their alpha, beta
    # (time in M), so the map here is the identity and only t0, phi0 are optimised (as spin_dep_split/score_rink.py).
    import fit_scaling_PN_opt_creative as creative
    from scipy.optimize import minimize_scalar
    NC = 12
    R = json.loads((REPO / "spin_dep_split" / "results" / "rink_scores.json").read_text())
    _r = {(r["q"], r["chi1"]): r["full_E_full"] for r in R["T96"]}
    REF = {k: _r[k] for k in CHK}

    def _e(x0, ts, h, tn, hn):
        return S._err_at_t0(x0, np.ones_like(ts), ts, ts, h, tn, hn)

    def err(k):
        import BHPTNRSur2dq1e3 as ppsur
        c = FS.build_case(*k)
        t, hd = ppsur.generate_surrogate(q=k[0], spin1=k[1], calibrated=True, modes=[(2, 2)], neg_modes=False)
        t = np.asarray(t, float); h = np.asarray(hd[(2, 2)])
        si = np.unique(np.r_[np.arange(0, len(t), S.SRC_STRIDE), len(t) - 1])
        ts = t - np.interp(S.T_ANCHOR, t, t)
        tf, hf = ts[si], h[si]
        f = lambda x0: _e(x0, tf, hf, c["fit_t_nr"], c["fit_h_nr"])
        g = np.linspace(-300, 200, 251); cen = float(g[int(np.argmin([f(v) for v in g]))])
        x = minimize_scalar(f, bounds=(cen - 6, cen + 6), method="bounded", options={"xatol": 1e-4}).x
        f2 = lambda x0: _e(x0, ts, h, c["t_nr"], c["h_nr"])
        return float(minimize_scalar(f2, bounds=(x - 3, x + 3), method="bounded", options={"xatol": 1e-4}).fun)
elif MODEL == "zhong":
    # Zhong et al. 2025 (arXiv:2503.04534) extracted (2,2) coefficients, their literal phase map (fdm/fd_models.py);
    # the T96 reference is fdm/zhong_score.py's own output (same harness, so this checks the case path only).
    sys.path.insert(0, str(HERE / "fdm"))
    import fd_models as FM
    NC = 8
    _z = json.loads((HERE / "fdm" / "results" / "zhong_scores.json").read_text())["zhong"]["T96_cases"]
    REF = {k: _z[k6(*k)] for k in CHK}

    def err(k):
        return FM.score("zhong", *k)
else:
    sys.exit(f"unknown model {MODEL}")


def stats(v):
    v = np.asarray(v); return dict(median=float(np.median(v)), mean=float(np.mean(v)), max=float(v.max()), n=len(v))


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(96)
    e = np.array(pool.map(err, CHK))
    r = e / np.array([REF[k] for k in CHK])
    print(f"{MODEL} ({NC}c): reproduce stored T96 on {len(CHK)} points, ratio {r.min():.4f}..{r.max():.4f}", flush=True)
    if not np.allclose(r, 1, atol=2e-3):
        sys.exit(f"{MODEL}: reproduction FAILED {np.round(r, 4)}; test set NOT scored")
    et = np.array(pool.map(err, TS.TEST))
    pool.close()
    er, el = et[:100], et[100:]
    out = dict(model=MODEL, n_coef=NC, seed=TS.SEED, reproduce_ratio=[float(r.min()), float(r.max())],
               R100=stats(er), L20=stats(el),
               cases={k6(*k): float(v) for k, v in zip(TS.TEST, et)})
    OUT.mkdir(exist_ok=True)
    (OUT / f"{MODEL}.json").write_text(json.dumps(out, indent=1))
    s, l = out["R100"], out["L20"]
    print(f"{MODEL} ({NC}c): R100 {s['median']:.3e} / {s['mean']:.3e} / {s['max']:.2e} | "
          f"L20 {l['median']:.3e} / {l['mean']:.3e} / {l['max']:.2e}   ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
