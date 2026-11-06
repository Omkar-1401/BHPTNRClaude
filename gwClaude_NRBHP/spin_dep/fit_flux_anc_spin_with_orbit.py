"""flux_anc_spin_with_orbit: fluxanchored_spin + the derived 1.5PN spin-orbit term in beta (9 coefficients, one form
for the whole inspiral-merger-ringdown).  Promoted 2026-10-05 (user) from the fluxanchored_spin_so variant.
md: scaling_flux_anc_spin_with_orbit.md.

    alpha = a_pp [1 + a_F F(t)]
    beta  = [b_pp + P E(t)] [1 + kappa_SO(q) chi1 x^(3/2)]
    (a_pp, a_F, b_pp, b_e = P / b_pp) = fit_fluxanchored_spin.params_at(q, chi1, theta)   (WEIGHT = "nu"; the parent's form)
    kappa_SO(q) = (3/2) [113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1]                         (derived; no fitted parameter)
    x = X1^(-4/5) min(x_pp, XCLIP), x_pp from the ppBHPT waveform (fit_pn_anchored_spin.build_case's `x`)

Scoring is the parent's own path (fit_fluxanchored_spin.score / plot_spin_paper_figs._flux): coarse t0 scan on the
fit grid, refine, then the full-resolution mismatch -- with the SO factor as the only change.

    python fit_flux_anc_spin_with_orbit.py --install      # write coeffs.json from the fitted theta (scores T96, gate)
    python fit_flux_anc_spin_with_orbit.py --refit        # refit theta on T96 (mean full-window E), then --install
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fit_fluxanchored_spin as FS
import fit_pn_anchored_spin as S
G, GG = FS.G, FS.FL.GG
PC = S.PC

MODEL = "flux_anc_spin_with_orbit"
MODEL_FORM = ("fluxanchored_spin form (WEIGHT nu) x [1 + kappa_SO chi1 x^(3/2)] on beta; "
              "kappa_SO = (3/2)(113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1)")
RESULTS = HERE / "flux_anc_spin_with_orbit_results"; RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs.json"
PROVENANCE = RESULTS / "fit_provenance.json"      # the original fit (job 3487767; per-case T96 + H270 scores)
NAMES = list(FS.NAMES)
TRAIN = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
build_case = FS.build_case


ALPHA_SO = False      # VARIANT switch (2026-10-05, pipeline item 3): the derived SO factor on alpha too; OFF for this model


def kappa_alpha(q):
    """Derived 1.5PN SO correction to ln(alpha) (amplitude matching; Arun et al. 2009, arXiv:0810.5336, Eq. h22):
    binary H_SO = (2/3) chi1 (nu - 2 X1) x^(3/2), test mass -(4/3) chi1 x_pp^(3/2), plus -(2/3) delta ln beta_SO."""
    X1 = q / (1 + q)
    return -97 / 12 * X1 ** 1.2 + 5 / 2 * X1 ** 2 + 67 / 12 * X1


def kappa_so(q):
    X1 = q / (1 + q)
    return 1.5 * (113 / 12 * X1 ** 1.2 - 19 / 6 * X1 ** 2 - 25 / 4 * X1)


def params_at(q, chi1, th):
    FS.WEIGHT = "nu"
    return FS.params_at(q, chi1, th)


def model_ab(th, q, chi1, case, key="full"):
    pre = "fit_" if key == "fit" else ""
    a_pp, a_f, b_pp, b_e = params_at(q, chi1, th)
    los = case[pre + "losses"]
    x = PC.x_scale(q) * np.minimum(case[pre + "x"], S.XCLIP)
    alpha = a_pp * (1.0 + a_f * los["flux_hat"])
    if ALPHA_SO:
        alpha = alpha * (1.0 + kappa_alpha(q) * chi1 * x ** 1.5)
    beta = b_pp * (1.0 + b_e * los["e_oft"]) * (1.0 + kappa_so(q) * chi1 * x ** 1.5)
    return alpha, beta


def model_shape(th, q, chi1, case, key="full"):
    pre = "fit_" if key == "fit" else ""
    alpha, beta = model_ab(th, q, chi1, case, key)
    if np.any(beta <= 0):
        return alpha, None, -1.0
    t = case[pre + "t_bhpt"]
    bc = G.creative.cumulative_trapezoid(beta, t)
    tau_shape = bc - float(np.interp(G.T_ANCHOR, t, bc))
    return alpha, tau_shape, float(np.min(np.diff(tau_shape)))


def fast_mismatch(th, q, chi1, case, t0_seed, key="fit"):
    """As fit_scaling_gwr_energy_flux.fast_mismatch (bounded +-60 M t0 search), with this model's alpha, beta."""
    from scipy.optimize import minimize_scalar
    pre = "fit_" if key == "fit" else ""
    alpha, tau_shape, dmin = model_shape(th, q, chi1, case, key)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_seed
    f = lambda t0: GG.err_at_t0(t0, alpha, tau_shape, case[pre + "h_bhpt"], case[pre + "t_nr"], case[pre + "h_nr"])
    r = minimize_scalar(f, bounds=(t0_seed - 60.0, t0_seed + 60.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 50})
    return float(r.fun), float(r.x)


def mismatch(th, q, chi1, case):
    """Full-window mathcalE and t0: the parent's protocol (cold coarse scan, fit-grid refine, full resolution)."""
    g = np.linspace(-250.0, 150.0, 81)
    t0 = float(g[int(np.argmin([fast_mismatch(th, q, chi1, case, x, "fit")[0] for x in g]))])
    _, t0 = fast_mismatch(th, q, chi1, case, t0, "fit")
    return fast_mismatch(th, q, chi1, case, t0, "full")


def key(q, c):
    return f"{q:.4f}_{c:+.1f}"          # house key format (as plot_spin_paper_figs.key)


def load_theta():
    d = json.loads(COEFFS.read_text())
    if d.get("model_form") != MODEL_FORM:
        raise RuntimeError(f"{COEFFS.name} is stamped {d.get('model_form')!r}, not this model's form")
    return np.array(d["theta"], float)


_TH = None


def _score(k):
    return k, mismatch(_TH, *k, build_case(*k))[0]


def score_grid(th, nproc):
    global _TH
    _TH = np.array(th, float)
    for k in TRAIN:                                    # build caches serially before forking
        build_case(*k)
    with get_context("fork").Pool(nproc) as pool:
        return dict(pool.map(_score, TRAIN, chunksize=1))


def install(th, provenance, nproc):
    E = score_grid(th, nproc)
    v = np.array(list(E.values()))
    ref = np.array([r["full_E_full"] for r in provenance["T96"]["flux_so_single"]])
    print(f"T96: median {np.median(v):.4e} mean {v.mean():.4e} max {v.max():.3e}  "
          f"(original fit's scorer: median {np.median(ref):.4e} mean {ref.mean():.4e} max {ref.max():.3e})", flush=True)
    if abs(np.median(v) / np.median(ref) - 1) > 0.01 or abs(v.mean() / ref.mean() - 1) > 0.01:
        sys.exit("gate: this module's scores differ from the original fit's by > 1% -- not installing")
    COEFFS.write_text(json.dumps(dict(
        model=MODEL, model_form=MODEL_FORM, weight="nu", n_coefficients=len(NAMES), names=NAMES,
        theta=list(map(float, th)), grid=dict(q=S.TRAIN_Q, chi1=S.CHIS),
        objective="MEAN over the 96-point grid of the full-window mismatch (this module's mismatch())",
        median=float(np.median(v)), mean=float(v.mean()), max=float(v.max()),
        mismatches={key(*k): float(e) for k, e in E.items()},
        provenance=("theta from fit_fluxanchored_spin_so.py (job 3487767, 2026-10-04; two starts converged to mean "
                    "1.0898e-3); see fit_provenance.json")), indent=1))
    print("wrote", COEFFS)


def refit(th0, nproc):
    global _TH
    for k in TRAIN:
        build_case(*k)
    with get_context("fork").Pool(nproc) as pool:
        def obj(th):
            global _TH
            _TH = np.array(th, float)
            return float(np.mean([e for _, e in pool.map(_score, TRAIN, chunksize=1)]))
        x = np.array(th0, float); t = time.time(); f0 = obj(x)
        r = minimize(obj, x, method="Nelder-Mead", options={"maxiter": 900 * len(x), "xatol": 1e-8,
                                                            "fatol": 1e-12, "adaptive": True})
        print(f"refit: mean {f0:.4e} -> {r.fun:.4e} ({time.time() - t:.0f}s)", flush=True)
    return r.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true"); ap.add_argument("--refit", action="store_true")
    ap.add_argument("--nproc", type=int, default=96)
    a = ap.parse_args()
    prov = json.loads(PROVENANCE.read_text())
    th = np.array(prov["theta_best"], float)
    if a.refit:
        th = refit(th, a.nproc)
    if a.install or a.refit:
        install(th, prov, a.nproc)


if __name__ == "__main__":
    main()
