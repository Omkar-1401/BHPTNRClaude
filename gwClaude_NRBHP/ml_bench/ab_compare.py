"""The final alpha(t), beta(t) of every paradigm on one footing, against the NR equal-phase alpha, beta (V300, score-only).

alpha-beta paradigms give alpha, beta directly ('native'): pass 40 (bhpt_spin_calib_p40.alpha_beta, T96 coefficients) and
the clock-tau RBF (models/abtau_rbf.pkl, 714 pool).  Direct paradigms give only a waveform h_model; their EFFECTIVE alpha,
beta come from the same equal-phase construction that defines the NR truth (ab_targets.py), applied to (h_pp, h_model).
Pass 40 is also put through that extraction as a consistency check of the two routes.

Per case and model, on ab_targets.S_GRID (s = t - t_peak,pp in m1) where the NR pairing is inside the NR window, weighted by
|h_NR(tau)|^2 ds:
    dlna  = rms(ln alpha - ln alpha_NR)
    dbeta = rms(beta / beta_NR - 1)
    dtau  = rms(tau - tau_NR) [M] after removing the weighted mean (the t0 freedom of the map)
in two gauges: NR truth with both phases zeroed at their own peak (delta = 0), and '|g': the NR pairing phase offset delta
chosen per case to best match the model's own clock (alpha, beta are defined only up to the map's phi0 / t0 freedom; pass 40
sits at delta ~ 2.5-3 rad).  Weights and the 'inside' mask are from the delta = 0 truth.

    python ab_compare.py --nproc 128      -> results/ab_compare.json (metrics, all V300) + results/ab_compare_curves.npz
"""
from __future__ import annotations
import argparse, json, pickle, time
from multiprocessing import get_context

import numpy as np
from scipy.integrate import cumulative_trapezoid

import common as C
import ab_targets as AB
import direct_targets as D
import bench_direct as B
import extrap as X
import gate
import nn_direct as N
import torch                                                   # noqa: E402  (after numpy)

RES = C.HERE / "results"; MOD = C.HERE / "models"
S = AB.S_GRID
DS = np.gradient(S)
MODELS = ["p40", "p40_x", "abtau_rbf", "mlp_T96", "mlp_572", "rbf_direct"]
ALL = MODELS + [m + "|g" for m in MODELS]       # '|g': vs the NR truth in the phase gauge matched to the model's own clock
PRED = {}                                                      # filled in main, inherited by the fork pool


def eq_ab(tp, hp, tm, hm, delta=0.0):
    """equal-phase alpha, beta, tau on S for pp (tp [m1], hp) vs a model or NR waveform (tm [M], hm); as ab_targets.targets.
    delta: phase offset of the pairing (phi_m(tau) = phi_pp(t) + delta); 0 = both phases zeroed at their own |h| peak"""
    sp, php = AB._phase(tp, hp); sn, phn = AB._phase(tm, hm)
    php, phn = AB._mono(php), AB._mono(phn)
    phs = np.interp(S, sp, php) + delta; phn_s = phn + 1e-12 * np.arange(len(phn))
    tau = np.interp(phs, phn_s, sn)
    beta = np.gradient(tau, S)
    alpha = np.interp(tau, sn, np.abs(hm)) / np.interp(S, sp, np.abs(hp))
    inside = (phs > phn_s[0]) & (phs < phn_s[-1])
    return alpha, beta, tau, inside


def native(tp, hp, a, b):
    s = tp - D._tpeak(tp, hp)
    al, be = np.interp(S, s, a), np.interp(S, s, b)
    return al, be, cumulative_trapezoid(be, S, initial=0.0)


def _dtau(ta, tN, w):
    dt = ta - tN; dt = dt - np.sum(w * dt); return float(np.sqrt(np.sum(w * dt ** 2)))


def gauge_match(tp, hp, c, ta, w):
    """phase offset delta of the NR pairing that best matches a native clock tau (alpha, beta are defined only up to the
    map's phi0 / t0 freedom; a native fit sits in its own gauge).  Scan +-6 pi, then bounded refine."""
    f = lambda d: _dtau(ta, eq_ab(tp, hp, c["t_nr"], c["h_nr"], d)[2], w)
    grid = np.linspace(-6 * np.pi, 6 * np.pi, 241); j = int(np.argmin([f(d) for d in grid]))
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(f, bounds=(grid[max(j - 1, 0)], grid[min(j + 1, 240)]), method="bounded", options=dict(xatol=1e-4))
    return float(r.x)


def one(i):
    k = C.V300[i]; c = C.case(k); tp, hp = c["t_bhpt"], c["h_bhpt"]
    aN, bN, tN, inside = eq_ab(tp, hp, c["t_nr"], c["h_nr"])
    w = np.interp(tN, c["t_nr"] - D._tpeak(c["t_nr"], c["h_nr"]), np.abs(c["h_nr"])) ** 2     # tau is on the NR peak clock
    w = w * DS * inside; w = w / w.sum()
    cur = {"nr": (aN, bN, tN)}
    a, b = gate.ab_fun(k, c)("full"); cur["p40"] = native(tp, hp, a, b)
    ts, hm = gate.wave_fun(k, c)("full"); cur["p40_x"] = eq_ab(tp, hp, ts, hm)[:3]
    la, td = PRED["abtau_rbf"][0][i], PRED["abtau_rbf"][1][i]
    be = np.clip(1.0 + np.gradient(td, S), 1e-3, None)
    cur["abtau_rbf"] = (np.exp(la), be, cumulative_trapezoid(be, S, initial=0.0))
    for m in ("mlp_T96", "mlp_572", "rbf_direct"):
        sm, hm = D.render(k, c, PRED[m][0][i], PRED[m][1][i], "full")
        cur[m] = eq_ab(tp, hp, sm, hm)[:3]
    # every model also against the NR truth in the gauge that best matches its own clock
    gauges = {}
    for m in MODELS:
        dl = gauge_match(tp, hp, c, cur[m][2], w); gauges[m] = dl
        cur[f"nr|{m}"] = eq_ab(tp, hp, c["t_nr"], c["h_nr"], dl)[:3]
    met = {"delta": gauges}
    for m in ALL:
        al, be, ta = cur[m.split("|")[0]]
        aN, bN, tN = cur["nr|" + m.split("|")[0]] if m.endswith("|g") else cur["nr"]
        met[m] = dict(dlna=float(np.sqrt(np.sum(w * (np.log(np.clip(al, 1e-12, None)) - np.log(np.clip(aN, 1e-12, None))) ** 2))),
                      dbeta=float(np.sqrt(np.sum(w * (be / np.clip(bN, 1e-3, None) - 1) ** 2))),
                      dtau=_dtau(ta, tN, w))
    return i, met, cur, inside


def predictions():
    V = np.array(C.V300)
    for tgt, m, name in (("abtau", "rbf", "abtau_rbf"), ("direct", "rbf", "rbf_direct")):
        B.TARGET = tgt
        PRED[name] = X._Unpickler(open(MOD / f"{tgt}_{m}.pkl", "rb")).load().predict(V)
    d = N.load(); keys = d["keys"]
    dV = dict(d, keys=V.astype(float))
    m = N.MLP(); m.load_state_dict(torch.load(MOD / "nn_mlp_w.pt"))
    PRED["mlp_572"] = N.curves_from(m, "mlp", dV, np.arange(len(V)), None)
    idx = {tuple(k): i for i, k in enumerate(map(tuple, keys))}; tr = np.array([idx[tuple(k)] for k in C.T96])
    A_, P_ = d["A"], d["P"]; mA, mP = A_[tr].mean(0), P_[tr].mean(0)        # T96 normalisation, as nn_extra
    d96 = dict(dV, mA=mA, mP=mP, sA=(A_[tr] - mA).std(), sP=(P_[tr] - mP).std())
    m = N.MLP(); m.load_state_dict(torch.load(MOD / "nn_mlp_w_T96.pt"))
    PRED["mlp_T96"] = N.curves_from(m, "mlp", d96, np.arange(len(V)), None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16); a = ap.parse_args()
    torch.set_num_threads(a.threads); t0 = time.time()
    predictions(); print(f"predictions {time.time() - t0:.0f} s", flush=True)
    # cases to keep curves for, chosen from the stored E only: p40 median, p40 worst, MLP(T96) worst relative to p40,
    # a retrograde high-q and a mid-range case
    p40 = np.array([json.loads((C.REPO / "training_size" / "results" / "p40__G96__t96.json").read_text())
                    ["per_case"]["V300"][f"{q:.6f}_{c:+.3f}"] for q, c in C.V300])
    mT = np.array(json.loads((RES / "nn_mlp_extra.json").read_text())["mlpw|T96|V300"]["cases"])
    V = np.array(C.V300)
    near = lambda q, c: int(np.argmin((V[:, 0] - q) ** 2 / 25 + (V[:, 1] - c) ** 2 / 1.96))
    keep = dict(p40_median=int(np.argsort(p40)[150]), p40_worst=int(np.argmax(p40)), mlpT96_vs_p40=int(np.argmax(mT / p40)),
                retro_highq=near(7.5, -0.5), mid=near(5.5, 0.2))
    with get_context("fork").Pool(a.nproc) as pool:
        res = pool.map(one, range(len(V)), chunksize=1)
    print(f"done {time.time() - t0:.0f} s", flush=True)
    out = dict(models=MODELS, keys=[list(k) for k in V], keep=keep,
               metrics={m: {s: [r[1][m][s] for r in res] for s in ("dlna", "dbeta", "dtau")} for m in ALL},
               delta={m: [r[1]["delta"][m] for r in res] for m in MODELS})
    for m in ALL:
        out[m] = {s: dict(C.stats(np.array(out["metrics"][m][s]))) for s in ("dlna", "dbeta", "dtau")}
        print(m, {s: f"{out[m][s]['median']:.2e}/{out[m][s]['max']:.2e}" for s in out[m]}, flush=True)
    (RES / "ab_compare.json").write_text(json.dumps(out, indent=1))
    curves = {"s": S}
    for name, i in keep.items():
        _, _, cur, inside = res[i]
        curves[f"{name}|inside"] = inside
        for m, (al, be, ta) in cur.items():
            curves[f"{name}|{m}|alpha"], curves[f"{name}|{m}|beta"], curves[f"{name}|{m}|tau"] = al, be, ta
    np.savez_compressed(RES / "ab_compare_curves.npz", **curves)


if __name__ == "__main__":
    main()
