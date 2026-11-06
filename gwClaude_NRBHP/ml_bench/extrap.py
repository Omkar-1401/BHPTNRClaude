"""Extrapolation and data-efficiency check, scored on the SAME points for the ML models and the hand-built pass 40.

Sets (cached, outside the [3, 8] training range; the sealed guard still applies; none is used for any choice):
    lowq   2.5 <= q < 3      (q = 2.5, 2.632, 2.947; the gate-L region)
    vlowq  q < 2.5           (q = 2.0, 2.316; the BHPT input itself is extrapolated below q = 2.5)
    highq  q = 9, 10         (exploration/p44 cache; NRHybSur3dq8 itself extrapolated above q = 8)
plus V300 (interpolation, score-only) for the N = 96 refits.
Models: every models/<target>_<method>.pkl (trained on the 714 pool) + the same configs refit on 96 random pool points
(same permutation as bench_direct.curve) + pass 40 (stand-alone generator, coefficients from T96).

    python extrap.py --target direct --nproc 128     -> results/extrap_<target>.json
"""
from __future__ import annotations
import argparse, json, pickle, sys
from multiprocessing import get_context

import numpy as np
import common as C
import bench_direct as B

RES = C.HERE / "results"
P44 = C.REPO / "exploration" / "p44" / "cache"


def sets():
    low, vlow = [], []
    for d in (C.G.WF_SHARED, C.G.WF_DIR):
        for f in d.glob("wf_q*_chi*.npz"):
            s = f.stem[4:]; q = float(s.split("_chi")[0]); c = float(s.split("_chi")[1])
            if C.C_LO - 1e-9 <= c <= C.C_HI + 1e-9:
                (low if 2.5 - 1e-9 <= q < 3 - 1e-9 else vlow if q < 2.5 - 1e-9 else []).append(C.G.key(q, c))
    high = []
    for f in (P44 / "wf").glob("wf_q*_chi*.npz"):
        s = f.stem[4:]; high.append(C.G.key(float(s.split("_chi")[0]), float(s.split("_chi")[1])))
    return {"lowq": sorted(set(low)), "vlowq": sorted(set(vlow)), "highq": sorted(set(high))}


class _Unpickler(pickle.Unpickler):
    """models were pickled from bench_direct run as __main__: map those classes back to the module"""
    def find_class(self, module, name):
        return getattr(B, name) if module == "__main__" else super().find_class(module, name)


def case_any(k):
    if k[0] > 8.5:                                         # p44 cache (not in the grids routing)
        C.S.CACHE = P44 / "wf"; C.FS.LOSSCACHE = P44 / "loss"
        return C.FS.build_case(*k)
    return C.case(k)


def _score_ml(a):
    tgt, k, dA, dP = a
    B.TARGET = tgt
    c = case_any(k)
    if tgt == "direct":
        return C.score_wave(lambda key: B.D.render(k, c, dA, dP, key), c)
    return C.score_ab(lambda key: B.AB.ab_on(k, c, dA, dP, key), c)


def _score_p40(k):
    sys.path.insert(0, str(C.HERE)); import gate
    c = case_any(k)
    return C.score_ab(gate.ab_fun(k, c), c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", default="direct", choices=["direct", "ab"])
    ap.add_argument("--nproc", type=int, default=128); ap.add_argument("--skip-p40", action="store_true")
    a = ap.parse_args()
    B.TARGET = a.target
    S = sets(); S96 = dict(S, V300=list(C.V300))
    print({k: len(v) for k, v in S.items()}, flush=True)
    keys, A, P = B.load_train()
    perm = np.random.default_rng(B.SEED).permutation(len(keys))[:96]
    out = {"sets": {k: [list(x) for x in v] for k, v in S.items()}}
    with get_context("fork").Pool(a.nproc) as pool:
        for name, ks in ([] if a.skip_p40 else S.items()):
            e = np.array(pool.map(_score_p40, ks, chunksize=1))
            out[f"p40|{name}"] = dict(C.stats(e), cases=e.tolist()); print(f"p40        {name:6s} {C.fmt(C.stats(e))}", flush=True)
        for m, tag in B.best_configs().items():
            f = C.HERE / "models" / f"{a.target}_{m}.pkl"
            if not f.exists():
                continue
            sur = _Unpickler(open(f, "rb")).load()
            _, kind, K = tag.split("|")
            s96 = B.Surrogate(m, kind, int(K)).fit(keys[perm], A[perm], P[perm])
            for lab, model, SS in (("714", sur, S), ("96", s96, S96)):
                for name, ks in SS.items():
                    Ap, Pp = model.predict(np.array(ks))
                    e = np.array(pool.map(_score_ml, [(a.target, tuple(k), Ap[i], Pp[i]) for i, k in enumerate(ks)], chunksize=1))
                    out[f"{m}|{lab}|{name}"] = dict(C.stats(e), cases=e.tolist())
                    print(f"{m:5s} N={lab:3s} {name:6s} {C.fmt(C.stats(e))}", flush=True)
    (RES / f"extrap_{a.target}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
