"""Paradigm A benchmark: regress the direct amplitude/phase corrections (direct_targets.py) over (q, chi1).

Representation: dlnA(s), dphi(s) on S_GRID -> PCA (SVD of the centred training curves, fitted on the TRAINING fold only),
K components each.  Regressor: features (q, chi1) in one of four parameterisations -> the 2K coefficients.
Selection on the training pool only:
    inner  hyperparameters per method by K-fold CV of the (standardised) coefficient MSE   (cheap)
    outer  5-fold CV of the WAVEFORM E (common.score_wave) for every method x features x K  (the decision metric)
Final: refit the CV-best (features, K) per method on the whole pool, score V300 once (score-only), record times.

    python bench_direct.py cv    --nproc 128      -> results/direct_cv.json
    python bench_direct.py final --nproc 128      -> results/direct_final.json (+ models/*.pkl)
    python bench_direct.py curve --nproc 128      -> results/direct_curve.json (learning curve, CV-best configs)
"""
from __future__ import annotations
import argparse, json, pickle, sys, time, warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as CK
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.interpolate import RBFInterpolator

import common as C
import direct_targets as D
import ab_targets as AB

TARGET = "direct"          # "direct": dlnA, dphi (paradigm A);  "ab": ln alpha, ln beta via the alpha-beta map (paradigm B)

warnings.filterwarnings("ignore")
RES = C.HERE / "results"; MOD = C.HERE / "models"; MOD.mkdir(exist_ok=True)
SEED = 20261031


# ------------------------------------------------------------------------------------------------ features
def feats(keys, kind):
    q, c = keys[:, 0], keys[:, 1]
    u = {"q": q, "nu": q / (1 + q) ** 2, "invq": 1 / q, "logq": np.log(q)}[kind]
    lo = {"q": 3.0, "nu": 8 / 81, "invq": 1 / 8, "logq": np.log(3)}[kind]
    hi = {"q": 8.0, "nu": 3 / 16, "invq": 1 / 3, "logq": np.log(8)}[kind]
    if kind in ("nu", "invq"):
        lo, hi = min(lo, hi), max(lo, hi)
    return np.c_[(u - lo) / (hi - lo), (c + 0.6) / 1.4]       # unit box


# ------------------------------------------------------------------------------------------------ regressors
class RBF(BaseEstimator, RegressorMixin):
    def __init__(self, kernel="thin_plate_spline", smoothing=0.0, degree=1):
        self.kernel, self.smoothing, self.degree = kernel, smoothing, degree

    def fit(self, X, y):
        self.f_ = RBFInterpolator(X, y, kernel=self.kernel, smoothing=self.smoothing, degree=self.degree); return self

    def predict(self, X):
        return self.f_(X)


class XGB(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=600, max_depth=4, learning_rate=0.05):
        self.n_estimators, self.max_depth, self.learning_rate = n_estimators, max_depth, learning_rate

    def fit(self, X, y):
        import xgboost as xgb
        self.m_ = xgb.XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                   learning_rate=self.learning_rate, subsample=0.8, n_jobs=1,
                                   multi_strategy="multi_output_tree", tree_method="hist", random_state=SEED)
        self.m_.fit(X, y); return self

    def predict(self, X):
        return self.m_.predict(X)


class GPR(BaseEstimator, RegressorMixin):
    """one GP per output (outputs standardised); ARD Matern-5/2 + white noise, hyperparameters by marginal likelihood"""
    def __init__(self, nu=2.5):
        self.nu = nu

    def fit(self, X, y):
        self.m_ = []
        for j in range(y.shape[1]):
            k = CK(1.0) * Matern(length_scale=[0.3, 0.3], length_scale_bounds=(1e-2, 1e2), nu=self.nu) \
                + WhiteKernel(1e-6, (1e-12, 1e-1))
            self.m_.append(GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1,
                                                    random_state=SEED).fit(X, y[:, j]))
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.m_])


def methods():
    """name -> (estimator, inner grid).  Grids are deliberately small (budget rule)."""
    return {
        "poly": (make_pipeline(PolynomialFeatures(3), Ridge(1e-8)),
                 {"polynomialfeatures__degree": [2, 3, 4, 5, 6], "ridge__alpha": [1e-10, 1e-6, 1e-3]}),
        "knn": (KNeighborsRegressor(), {"n_neighbors": [2, 4, 8, 16], "weights": ["uniform", "distance"]}),
        "rf": (RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=1),
               {"min_samples_leaf": [1, 2, 4], "max_features": [1.0, 0.5]}),
        "et": (ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=1),
               {"min_samples_leaf": [1, 2, 4], "max_features": [1.0, 0.5]}),
        "xgb": (XGB(), {"max_depth": [3, 5], "learning_rate": [0.03, 0.1]}),
        "gpr": (GPR(), {}),                      # kernel hyperparameters by marginal likelihood, no inner grid
        "rbf": (RBF(), {"kernel": ["thin_plate_spline", "cubic", "quintic"], "smoothing": [0.0, 1e-6, 1e-3]}),
        "mlp": (make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                                                             max_iter=4000, tol=1e-7, random_state=SEED)),
                {"mlpregressor__alpha": [1e-6, 1e-4], "mlpregressor__hidden_layer_sizes": [(64, 64), (128, 128, 128)]}),
    }


# ------------------------------------------------------------------------------------------------ PCA + model
class Surrogate:
    def __init__(self, method, kind, K, params=None):
        self.method, self.kind, self.K, self.params = method, kind, K, params or {}

    def fit(self, keys, A, P, inner=True):
        t = time.time()
        self.mA, self.mP = A.mean(0), P.mean(0)
        _, _, VA = np.linalg.svd(A - self.mA, full_matrices=False); _, _, VP = np.linalg.svd(P - self.mP, full_matrices=False)
        self.VA, self.VP = VA[:self.K], VP[:self.K]
        Y = np.c_[(A - self.mA) @ self.VA.T, (P - self.mP) @ self.VP.T]
        self.ys = Y.std(0) + 1e-30; Yn = Y / self.ys
        X = feats(keys, self.kind)
        est, grid = methods()[self.method]
        if inner and not self.params and grid:
            g = GridSearchCV(est, grid, cv=KFold(4, shuffle=True, random_state=SEED), scoring="neg_mean_squared_error",
                             n_jobs=1).fit(X, Yn)
            self.params = g.best_params_
        self.est = est.set_params(**self.params).fit(X, Yn)
        self.train_s = time.time() - t
        return self

    def predict(self, keys):
        Y = self.est.predict(feats(np.atleast_2d(keys), self.kind)) * self.ys
        return self.mA + Y[:, :self.K] @ self.VA, self.mP + Y[:, self.K:] @ self.VP


# ------------------------------------------------------------------------------------------------ scoring (parallel)
def _score(a):
    k, dA, dP = a
    f = {"direct": D.score, "ab": AB.score, "abtau": AB.score_tau}[TARGET]
    return f(tuple(k), dA, dP)


def score_keys(pool, sur, keys):
    A, P = sur.predict(np.asarray(keys))
    return np.array(pool.map(_score, [(tuple(k), A[i], P[i]) for i, k in enumerate(keys)], chunksize=2))


def load_train():
    if TARGET == "ab":
        d = np.load(RES / "ab_targets.npz"); return d["keys"], d["lna"], d["lnb"]
    if TARGET == "abtau":                  # regress the clock tau(s) - s instead of ln beta (beta acts through int beta)
        d = np.load(RES / "ab_targets.npz"); return d["keys"], d["lna"], d["tau"]
    d = np.load(RES / "direct_targets.npz")
    return d["keys"], d["dlnA"], d["dphi"]


KINDS = ["q", "nu", "invq", "logq"]
KS = [6, 12, 20]


def _fit_job(a):
    m, kind, K, tr, keys, A, P = a
    return Surrogate(m, kind, K).fit(keys[tr], A[tr], P[tr])


def _pred_job(a):
    sur, ktest = a
    A, P = sur.predict(ktest); return A, P


def cv(nproc, meths):
    """all (method, features, K, fold) fits in one parallel map, then all waveform scores in one map"""
    keys, A, P = load_train()
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(keys))
    path = RES / f"{TARGET}_cv.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    tags = [f"{m}|{kind}|{K}" for m in meths for kind in KINDS for K in KS]
    tags = [t for t in tags if t not in out]
    jobs = [(t, f) for t in tags for f in range(len(folds))]
    t0 = time.time()
    with get_context("fork").Pool(nproc) as pool:
        surs = pool.map(_fit_job, [(t.split("|")[0], t.split("|")[1], int(t.split("|")[2]), folds[f][0], keys, A, P)
                                   for t, f in jobs], chunksize=1)
        print(f"fits: {len(jobs)} in {time.time() - t0:.0f} s", flush=True)
        sjobs = []
        for (t, f), sur in zip(jobs, surs):
            te = folds[f][1]; Ap, Pp = sur.predict(keys[te])
            sjobs += [(t, i, (tuple(keys[i]), Ap[j], Pp[j])) for j, i in enumerate(te)]
        es = pool.map(_score, [j[2] for j in sjobs], chunksize=8)
    print(f"scores: {len(sjobs)} in {time.time() - t0:.0f} s", flush=True)
    for t in tags:
        e = np.empty(len(keys))
        for (tt, i, _), v in zip(sjobs, es):
            if tt == t:
                e[i] = v
        ss = [s for (tt, _), s in zip(jobs, surs) if tt == t]
        out[t] = dict(C.stats(e), params=[str(s.params) for s in ss], train_s=float(np.mean([s.train_s for s in ss])))
        print(f"{t:22s} CV E {C.fmt(C.stats(e))}  train {out[t]['train_s']:.1f}s", flush=True)
    path.write_text(json.dumps(out, indent=1))


def best_configs():
    cvr = json.loads((RES / f"{TARGET}_cv.json").read_text())
    best = {}
    for tag, r in cvr.items():
        m = tag.split("|")[0]
        if m not in best or r["mean"] < cvr[best[m]]["mean"]:          # selection on CV mean E (training pool only)
            best[m] = tag
    return best


def final(nproc):
    keys, A, P = load_train()
    V = np.array(C.V300)
    out = {}
    with get_context("fork").Pool(nproc) as pool:
        for m, tag in best_configs().items():
            _, kind, K = tag.split("|")
            sur = Surrogate(m, kind, int(K)).fit(keys, A, P)
            t = time.time(); sur.predict(V); pred_s = (time.time() - t) / len(V)
            e = score_keys(pool, sur, V)
            out[m] = dict(config=tag, params=str(sur.params), V300=C.stats(e), V300_cases=e.tolist(),
                          train_s=sur.train_s, predict_s_per_case=pred_s, n_train=len(keys))
            pickle.dump(sur, open(MOD / f"{TARGET}_{m}.pkl", "wb"))
            print(f"{m:5s} {tag:20s} V300 {C.fmt(C.stats(e))}  train {sur.train_s:.1f}s  predict {pred_s*1e3:.2f} ms/case", flush=True)
    (RES / f"{TARGET}_final.json").write_text(json.dumps(out, indent=1))


def curve(nproc, sizes=(48, 96, 192, 384)):
    keys, A, P = load_train()
    V = np.array(C.V300); r = np.random.default_rng(SEED); perm = r.permutation(len(keys))
    out = {}
    with get_context("fork").Pool(nproc) as pool:
        for m, tag in best_configs().items():
            _, kind, K = tag.split("|")
            for n in list(sizes) + [len(keys)]:
                idx = perm[:n]
                sur = Surrogate(m, kind, min(int(K), n - 1)).fit(keys[idx], A[idx], P[idx])
                e = score_keys(pool, sur, V)
                out[f"{m}|{n}"] = C.stats(e)
                print(f"{m:5s} n={n:4d} V300 {C.fmt(C.stats(e))}", flush=True)
    (RES / f"{TARGET}_curve.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cv", "final", "curve"])
    ap.add_argument("--nproc", type=int, default=128)
    ap.add_argument("--methods", default="poly,knn,rf,et,xgb,gpr,rbf,mlp")
    ap.add_argument("--target", default="direct", choices=["direct", "ab", "abtau"])
    ap.add_argument("--ks", default="6,12,20")
    a = ap.parse_args(); TARGET = a.target; KS = [int(x) for x in a.ks.split(",")]
    {"cv": lambda: cv(a.nproc, a.methods.split(",")), "final": lambda: final(a.nproc),
     "curve": lambda: curve(a.nproc)}[a.cmd]()
