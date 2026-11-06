"""Paradigm E for the classic regressors: KNN, RF, ExtraTrees, XGBoost, GPR, RBF and sparse regression as global
alpha(t; nu, chi1), beta(t; nu, chi1), refined THROUGH E on T96 (same loss and t0 handling as efit.py / the hand forms).

Time: ln alpha and ln(beta / X1) are carried as values on the efit time grid S_C (K = 1900 points, uniform in log time
to merger, then uniform in t post-merger), so each method supplies 2K numbers c(nu, chi1) -- curves, combined across
parameter space by the method.  (A smooth basis was tried: 300 cubic B-splines already cost E ~ 2e-4 by projection,
because beta errors integrate; the grid costs 4e-6.)  Every method's prediction is LINEAR in a parameter matrix W once its structure is fixed:

    knn    c(x) = sum_i w_i(x) W_i            w: inverse-distance weights of the k nearest T96 points; W: per training point
    rbf    c(x) = sum_i l_i(x) W_i                        thin-plate spline + linear polynomial in cardinal form (l_i:
                                                         Lagrange functions of the T96 points; W: values there)
    gpr    c(x) = m + k(x, X) (K + s^2 I)^-1 (W - m)     posterior mean; ARD Matern-5/2 + white kernel by marginal likelihood
    rf/et  c(x) = (1/T) sum_trees W[leaf_t(x)]           forest structure; W: leaf values
    xgb    c(x) = b + leaf-indicator ridge on the boosted structure, in dual form; W: values at the T96 points
    sparse c(x) = sum_j m_j(x) W_j                       library m_j = Legendre products P_i(nu) P_j(chi1), i + j <= 4
                                                         (scaled features); dense fit through E, then GROUP pruning (a
                                                         term's whole alpha or beta curve, norm < 5 % of the largest)
                                                         and refit.  (First try: elementwise L1 + pruning -> E 0.22, the
                                                         penalty dominated E and pruning single grid values broke beta.)

Structure and starting W come from an ordinary fit to the model-free equal-phase alpha, beta of the T96 cases
(results/ab_targets.npz; allowed as an initialisation).  Then W (only) is optimised to minimise the T96 mean E.
Nothing uses V300; V300 is scored once at the end with common.score_ab.

    python efit_lin.py --methods knn rbf gpr rf et xgb sparse --minutes 8      -> results/efit_lin_<method>.json
"""
from __future__ import annotations
import argparse, json, time, warnings
from multiprocessing import get_context

import numpy as np
import common as C
import efit as F
import torch                                                   # noqa: E402

warnings.filterwarnings("ignore")
K = len(F.S_C)                     # values on the efit time grid (no basis: a 300-function B-spline basis already
                                   # costs E ~ 2e-4 by projection, the grid itself 4e-6)


def init_targets(keys):
    """model-free ln alpha, ln(beta/X1) of each key on S_C -> (n, 2K)"""
    import ab_targets as AB
    z = np.load(C.HERE / "results" / "ab_targets.npz"); d = {x: z[x] for x in ("keys", "lna", "tau")}
    idx = {tuple(np.round(k, 6)): i for i, k in enumerate(d["keys"])}
    out = []
    for k in keys:
        i = idx[tuple(np.round(k, 6))]; X1 = k[0] / (1 + k[0])
        b = np.clip(1.0 + np.gradient(d["tau"][i], AB.S_GRID), 1e-3, None)
        out.append(np.r_[np.interp(F.S_C, AB.S_GRID, d["lna"][i]), np.log(np.interp(F.S_C, AB.S_GRID, b) / X1)])
    return np.array(out)


class LinModel(torch.nn.Module):
    """c(x) = offset + design(x) @ (W - shift); values on S_C (efit model interface)"""
    def __init__(self, design, W0, offset=None, shift=None):
        super().__init__()
        self.design = design; self.W = torch.nn.Parameter(torch.tensor(W0))
        self.register_buffer("offset", torch.tensor(np.zeros(W0.shape[1]) if offset is None else offset))
        self.register_buffer("shift", torch.tensor(np.zeros(W0.shape[1]) if shift is None else shift))
        self.register_buffer("mask", torch.ones_like(self.W)); self._cache = {}

    def A(self, p):
        key = p.detach().numpy().tobytes()
        if key not in self._cache:
            self._cache[key] = torch.tensor(self.design(p.detach().numpy()))
        return self._cache[key]

    def forward(self, p):
        c = self.offset + self.A(p) @ (self.W * self.mask - self.shift)          # (B, 2K)
        return torch.stack([c[:, :K], c[:, K:]], -1)


def build(method, X, Y):
    """structure + starting W from the initial fit of Y (n, 2K) on features X (n, 2); returns LinModel, info"""
    n = len(X)
    if method == "knn":
        from sklearn.neighbors import NearestNeighbors
        k = 6; nn = NearestNeighbors(n_neighbors=k).fit(X)
        def design(Z):
            d, j = nn.kneighbors(Z); w = 1.0 / np.maximum(d, 1e-9); w[d < 1e-9] = 1e9; w /= w.sum(1, keepdims=True)
            A = np.zeros((len(Z), n)); np.put_along_axis(A, j, w, 1); return A
        return LinModel(design, Y.copy()), dict(k=k)
    if method == "rbf":
        phi = lambda r: np.where(r > 0, r ** 2 * np.log(np.maximum(r, 1e-300)), 0.0)
        P = lambda Z: np.c_[np.ones(len(Z)), Z]
        M = np.block([[phi(np.linalg.norm(X[:, None] - X[None], axis=-1)), P(X)], [P(X).T, np.zeros((3, 3))]])
        # parametrised by its VALUES at the training points (cardinal form): the same interpolant, but well conditioned
        # for the E refinement (refining the raw thin-plate weights diverged: T96 E 3e-7 -> 0.1, job 3545746)
        Minv = np.linalg.inv(M)[:, :n]
        design = lambda Z: np.c_[phi(np.linalg.norm(Z[:, None] - X[None], axis=-1)), P(Z)] @ Minv
        return LinModel(design, Y.copy()), dict(kernel="thin_plate_spline", degree=1, param="values at T96 points")
    if method == "gpr":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        m = Y.mean(0); s = Y.std(0) + 1e-12
        g = GaussianProcessRegressor(ConstantKernel(1.0) * Matern([0.5, 0.5], nu=2.5) + WhiteKernel(1e-4),
                                     normalize_y=False, n_restarts_optimizer=3, random_state=0).fit(X, (Y - m) / s)
        kern = g.kernel_; Kinv = np.linalg.inv(kern(X))
        design = lambda Z: kern.k1(Z, X) @ Kinv                   # k1: the noise-free part (Constant * Matern)
        return LinModel(design, Y.copy(), offset=m, shift=m), dict(kernel=str(kern))
    if method in ("rf", "et"):
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        cls = RandomForestRegressor if method == "rf" else ExtraTreesRegressor
        f = cls(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=8).fit(X, Y)
        leaves, vals = [], []
        for t in f.estimators_:
            lid = np.where(t.tree_.children_left == -1)[0]; leaves.append({l: len(vals) + i for i, l in enumerate(lid)})
            vals += [t.tree_.value[l][:, 0] for l in lid]
        T = len(f.estimators_)
        def design(Z):
            A = np.zeros((len(Z), len(vals)))
            for t, mp in zip(f.estimators_, leaves):
                for r, l in enumerate(t.apply(Z)):
                    A[r, mp[l]] += 1.0 / T
            return A
        return LinModel(design, np.array(vals)), dict(n_leaves=len(vals))
    if method == "xgb":
        import xgboost as xgb
        b = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, tree_method="hist",
                             multi_strategy="multi_output_tree", random_state=0).fit(X, Y)
        bo = b.get_booster(); L = bo.predict(xgb.DMatrix(X), pred_leaf=True).astype(int)       # (n, trees)
        cols = {}
        for t in range(L.shape[1]):
            for l in np.unique(L[:, t]):
                cols[(t, l)] = len(cols)
        def design(Z):
            Lz = bo.predict(xgb.DMatrix(Z), pred_leaf=True).astype(int); A = np.zeros((len(Z), len(cols)))
            for r in range(len(Z)):
                for t in range(Lz.shape[1]):
                    j = cols.get((t, Lz[r, t]))
                    if j is not None:
                        A[r, j] = 1.0
            return A
        # leaf values = ridge least squares on the boosted structure (the booster's own vector leaves are not exposed),
        # written in its dual / cardinal form c(x) = b0 + A(x) A^T (A A^T + l I)^-1 (W - b0), W = values at the T96 points:
        # identical predictions, but refining the 2200 raw (collinear) leaf values through E diverged (T96 E -> 0.39, job 3545746)
        A = design(X); b0 = Y.mean(0); G = A.T @ np.linalg.inv(A @ A.T + 1e-3 * np.eye(n))
        return LinModel(lambda Z: design(Z) @ G, Y.copy(), offset=b0, shift=b0), dict(n_leaves=len(cols), param="values at T96 points")
    if method == "sparse":
        from itertools import product
        from numpy.polynomial import legendre as LG
        pw = [(i, j) for i, j in product(range(5), range(5)) if i + j <= 4]
        Pl = lambda x, i: LG.legval(x, np.eye(5)[i])
        design = lambda Z: np.column_stack([Pl(Z[:, 0], i) * Pl(Z[:, 1], j) for i, j in pw])
        A = design(X); W0 = np.linalg.lstsq(A, Y, rcond=None)[0]
        return LinModel(design, W0), dict(library=[f"P{i}(nu) P{j}(chi)" for i, j in pw])
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["knn", "rbf", "gpr", "rf", "et", "xgb", "sparse"])
    ap.add_argument("--minutes", type=float, default=8); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--nproc", type=int, default=96)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args(); torch.set_num_threads(a.threads); log = lambda s: print(s, flush=True)
    keys = list(C.T96); X = F.pfeat(keys).numpy(); Y = init_targets(keys)
    with get_context("fork").Pool(a.nproc) as pool:
        # representation floor: the model-free targets themselves through the basis (per case, no regression)
        fl = F.evaluate(LinModel(lambda Z: np.eye(len(keys)) if len(Z) == len(keys) else None, Y), pool, {"T96": keys})["T96"]
        log(f"basis floor (T96 own targets, K={K}): {C.fmt(fl)}")
        for m in a.methods:
            t0 = time.time(); model, info = build(m, X, Y); t_init = time.time() - t0
            e_init = F.evaluate(model, pool, {"T96": keys})["T96"]
            log(f"{m}: {model.W.numel()} values, init (regression on model-free targets) T96 {C.fmt(e_init)}")
            if m == "sparse":
                # group sparsity (SINDy-style): a library term's whole curve for alpha (or beta) is kept or dropped
                tr = F.train(model, keys, a.minutes / 2, pool, log, lr=a.lr, tag=f"{m}-dense")
                info["dense"] = F.evaluate(model, pool, {"T96": keys, "V300": list(C.V300)})
                log(f"sparse dense ({len(info['library'])} terms x 2): T96 {C.fmt(info['dense']['T96'])}   V300 {C.fmt(info['dense']['V300'])}")
                with torch.no_grad():
                    for sl in (slice(0, K), slice(K, 2 * K)):
                        g = model.W[:, sl].norm(dim=1); keep = (g >= 0.05 * g.max()).double()
                        model.mask[:, sl] = keep[:, None]
                kept = model.mask[:, 0].sum().item(), model.mask[:, K].sum().item()
                info["terms_kept_alpha_beta"] = [int(kept[0]), int(kept[1])]
                info["n_nonzero"] = int(model.mask.sum().item())
                log(f"sparse: kept {int(kept[0])} alpha terms, {int(kept[1])} beta terms of {len(info['library'])}")
                tr2 = F.train(model, keys, a.minutes / 2, pool, log, lr=a.lr, tag=f"{m}-pruned")
                tr = dict(steps=tr["steps"] + tr2["steps"], train_s=tr["train_s"] + tr2["train_s"], hist=tr["hist"] + tr2["hist"])
            else:
                tr = F.train(model, keys, a.minutes, pool, log, lr=a.lr, tag=m)
            ev = F.evaluate(model, pool, {"T96": keys, "V300": list(C.V300)})
            log(f"{m}: T96 {C.fmt(ev['T96'])}   V300 {C.fmt(ev['V300'])}")
            (F.RES / f"efit_lin_{m}.json").write_text(json.dumps(dict(method=m, K=K, n_values=int(model.W.numel()), init_s=t_init,
                                                                      init_T96=e_init, floor_T96=fl, **info, **tr, **ev), indent=1))
            torch.save(model.state_dict(), F.MOD / f"efit_lin_{m}.pt")


if __name__ == "__main__":
    main()
