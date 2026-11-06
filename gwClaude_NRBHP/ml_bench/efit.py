"""Paradigm E: ML alpha(t; q, chi1), beta(t; q, chi1) fitted THROUGH THE MAP, the way the exploration forms are fitted.

Same map and loss as the hand models: tau = t0 + int beta dt (anchored at t = S.T_ANCHOR), h_model(tau) = alpha h_pp(t),
E = ||h_NR - e^{i phi0} h_model||^2 / (2 ||h_NR||^2) on the NR samples of the overlap, phi0 analytic, t0 per case;
objective = mean E over the training cases at fit stride (as training_size/fit_*.py).  Only the function class changes:

    mlp  pointwise network  (u(s), nu, chi1) -> (ln alpha, ln beta / X1)        black box, alpha/beta as functions of time
    gru  bidirectional GRU over the time grid, input (u(s), nu, chi1) per step   black box, alpha/beta as a time series
    sr   PySR expressions for ln alpha, ln(beta / X1) in (u, nu, chi1), distilled from the trained mlp, constants refitted
         through E (functional form)

s = t - t_peak,pp [m1]; u = warp(s) (log time to merger, nn_direct.warp).  beta = X1 exp(.) : X1 = q/(1+q) is the m1 -> M
unit conversion, so a zero network is the identity map (alpha = 1, beta = X1) -- the day-1 start, no physics put in.
Networks output on a grid S_C (uniform in u, dense at merger; floor below) interpolated linearly onto the fit samples.
t0: a free parameter per training case, trained jointly (gradient) and re-seeded every --reseed steps by the scorer's own
grid search (common._t0_err) so the fit can jump between phase cycles.  Final numbers: common.score_ab (gate.py-verified),
independent of the torch loss.  V300 score-only; nothing selected on it.

    python efit.py gate                                   torch E == fit_pn_anchored_spin._err_at_t0 (pass 40 alpha, beta)
    python efit.py train --arch mlp --minutes 25          -> models/efit_<arch>_<train>.pt, results/efit_<arch>_<train>.json
    python efit.py train --arch mlp --init targets --pre-minutes 5 --minutes 25   -> efit_mlp_T96_tinit.*  (pre-fit to the
                                                          model-free alpha, beta first, then the same E training)
    python efit.py sr --minutes 12                        -> results/efit_sr_T96.json (efit_sr.py; needs efit_mlp_T96.pt)
"""
from __future__ import annotations
import argparse, json, time
from multiprocessing import get_context

import numpy as np
import common as C
from direct_targets import _tpeak
from nn_direct import warp, nu_feat
import torch                                                   # noqa: E402  (after numpy)

torch.set_default_dtype(torch.float64)
RES = C.HERE / "results"; MOD = C.HERE / "models"
T_MIN = -8000.0                                                # m1; covers the NR window [-5000, 100] M for every q >= 2
# 1500 points uniform in warp(s) on [T_MIN, 0) + 400 uniform on [0, 130].  Representation floor (model-free alpha, beta
# of 8 T96 cases linearly interpolated from this grid): E median 4.1e-6, max 3.9e-5 (600 points: 1.2e-4 / 5.2e-4, the
# post-merger resolution dominates)
S_C = np.r_[-20.0 * np.expm1(-np.linspace(-1.0, 0.0, 1500, endpoint=False) * np.log1p(-T_MIN / 20.0)),
            np.linspace(0.0, 130.0, 400)]
S_C = np.sort(S_C); S_C[0] = min(S_C[0], T_MIN)
U_C = warp(S_C)


# ------------------------------------------------------------------------------------------------ data
def _lin(x, xp):
    """indices / weights of linear interpolation of grid xp at points x (clamped)"""
    j = np.clip(np.searchsorted(xp, x), 1, len(xp) - 1); w = np.clip((x - xp[j - 1]) / (xp[j] - xp[j - 1]), 0, 1)
    return torch.tensor(j), torch.tensor(w)


def prep(k, key="fit"):
    pre = "fit_" if key == "fit" else ""
    c = C.case(k); t, h = c[pre + "t_bhpt"], c[pre + "h_bhpt"]
    tpk = _tpeak(c["t_bhpt"], c["h_bhpt"]); m = t - tpk >= T_MIN
    t, h, s = t[m], h[m], t[m] - tpk
    j, w = _lin(s, S_C)
    ja, wa = _lin(np.array([C.S.T_ANCHOR]), t)
    tn, hn = c[pre + "t_nr"], c[pre + "h_nr"]
    return dict(k=k, X1=k[0] / (1 + k[0]), t=torch.tensor(t), h=torch.tensor(h), j=j, w=w, ja=ja, wa=wa,
                tn=torch.tensor(tn), hn=torch.tensor(hn), n1=float(np.sum(np.abs(hn) ** 2)))


def _interp_grid(v, d):                    # v on S_C -> fit samples
    return v[d["j"] - 1] * (1 - d["w"]) + v[d["j"]] * d["w"]


def clock(beta, d):
    t = d["t"]; bc = torch.cat([torch.zeros(1), torch.cumsum(0.5 * (beta[1:] + beta[:-1]) * (t[1:] - t[:-1]), 0)])
    ba = bc[d["ja"] - 1] * (1 - d["wa"]) + bc[d["ja"]] * d["wa"]
    return bc - ba


def E_torch(alpha, ts, d, t0):
    """fit_pn_anchored_spin._err_at_t0 in torch (alpha, ts on the fit samples of d)"""
    tau = t0 + ts; tn, hn = d["tn"], d["hn"]
    lo, hi = torch.maximum(tn[0], tau[0]), torch.minimum(tn[-1], tau[-1])
    m = (tn >= lo) & (tn <= hi)
    if m.sum().item() / len(tn) < C.S.H.MIN_COVERAGE:
        return None
    tc, r = tn[m], hn[m]
    j = torch.clamp(torch.searchsorted(tau.detach(), tc), 1, len(tau) - 1)
    w = (tc - tau[j - 1]) / (tau[j] - tau[j - 1])
    g = d["h"][j - 1] * (1 - w) + d["h"][j] * w
    gg = (alpha[j - 1] * (1 - w) + alpha[j] * w) * g
    n1 = torch.sum(torch.abs(r) ** 2); n2 = torch.sum(torch.abs(gg) ** 2); Z = torch.abs(torch.sum(r * torch.conj(gg)))
    return (n1 + n2 - 2 * Z) / (2 * n1)


# ------------------------------------------------------------------------------------------------ models
def pfeat(keys):
    return torch.tensor(nu_feat(np.atleast_2d(np.asarray(keys, float))))


class MLP(torch.nn.Module):
    def __init__(self, nf=16, width=128, depth=4):
        super().__init__()
        self.register_buffer("freq", torch.tensor(np.pi * 2.0 ** np.linspace(0, 6, nf)))
        layers, d = [], 2 + 1 + 2 * nf
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]; d = width
        self.net = torch.nn.Sequential(*layers); self.head = torch.nn.Linear(d, 2)
        torch.nn.init.zeros_(self.head.weight); torch.nn.init.zeros_(self.head.bias)     # identity map at start

    def forward(self, p):                  # p (B, 2) -> (B, len(S_C), 2)
        u = torch.tensor(U_C)[None, :, None].expand(len(p), -1, 1); z = u * self.freq
        x = torch.cat([p[:, None, :].expand(-1, len(U_C), -1), u, torch.sin(z), torch.cos(z)], -1)
        return self.head(self.net(x))


class GRU(torch.nn.Module):
    def __init__(self, hidden=64, layers=2, stride=2):
        super().__init__()
        self.stride = stride; self.idx = np.arange(0, len(U_C), stride)
        if self.idx[-1] != len(U_C) - 1:
            self.idx = np.r_[self.idx, len(U_C) - 1]
        self.rnn = torch.nn.GRU(3, hidden, layers, batch_first=True, bidirectional=True)
        self.head = torch.nn.Linear(2 * hidden, 2)
        torch.nn.init.zeros_(self.head.weight); torch.nn.init.zeros_(self.head.bias)

    def forward(self, p):
        u = torch.tensor(U_C[self.idx])[None, :, None].expand(len(p), -1, 1)
        y = self.head(self.rnn(torch.cat([u, p[:, None, :].expand(-1, len(self.idx), -1)], -1))[0])
        # linear interpolation from the GRU sub-grid back to S_C
        j, w = _lin(U_C, U_C[self.idx]); w = w[None, :, None]
        return y[:, j - 1] * (1 - w) + y[:, j] * w


def ab_grid(model, keys):
    """alpha, beta on S_C for each key, (B, len(S_C)) each"""
    y = model(pfeat(keys)); X1 = torch.tensor([k[0] / (1 + k[0]) for k in keys])[:, None]
    return torch.exp(y[..., 0]), X1 * torch.exp(y[..., 1])


def ab_numpy(model, keys):
    with torch.no_grad():
        a, b = ab_grid(model, keys)
    return a.numpy(), b.numpy()


def ab_fun_from(aC, bC, k, c):
    """common.score_ab interface from alpha, beta on S_C (held at the grid ends outside it)"""
    tpk = _tpeak(c["t_bhpt"], c["h_bhpt"])
    def f(key):
        s = c[("fit_" if key == "fit" else "") + "t_bhpt"] - tpk
        return np.interp(s, S_C, aC), np.interp(s, S_C, bC)
    return f


def _score(a):
    k, aC, bC = a; c = C.case(k)
    return C.score_ab(ab_fun_from(aC, bC, k, c), c)


def _seed(a):
    k, aC, bC = a; c = C.case(k)
    al, be = ab_fun_from(aC, bC, k, c)("fit")
    return C.score_ab_key(al, be, c, "fit")                      # (E, t0) with the full grid search


# ------------------------------------------------------------------------------------------------ training
def train(model, keys, minutes, pool, log, lr=2e-3, reseed=150, batch=None, tag="", penalty=None):
    """mean E over keys (+ penalty(model) if given, e.g. an L1 term for sparse regression)"""
    D = [prep(k) for k in keys]; n = len(D)
    t0 = torch.zeros(n, requires_grad=True)
    def reseed_all():
        aC, bC = ab_numpy(model, keys)
        r = pool.map(_seed, [(keys[i], aC[i], bC[i]) for i in range(n)], chunksize=1)
        with torch.no_grad():
            for i, (e, x0) in enumerate(r):
                cur = E_torch(_interp_grid(torch.tensor(aC[i]), D[i]), clock(_interp_grid(torch.tensor(bC[i]), D[i]), D[i]), D[i], t0[i])
                if cur is None or e < cur.item():
                    t0[i] = x0
        return float(np.mean([e for e, _ in r]))
    e0 = reseed_all(); log(f"{tag} start: mean E {e0:.4e}")
    opt = torch.optim.Adam([dict(params=model.parameters(), lr=lr), dict(params=[t0], lr=0.05)])
    T = minutes * 60; tstart = time.time(); step = 0; hist = []; scale = 1.0; rejected = 0
    rng = np.random.default_rng(C.SEED if hasattr(C, "SEED") else 20261031)
    # step rejection: a divergent step can push every case's tau out of the NR window (no E, no gradient) -> restore the
    # last good snapshot and halve the learning rate
    snap = lambda: ({k: v.detach().clone() for k, v in model.state_dict().items()}, t0.detach().clone())
    good, ref = snap(), e0
    while time.time() - tstart < T:
        frac = (time.time() - tstart) / T
        for g, base in zip(opt.param_groups, (lr, 0.05)):
            g["lr"] = scale * base * 0.5 * (1 + np.cos(np.pi * min(frac, 1.0)))
        idx = np.arange(n) if batch is None else rng.choice(n, batch, replace=False)
        aG, bG = ab_grid(model, [keys[i] for i in idx])
        es = []
        for b, i in enumerate(idx):
            e = E_torch(_interp_grid(aG[b], D[i]), clock(_interp_grid(bG[b], D[i]), D[i]), D[i], t0[i])
            es.append(e if e is not None else torch.tensor(1.0))
        loss = torch.stack(es).mean()
        if penalty is not None:
            loss = loss + penalty(model)
        if not loss.requires_grad or not torch.isfinite(loss) or loss.item() > max(3 * ref, ref + 0.05):
            with torch.no_grad():
                model.load_state_dict(good[0]); t0.copy_(good[1])
            scale *= 0.5; rejected += 1; opt.state.clear()
            if rejected <= 5 or rejected % 20 == 0:
                log(f"{tag} step {step}: rejected (loss {loss.item():.3e} vs {ref:.3e}), lr scale -> {scale:.3g}")
            if scale < 1e-4:
                log(f"{tag}: lr scale below 1e-4, stopping"); break
            continue
        if step % 25 == 0 and loss.item() < 1.2 * ref:
            good, ref = snap(), loss.item()
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 100 == 0:
            hist.append((step, time.time() - tstart, loss.item())); log(f"{tag} step {step:5d}  {time.time() - tstart:6.0f}s  mean E {loss.item():.4e}")
        if step % reseed == 0 and frac < 0.9:
            e = reseed_all()
            if step % (4 * reseed) == 0:
                log(f"{tag} reseed t0: grid-search mean E {e:.4e}")
    return dict(steps=step, train_s=time.time() - tstart, hist=hist, rejected=rejected)


def evaluate(model, pool, sets):
    out = {}
    for name, keys in sets.items():
        aC, bC = ab_numpy(model, keys)
        e = np.array(pool.map(_score, [(keys[i], aC[i], bC[i]) for i in range(len(keys))], chunksize=1))
        out[name] = dict(C.stats(e), cases=e.tolist())
    return out


def train_keys(which):
    if which == "T96":
        return list(C.T96)
    import nn_direct as N                                        # the same 572 training cases as the direct MLP
    d = N.load(); return [tuple(k) for k in d["keys"][d["tr"]]]


# ------------------------------------------------------------------------------------------------ commands
def gate():
    import gate as G
    worst = 0.0
    for k in [C.T96[i] for i in (0, 30, 60, 95)]:
        c = C.case(k); al, be = G.ab_fun(k, c)("fit"); d = prep(k)
        tpk = _tpeak(c["t_bhpt"], c["h_bhpt"]); m = c["fit_t_bhpt"] - tpk >= T_MIN
        e_np, t0 = C.score_ab_key(al, be, c, "fit")
        e_t = E_torch(torch.tensor(al[m]), clock(torch.tensor(be[m]), d), d, torch.tensor(t0)).item()
        worst = max(worst, abs(e_t / e_np - 1)); print(f"{k}: numpy {e_np:.6e}  torch {e_t:.6e}  ratio {e_t / e_np:.6f}")
    print("efit gate worst |ratio-1|", f"{worst:.1e}", "PASS" if worst < 1e-4 else "FAIL")


def pretrain(model, keys, minutes, log, lr=2e-3):
    """fit the network to the model-free equal-phase ln alpha, ln(beta/X1) of keys on S_C (Huber loss) -- an allowed
    initialisation; the E training that follows decides the result"""
    import efit_lin as L
    Y = torch.tensor(L.init_targets(keys)); K = len(S_C); p = pfeat(keys)
    opt = torch.optim.Adam(model.parameters(), lr=lr); t = time.time(); step = 0
    while time.time() - t < minutes * 60:
        y = model(p); loss = torch.nn.functional.huber_loss(torch.cat([y[..., 0], y[..., 1]], 1), Y, delta=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 200 == 0:
            log(f"pretrain step {step}  {time.time() - t:.0f}s  huber {loss.item():.3e}")
    return dict(pre_steps=step, pre_s=time.time() - t)


def main_train(a):
    torch.set_num_threads(a.threads); log = lambda s: print(s, flush=True)
    keys = train_keys(a.train); tag = f"{a.arch}_{a.train}" + ("_tinit" if a.init == "targets" else "")
    torch.manual_seed(20261031)
    model = MLP() if a.arch == "mlp" else GRU()
    npar = sum(p.numel() for p in model.parameters()); log(f"{a.arch} {npar} parameters, {len(keys)} training cases, init {a.init}")
    pre = pretrain(model, keys, a.pre_minutes, log) if a.init == "targets" else {}
    with get_context("fork").Pool(a.nproc) as pool:
        info = train(model, keys, a.minutes, pool, log, lr=a.lr, batch=a.batch, tag=a.arch)
        torch.save(model.state_dict(), MOD / f"efit_{tag}.pt")
        ev = evaluate(model, pool, {"train": keys if a.train == "T96" else list(C.T96), "V300": list(C.V300)})
    for kk, v in ev.items():
        log(f"{a.arch} ({tag}) {kk}: {C.fmt(v)}")
    (RES / f"efit_{tag}.json").write_text(json.dumps(dict(arch=a.arch, train_set=a.train, init=a.init, n_params=npar,
                                                          **pre, **info, **ev), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gate", "train", "sr"])
    ap.add_argument("--arch", default="mlp", choices=["mlp", "gru"]); ap.add_argument("--train", default="T96", choices=["T96", "pool"])
    ap.add_argument("--minutes", type=float, default=25); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--init", default="identity", choices=["identity", "targets"]); ap.add_argument("--pre-minutes", type=float, default=5)
    ap.add_argument("--src", default="mlp_T96", help="sr: which trained mlp to distil (models/efit_<src>.pt)")
    ap.add_argument("--refit-minutes", type=float, default=5); ap.add_argument("--maxsize", type=int, default=35)
    ap.add_argument("--procs", type=int, default=120)
    ap.add_argument("--nproc", type=int, default=96); ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "gate":
        gate()
    elif a.cmd == "train":
        main_train(a)
    else:
        import efit_sr; efit_sr.main(a)
