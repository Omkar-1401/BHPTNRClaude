"""Paradigm C: neural networks on the direct amplitude/phase corrections (direct_targets.py), trained end to end on the
curves (Zhong+ 2503.04534 style: MSE on amplitude/phase, Adam).

    mlp  pointwise:  (nu, chi1, Fourier features of the warped time u(s)) -> (dlnA, dphi)          ['direct regression']
    cfc  sequence:   CfC (ncps, AutoNCP wiring as Zhong+) reading [ln|h_pp|, dphi_pp/ds, nu, chi1, ds] along s -> (dlnA, dphi)
Targets are standardised residuals from the training-mean curve.  Loss variants: 'mse' (plain, as Zhong+) or 'w' (weighted
by the mean NR power density |h_NR(s)|^2, which is what E weights).  Split: 80 / 20 of the training pool (fixed seed);
early stopping and the variant choice on the 20 % (waveform E); V300 scored once with the chosen model.
Budget: --minutes of training per variant (budget rule: ~1 node-hour per paradigm in total).

IMPORTANT: numpy is imported before torch (torch first loads the system libstdc++ and breaks numpy's extension).

    python nn_direct.py --arch mlp --loss mse,w --minutes 25 --nproc 128     -> results/nn_mlp.json, models/nn_mlp_*.pt
"""
from __future__ import annotations
import argparse, json, time
from multiprocessing import get_context

import numpy as np
import common as C
import direct_targets as D
import torch                                               # noqa: E402  (after numpy, see docstring)

RES = C.HERE / "results"; MOD = C.HERE / "models"; MOD.mkdir(exist_ok=True)
SEED = 20261031
S = D.S_GRID


def nu_feat(keys):
    q = keys[:, 0]; nu = q / (1 + q) ** 2
    return np.c_[(nu - 8 / 81) / (3 / 16 - 8 / 81) * 2 - 1, (keys[:, 1] - 0.1) / 0.7]      # ~[-1, 1]


def warp(s):
    """monotone time warp that spends resolution near the merger: u in [-1, ~0.02]"""
    return -np.log1p(np.clip(-s, 0, None) / 20.0) / np.log1p(5020 / 20.0) + np.clip(s, 0, None) / 5000.0


# ------------------------------------------------------------------------------------------------ data
def load():
    d = np.load(RES / "direct_targets.npz"); keys, A, P = d["keys"], d["dlnA"], d["dphi"]
    r = np.random.default_rng(SEED); perm = r.permutation(len(keys)); nv = len(keys) // 5
    va, tr = perm[:nv], perm[nv:]
    mA, mP = A[tr].mean(0), P[tr].mean(0); sA, sP = (A[tr] - mA).std(), (P[tr] - mP).std()
    return dict(keys=keys, A=A, P=P, tr=tr, va=va, mA=mA, mP=mP, sA=sA, sP=sP)


def _pp_feats(k):
    c = C.case(tuple(k)); X1 = k[0] / (1 + k[0])
    sp, ap, php = D._aligned(c["t_bhpt"] * X1, c["h_bhpt"])
    la = np.log(np.interp(S, sp, ap)); w = np.gradient(np.interp(S, sp, php), S)
    sn, an, _ = D._aligned(c["t_nr"], c["h_nr"])
    return la, w, np.interp(S, sn, an) ** 2


def pp_features(keys, nproc):
    with get_context("fork").Pool(nproc) as p:
        r = p.map(_pp_feats, list(map(tuple, keys)), chunksize=4)
    return np.array([x[0] for x in r]), np.array([x[1] for x in r]), np.array([x[2] for x in r])


# ------------------------------------------------------------------------------------------------ models
class MLP(torch.nn.Module):
    def __init__(self, nf=24, width=256, depth=5):
        super().__init__()
        self.register_buffer("freq", torch.tensor(np.pi * 2.0 ** np.linspace(0, 7, nf), dtype=torch.float32))
        layers, d = [], 2 + 1 + 2 * nf
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]; d = width
        layers += [torch.nn.Linear(d, 2)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, p, u):          # p (N, 2), u (N, 1)
        z = u * self.freq
        return self.net(torch.cat([p, u, torch.sin(z), torch.cos(z)], -1))


class CFC(torch.nn.Module):
    def __init__(self, units=64):
        super().__init__()
        from ncps.torch import CfC
        from ncps.wirings import AutoNCP
        self.rnn = CfC(5, AutoNCP(units, 2), batch_first=True)

    def forward(self, x, ts):         # x (B, T, 4), ts (B, T): the step size enters as a 5th input channel
        # (ncps 1.0.1 squeezes batched timespans to (B,) and fails to broadcast, so timespans stay at the default 1)
        y, _ = self.rnn(torch.cat([x, (ts / 4.0).unsqueeze(-1)], -1))
        return y


# ------------------------------------------------------------------------------------------------ training
def curves_from(model, arch, d, idx, extra):
    """standardised predictions -> (dlnA, dphi) on S for cases idx"""
    model.eval()
    with torch.no_grad():
        if arch == "mlp":
            u = torch.tensor(warp(S)[:, None], dtype=torch.float32); out = []
            for i in idx:
                p = torch.tensor(np.repeat(nu_feat(d["keys"][i:i + 1]), len(S), 0), dtype=torch.float32)
                out.append(model(p, u).numpy())
            Y = np.array(out)
        else:
            x, ts, sub = extra["seq"](idx)
            Ys = model(x, ts).numpy()                                   # (B, Tsub, 2) on the sub-grid
            Y = np.stack([np.stack([np.interp(S, S[sub], Ys[b, :, j]) for j in range(2)], -1) for b in range(len(idx))])
    model.train()
    return d["mA"] + d["sA"] * Y[..., 0], d["mP"] + d["sP"] * Y[..., 1]


def _score(a):
    k, dA, dP = a
    return D.score(tuple(k), dA, dP)


def score_cases(pool, keys, A, P):
    return np.array(pool.map(_score, [(tuple(keys[i]), A[i], P[i]) for i in range(len(keys))], chunksize=2))


def train(arch, loss, d, minutes, extra, log):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = MLP() if arch == "mlp" else CFC()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr = d["tr"]; YA = (d["A"] - d["mA"]) / d["sA"]; YP = (d["P"] - d["mP"]) / d["sP"]
    W = extra["w"]                                                       # (T,) power weights, mean 1
    t0 = time.time(); step = 0; best = (np.inf, None); last_eval = t0
    u_all = warp(S).astype(np.float32); pf = nu_feat(d["keys"]).astype(np.float32)
    while time.time() - t0 < 60 * minutes:
        frac = (time.time() - t0) / (60 * minutes)
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 * (1 + np.cos(np.pi * frac)) + 1e-5
        if arch == "mlp":
            ci = np.random.choice(tr, 8192); ti = np.random.randint(0, len(S), 8192)
            p = torch.tensor(pf[ci]); u = torch.tensor(u_all[ti][:, None])
            y = torch.tensor(np.c_[YA[ci, ti], YP[ci, ti]], dtype=torch.float32)
            w = torch.tensor(W[ti] if loss == "w" else np.ones(len(ti)), dtype=torch.float32)[:, None]
            l = (w * (model(p, u) - y) ** 2).mean()
        else:
            ci = np.random.choice(tr, 16); x, ts, sub = extra["seq"](ci)
            y = torch.tensor(np.stack([YA[ci][:, sub], YP[ci][:, sub]], -1), dtype=torch.float32)
            w = torch.tensor(W[sub] if loss == "w" else np.ones(len(sub)), dtype=torch.float32)[None, :, None]
            l = (w * (model(x, ts) - y) ** 2).mean()
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        if time.time() - last_eval > 60 * minutes / 10:                  # 10 checkpoints: validation curve MSE
            last_eval = time.time()
            A, P = curves_from(model, arch, d, d["va"], extra)
            v = float(np.mean(W * ((A - d["A"][d["va"]]) ** 2 / d["sA"] ** 2 + (P - d["P"][d["va"]]) ** 2 / d["sP"] ** 2)))
            if v < best[0]:
                best = (v, {k: t.clone() for k, t in model.state_dict().items()})
            log(f"  {arch}/{loss} step {step} loss {float(l):.3e} val-wMSE {v:.3e} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best[1])
    return model, step, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "cfc"], required=True)
    ap.add_argument("--loss", default="mse,w"); ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--nproc", type=int, default=128); ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    d = load(); V = np.array(C.V300)
    t = time.time()
    la, wpp, pw = pp_features(np.r_[d["keys"], V], a.nproc)
    nT = len(d["keys"]); W = pw[:nT][d["tr"]].mean(0); W = W / W.mean()
    log(f"features {time.time() - t:.0f}s; train {len(d['tr'])} val {len(d['va'])} V300 {len(V)}")
    sub = np.unique(np.r_[np.arange(0, np.searchsorted(S, -200.0), 4), np.arange(np.searchsorted(S, -200.0), len(S), 5),
                          len(S) - 1])                                   # CfC sub-grid: 4 M inspiral, 1 M merger (~1500 steps)
    laN = (la - la.mean()) / la.std(); wN = (wpp - wpp.mean()) / wpp.std()
    pf_all = nu_feat(np.r_[d["keys"], V])
    dts = np.diff(S[sub], prepend=S[sub][0] - (S[sub][1] - S[sub][0]))

    def seq(idx, offset=0):
        idx = np.asarray(idx) + offset
        x = np.stack([laN[idx][:, sub], wN[idx][:, sub], np.repeat(pf_all[idx][:, :1], len(sub), 1),
                      np.repeat(pf_all[idx][:, 1:], len(sub), 1)], -1)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(np.repeat(dts[None], len(idx), 0), dtype=torch.float32), sub

    torch.set_num_threads(a.threads)
    extra = dict(w=W, seq=seq)
    out = {}; best = None
    with get_context("fork").Pool(a.nproc) as pool:
        for loss in a.loss.split(","):
            model, steps, secs = train(a.arch, loss, d, a.minutes, extra, log)
            A, P = curves_from(model, a.arch, d, d["va"], extra)
            ev = score_cases(pool, d["keys"][d["va"]], A, P)
            nparam = sum(p.numel() for p in model.parameters())
            out[loss] = dict(val=C.stats(ev), steps=steps, train_s=secs, n_params=int(nparam))
            torch.save(model.state_dict(), MOD / f"nn_{a.arch}_{loss}.pt")
            log(f"{a.arch}/{loss}: val E {C.fmt(C.stats(ev))}  params {nparam}  {steps} steps {secs:.0f}s")
            if best is None or out[loss]["val"]["mean"] < out[best[0]]["val"]["mean"]:
                best = (loss, model)
        loss, model = best                                                   # chosen on validation E (training pool)
        dV = dict(d, keys=V, mA=d["mA"], mP=d["mP"])
        extraV = dict(extra, seq=lambda idx: seq(idx, offset=nT))
        t = time.time(); A, P = curves_from(model, a.arch, dV, np.arange(len(V)), extraV); pred_s = (time.time() - t) / len(V)
        eV = score_cases(pool, V, A, P)
        out["chosen"] = dict(loss=loss, V300=C.stats(eV), V300_cases=eV.tolist(), predict_s_per_case=pred_s)
        log(f"{a.arch} chosen {loss}: V300 {C.fmt(C.stats(eV))}  predict {pred_s * 1e3:.1f} ms/case")
    (RES / f"nn_{a.arch}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
