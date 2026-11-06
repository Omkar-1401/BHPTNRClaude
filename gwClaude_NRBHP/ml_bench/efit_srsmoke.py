"""2-minute smoke test of the efit_sr path (PySR export -> SRModel -> refit through E) on toy expressions"""
import numpy as np, torch, efit as F, efit_sr as R, common as C
from multiprocessing import get_context
from pysr import PySRRegressor
rng = np.random.default_rng(0); X = rng.uniform(-1, 1, (300, 3)); X[:, 1] = rng.uniform(0.1, 0.18, 300)
ms = []
for y in (0.01 * X[:, 0] + 0.1 * X[:, 2], -0.05 * X[:, 0] ** 2 + 0.2 * X[:, 1]):
    m = PySRRegressor(niterations=5, binary_operators=["+", "-", "*", "/"], unary_operators=["square", "exp"], maxsize=15,
                      procs=0, parallelism="serial", variable_names=["u", "nu", "chi"], verbosity=0, progress=False,
                      temp_equation_file=True)
    m.fit(X, y); ms.append(m); print(m.equations_[["complexity", "loss", "sympy_format"]].tail(2).to_string(), flush=True)
mod = R.SRModel(ms[0].pytorch(index=R.candidates(ms[0], 15)[-1]), ms[1].pytorch(index=R.candidates(ms[1], 15)[-1]))
print("params", [p.dtype for p in mod.parameters()], tuple(mod(F.pfeat([C.T96[0], C.T96[5]])).shape), flush=True)
keys = [C.T96[0], C.T96[40]]
with get_context("fork").Pool(2) as pool:
    info = F.train(mod, keys, 0.3, pool, print, lr=5e-3, reseed=40, tag="srsmoke")
    print("SMOKE OK", info["steps"], F.evaluate(mod, pool, {"t": keys})["t"]["mean"], flush=True)
