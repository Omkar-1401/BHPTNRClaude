"""Cache routing for the frequency-domain mismatch study (import FIRST, before any model module).

Existing (ppBHPT, NR) pairs are READ from spin_dep/spin_wf_cache + spin_dep/flux_loss_cache, or from training_size/cache
(V300).  A pair that exists in neither is built into exploration/fdm/cache/{wf,loss}.  Nothing is written into spin_dep/
or training_size/.  The sealed test caches (exploration/t100/cache) are never read: test_guard stays closed.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; EXPL = HERE.parent; REPO = EXPL.parent
for p in (EXPL, REPO / "spin_dep", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import fit_pn_anchored_spin as S              # noqa: E402
import fit_fluxanchored_spin as FS            # noqa: E402

CACHE = HERE / "cache"; WF_DIR = CACHE / "wf"; LOSS_DIR = CACHE / "loss"
WF_DIR.mkdir(parents=True, exist_ok=True); LOSS_DIR.mkdir(parents=True, exist_ok=True)
TS_CACHE = REPO / "training_size" / "cache"
_WF0, _LOSS0 = S.CACHE, FS.LOSSCACHE
_build0 = FS.build_case
_name = lambda kind, q, c: f"{kind}_q{q:.6f}_chi{c:+.3f}.npz"


def _dirs(q, c):
    """(wf_dir, loss_dir) holding this key: spin_dep first, then training_size, else ours (built there)."""
    for wf, lo in ((_WF0, _LOSS0), (TS_CACHE / "wf", TS_CACHE / "loss")):
        if (wf / _name("wf", q, c)).exists() and (lo / _name("loss", q, c)).exists():
            return wf, lo
    return WF_DIR, LOSS_DIR


def build_case(q, chi1):
    S.CACHE, FS.LOSSCACHE = _dirs(float(q), float(chi1))
    try:
        return _build0(q, chi1)
    finally:
        S.CACHE, FS.LOSSCACHE = _WF0, _LOSS0


FS.build_case = build_case
import test_guard                             # noqa: E402,F401  (sealed test points refused; the gate stays closed)
