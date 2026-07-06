"""§5 — Grandeurs du projet réinterprétées en langage apprentissage.

robustesse ≈ marge / tolérance au bruit ≈ proxy de généralisation.
"""
from __future__ import annotations

import numpy as np

from .core.network import SBN


def basin_robustness(net: SBN, x0: np.ndarray, n_flips: int = 1,
                     trials: int = 50, rng: np.random.Generator | None = None) -> float:
    """Fraction des perturbations à `n_flips` bits qui retombent sur le même
    attracteur (proxy de taille de bassin / tolérance au bruit)."""
    rng = rng or np.random.default_rng()
    x0 = np.asarray(x0, dtype=int)
    ref = tuple(sorted(net.attractor_sync(x0)))
    same = 0
    for _ in range(trials):
        x = x0.copy()
        idx = rng.choice(net.n, size=n_flips, replace=False)
        x[idx] ^= 1
        if tuple(sorted(net.attractor_sync(x))) == ref:
            same += 1
    return same / trials


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(true)).mean())
