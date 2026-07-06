"""§4.5 — Apprentissage PAR RECHERCHE dans l'espace discret des poids/SBF.

Recherche locale / évolutionnaire / Monte-Carlo, fitness = précision de
prédiction (+ régularisation par robustesse). Réutilise l'échantillonnage
type MCSBN (core.network.random_sbn).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.network import SBN, random_sbn


def hill_climb(fitness: Callable[[SBN], float], n: int, wb_max: int,
               iters: int = 500, rng: np.random.Generator | None = None) -> tuple[SBN, float]:
    """Recherche locale : mutation d'un poids à la fois, on garde si ça améliore."""
    rng = rng or np.random.default_rng()
    net = random_sbn(n, wb_max, rng)
    score = fitness(net)
    for _ in range(iters):
        i, j = rng.integers(n), rng.integers(n)
        old = net.W[i, j]
        net.W[i, j] = rng.integers(-wb_max, wb_max + 1)
        new_score = fitness(net)
        if new_score >= score:
            score = new_score
        else:
            net.W[i, j] = old  # revert
    return net, score
