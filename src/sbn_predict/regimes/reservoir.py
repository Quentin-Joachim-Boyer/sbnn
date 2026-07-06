"""Régime C — Reservoir computing (synthese §2.C, jalon n°1).

SBN aléatoire FIXE (jamais entraîné) comme réservoir dynamique ; seul un
readout linéaire est entraîné sur les états visités. Sidestep total de la
non-différentiabilité. Prototype recommandé en premier.
"""
from __future__ import annotations

import numpy as np

from ..core.network import SBN


def collect_states(net: SBN, x0: np.ndarray, T: int) -> np.ndarray:
    """Concatène les états x(0..T) en un vecteur de features (§3.3)."""
    x = np.asarray(x0, dtype=int).copy()
    feats = [x.copy()]
    for _ in range(T):
        x = net.step_sync(x)
        feats.append(x.copy())
    return np.concatenate(feats)


def reservoir_features(net: SBN, X: np.ndarray, T: int, input_nodes: list[int]) -> np.ndarray:
    """Features réservoir pour un batch X (clamp des input_nodes à chaque ligne).

    TODO(jalon 1) : variante "statistiques" (fréquence d'activation, attracteur
    atteint) au lieu de la trajectoire brute.
    """
    out = []
    for row in X:
        net.clamped = {node: int(b) for node, b in zip(input_nodes, row)}
        x0 = np.zeros(net.n, dtype=int)
        for node, b in zip(input_nodes, row):
            x0[node] = int(b)
        out.append(collect_states(net, x0, T))
    return np.asarray(out)


# Readout : voir sklearn LogisticRegression dans experiments/milestone1_reservoir.py
# (le gradient marche ici, on est HORS du réservoir).
