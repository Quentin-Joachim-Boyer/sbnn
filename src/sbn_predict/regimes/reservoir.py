"""Regime C -- Reservoir computing (synthese section 2.C, jalon n.1).

SBN aleatoire FIXE (jamais entraine) comme reservoir dynamique ; seul un
readout lineaire est entraine sur les etats visites. Sidestep total de la
non-differentiabilite. Prototype recommande en premier.
"""
from __future__ import annotations

import numpy as np

from ..core.network import SBN


def collect_states(net: SBN, x0: np.ndarray, T: int) -> np.ndarray:
    """Concatene les etats x(0..T) en un vecteur de features (section 3.3)."""
    x = np.asarray(x0, dtype=int).copy()
    feats = [x.copy()]
    for _ in range(T):
        x = net.step_sync(x)
        feats.append(x.copy())
    return np.concatenate(feats)


def reservoir_features(net: SBN, X: np.ndarray, T: int, input_nodes: list[int]) -> np.ndarray:
    """Features reservoir pour un batch X (clamp des input_nodes a chaque ligne).

    Restaure `net.clamped` en sortie : la fonction ne laisse AUCUN effet de bord
    sur le reseau (le clamp de la derniere ligne fuyait auparavant vers, p. ex.,
    metrics.basin_robustness lance apres coup).

    TODO(jalon 1) : variante "statistiques" (frequence d'activation, attracteur
    atteint) au lieu de la trajectoire brute.
    """
    saved_clamp = dict(net.clamped)
    try:
        out = []
        for row in X:
            net.clamped = {node: int(b) for node, b in zip(input_nodes, row)}
            x0 = np.zeros(net.n, dtype=int)
            for node, b in zip(input_nodes, row):
                x0[node] = int(b)
            out.append(collect_states(net, x0, T))
        return np.asarray(out)
    finally:
        net.clamped = saved_clamp


# Readout : voir sklearn LogisticRegression dans experiments/milestone1_reservoir.py
# (le gradient marche ici, on est HORS du reservoir).
