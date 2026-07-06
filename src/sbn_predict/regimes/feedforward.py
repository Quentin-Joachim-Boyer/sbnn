"""Régime A — Classifieur feedforward (SBN acyclique déplié en MLP de portes à
seuil), synthese §2.A. Obstacle central : apprentissage multicouche = gradient
sur un seuil non différentiable -> voir learning/ste.py (straight-through).

Une seule porte ne calcule pas XOR (Minsky-Papert) -> couches cachées requises.
"""
from __future__ import annotations

import numpy as np

from ..core.network import SBN


def unroll_forward(net: SBN, x0: np.ndarray, L: int, output_nodes: list[int]) -> np.ndarray:
    """Clamp l'entrée pendant L pas (= profondeur), lit les nœuds de sortie."""
    x = np.asarray(x0, dtype=int).copy()
    for _ in range(L):
        x = net.step_sync(x)
    return x[output_nodes]
