"""Encodage entrée/sortie (features <-> bits).  Cf. synthese §3.

Le choix d'encodage influence fortement la séparabilité par des portes à seuil.
"""
from __future__ import annotations

import numpy as np


def one_hot(index: int, size: int) -> np.ndarray:
    """Catégoriel : robuste, coûteux en nœuds."""
    v = np.zeros(size, dtype=int)
    v[index] = 1
    return v


def thermometer(value: float, lo: float, hi: float, levels: int) -> np.ndarray:
    """Thermomètre / unaire : v -> 1^k 0^{m-k}. Préserve l'ordre, adapté aux
    seuils signés. Recommandé pour les features numériques (§3.1)."""
    if hi <= lo:
        raise ValueError("hi doit être > lo")
    frac = (value - lo) / (hi - lo)
    k = int(round(np.clip(frac, 0.0, 1.0) * levels))
    v = np.zeros(levels, dtype=int)
    v[:k] = 1
    return v


def with_bias(x: np.ndarray) -> np.ndarray:
    """Ajoute un nœud de biais clampé à 1 (indispensable si theta=0, §0.2).
    Retourne (x augmenté, index_du_biais)."""
    return np.concatenate([np.asarray(x, dtype=int), [1]])


def argmax_readout(output_bits: np.ndarray, n_classes: int) -> int:
    """One-hot + argmax sur C nœuds de sortie (§3.2). En cas d'égalité,
    renvoie le plus petit index."""
    return int(np.argmax(np.asarray(output_bits[:n_classes])))
