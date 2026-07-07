"""§4.2 — Règle du perceptron, PAR NŒUD (converge si séparable linéairement).
Applicable au readout (régime C) et à la couche de sortie (régime A).
Ne résout pas l'affectation de crédit des couches cachées.
"""
from __future__ import annotations

import numpy as np


def train_perceptron(X: np.ndarray, y: np.ndarray, wb_max: int,
                     epochs: int = 100) -> tuple[np.ndarray, int]:
    """Perceptron à sortie seuil, poids projetés dans [-wb_max, wb_max] ∩ ℤ.
    Renvoie (poids, biais/theta négatif)."""
    X = np.asarray(X, dtype=int)
    y = np.asarray(y, dtype=int)
    n = X.shape[1]
    w = np.zeros(n, dtype=int)
    b = 0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            pred = 1 if int(xi @ w) + b > 0 else 0
            if pred != yi:
                w = np.clip(w + (yi - pred) * xi, -wb_max, wb_max)
                b += (yi - pred)
                errors += 1
        if errors == 0:
            break
    return w, b


def train_perceptron_margin(X: np.ndarray, y: np.ndarray, wb_max: int,
                            kappa: int = 1, epochs: int = 300) -> tuple[np.ndarray, int]:
    """Perceptron à MARGE de stabilité `kappa` : on exige non pas seulement le bon
    signe, mais un champ correctement signé d'au moins `kappa`. En stockage de
    points fixes (mémoire associative), `kappa` = profondeur de bassin visée
    (la « robustesse = marge » du §5 devenue objectif d'apprentissage).

    Renvoie (poids entiers dans [-wb_max, wb_max], biais b) ; seuil theta = -b.
    Note : atteindre une grande marge exige de gros poids ; la borne `wb_max`
    (signature du projet) plafonne donc le compromis capacité × robustesse.
    """
    X = np.asarray(X, dtype=int)
    y = np.asarray(y, dtype=int)
    n = X.shape[1]
    w = np.zeros(n, dtype=int)
    b = 0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            s = 1 if yi == 1 else -1
            field = int(xi @ w) + b
            if s * field < kappa:                 # marge insuffisante
                w = np.clip(w + s * xi, -wb_max, wb_max)
                b += s
                errors += 1
        if errors == 0:
            break
    return w, b
