"""§4.3 — Ajustement EXACT par énumération de SBF (levier spécifique au projet).

Pour un nœud dont on connaît les paires (entrée_du_nœud -> sortie_voulue),
choisir la SBF qui minimise l'erreur = argmin sur la table énumérée. Exact,
sans gradient. Trivial pour la couche de sortie et le readout.
"""
from __future__ import annotations

import numpy as np

from ..core.sbf import SBF, iter_sbf


def best_sbf(inputs: np.ndarray, targets: np.ndarray, allow_theta: bool = False) -> tuple[SBF, int]:
    """inputs : (m, k) en {0,1} ; targets : (m,) en {0,1}.
    Renvoie (meilleure SBF, nb d'erreurs)."""
    inputs = np.asarray(inputs, dtype=int)
    targets = np.asarray(targets, dtype=int)
    k = inputs.shape[1]
    best, best_err = None, None
    for f in iter_sbf(k, allow_theta):
        pred = np.array([f(tuple(row)) for row in inputs])
        err = int((pred != targets).sum())
        if best_err is None or err < best_err:
            best, best_err = f, err
            if err == 0:
                break
    return best, best_err
