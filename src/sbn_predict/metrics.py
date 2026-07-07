"""Section 5 -- grandeurs du projet reinterpretees en langage apprentissage.

robustesse ~ marge / tolerance au bruit ~ proxy de generalisation.

Deux robustesses distinctes, a ne pas confondre :
- `basin_robustness` : stabilite de l'attracteur en dynamique LIBRE (non clampee).
  Metrique historique (taille de bassin, regime B).
- `readout_robustness` : stabilite de la DECISION du readout quand on perturbe
  les bits d'entree, reseau en regime reservoir (entrees CLAMPEES). Robustesse
  propre au regime C -- celle qu'on relie a la generalisation dans l'exp. section 5.
"""
from __future__ import annotations

import numpy as np

from .core.network import SBN
from .regimes.reservoir import reservoir_features


def basin_robustness(net: SBN, x0: np.ndarray, n_flips: int = 1,
                     trials: int = 50, rng: np.random.Generator | None = None) -> float:
    """Fraction des perturbations a `n_flips` bits qui retombent sur le meme
    attracteur (proxy de taille de bassin / tolerance au bruit)."""
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


def readout_robustness(net: SBN, X: np.ndarray, clf, T: int, input_nodes: list[int],
                       n_flips: int = 1) -> float:
    """Robustesse du regime reservoir : fraction des flips de `n_flips` bit(s)
    d'entree qui LAISSENT INCHANGEE la prediction du readout `clf`.

    Interpretation section 5 : ~ marge / tolerance au bruit d'entree ~ proxy de
    generalisation. Ne necessite AUCUNE etiquette de test : propriete label-free
    du couple (reservoir, readout) mesuree sur les entrees fournies (le train).
    On peut donc l'utiliser pour PREDIRE l'accuracy test.

    Renvoie une valeur dans [0, 1] moyennee sur (exemples x positions de flip).
    Pour n_flips=1 on balaie TOUTES les positions d'entree (deterministe).
    """
    X = np.asarray(X, dtype=int)
    input_nodes = list(input_nodes)
    base_feats = reservoir_features(net, X, T, input_nodes)
    base_pred = clf.predict(base_feats)

    stable = 0
    total = 0
    for i, row in enumerate(X):
        for p in range(len(input_nodes)):
            pert = row.copy()
            pert[p] ^= 1
            feats = reservoir_features(net, pert[None, :], T, input_nodes)
            pred = clf.predict(feats)[0]
            stable += int(pred == base_pred[i])
            total += 1
    return stable / total if total else 0.0


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(true)).mean())
