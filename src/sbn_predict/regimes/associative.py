"""Régime B — Mémoire associative / attracteurs (synthese §2.B, jalon n°2).

Encoder les cibles comme attracteurs ; présenter une entrée partielle/bruitée
comme état initial ; laisser converger -> l'attracteur = la prédiction.
C'est aussi le socle de l'ATTENTION *hard* par récurrence (adressage par
contenu ; la relaxation continue = attention softmax, cf. Ramsauer 2020).

[À trancher] synchrone (2-cycles à gérer) vs asynchrone (convergence Hopfield).
"""
from __future__ import annotations

import numpy as np

from ..core.network import SBN


def hebbian_weights(patterns: np.ndarray) -> np.ndarray:
    """Stockage hebbien one-shot (règle de Hopfield adaptée au codage {0,1}).

    patterns : (P, n) motifs cibles en {0,1}. Convention : on passe en {-1,+1}
    pour la corrélation, diagonale nulle. Capacité ~ 0.14 n (§4.1).
    """
    P = np.where(np.asarray(patterns) > 0, 1, -1)
    n = P.shape[1]
    W = P.T @ P
    np.fill_diagonal(W, 0)
    return W


def make_memory(patterns: np.ndarray, theta: np.ndarray | None = None) -> SBN:
    return SBN(W=hebbian_weights(patterns), theta=theta)


def recall(net: SBN, cue: np.ndarray, async_order: list[int] | None = None,
           max_steps: int = 100) -> np.ndarray:
    """Rappel : converge depuis un indice bruité vers l'attracteur (= attention
    hard : sélection du motif stocké le plus proche)."""
    if async_order is not None:
        x = np.asarray(cue, dtype=int).copy()
        prev = None
        for _ in range(max_steps):
            x = net.step_async(x, async_order)
            if prev is not None and np.array_equal(x, prev):
                break
            prev = x.copy()
        return x
    cycle = net.attractor_sync(cue, max_steps)
    return np.asarray(cycle[0], dtype=int) if cycle else np.asarray(cue, dtype=int)
