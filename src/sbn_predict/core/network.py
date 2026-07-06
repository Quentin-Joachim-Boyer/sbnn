"""SBN — réseau de fonctions à signe, et sa dynamique sur l'hypercube.

État x(t) in {0,1}^n. La colonne j de la matrice de poids W définit la SBF du
nœud j :  x_j(t+1) = 1  <=>  sum_i W[i, j] * x_i(t) > theta[j].

Mise à jour :
- synchrone   : tous les nœuds en parallèle (dynamique déterministe).
  Théorème de Goles-Olivos : si W est symétrique, les seuls attracteurs sont
  des points fixes ou des cycles de période 2.
- asynchrone  : un nœud à la fois (ordre fixé ou aléatoire) -> convergence
  type Hopfield vers des points fixes (avec W symétrique).

`clamped` fige des nœuds (entrée clampée, nœud de biais à 1) : ils gardent leur
valeur imposée à chaque pas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

State = tuple[int, ...]


@dataclass
class SBN:
    W: np.ndarray                      # (n, n) entiers ; colonne j = poids du nœud j
    theta: np.ndarray | None = None    # (n,) seuils entiers ; défaut 0
    clamped: dict[int, int] = field(default_factory=dict)  # {index: valeur figée}

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=int)
        if self.W.ndim != 2 or self.W.shape[0] != self.W.shape[1]:
            raise ValueError("W doit être carrée (n, n)")
        n = self.W.shape[0]
        if self.theta is None:
            self.theta = np.zeros(n, dtype=int)
        else:
            self.theta = np.asarray(self.theta, dtype=int)
            if self.theta.shape != (n,):
                raise ValueError("theta doit être de forme (n,)")

    @property
    def n(self) -> int:
        return self.W.shape[0]

    # --- un pas ---------------------------------------------------------
    def _apply_clamp(self, x: np.ndarray) -> np.ndarray:
        for i, v in self.clamped.items():
            x[i] = v
        return x

    def step_sync(self, x: np.ndarray) -> np.ndarray:
        s = x @ self.W                  # somme pondérée par nœud
        nxt = (s > self.theta).astype(int)
        return self._apply_clamp(nxt)

    def step_async(self, x: np.ndarray, order: list[int]) -> np.ndarray:
        x = x.copy()
        for j in order:
            if j in self.clamped:
                x[j] = self.clamped[j]
                continue
            s = int(x @ self.W[:, j])
            x[j] = 1 if s > self.theta[j] else 0
        return x

    # --- trajectoires / attracteurs ------------------------------------
    def run_sync(self, x0, max_steps: int = 1000) -> tuple[list[State], list[State]]:
        """Itère la dynamique synchrone jusqu'à retomber sur un état déjà vu.

        Renvoie (transient, cycle) : le cycle est l'attracteur (longueur 1 =
        point fixe, 2 = 2-cycle, etc.)."""
        x = self._apply_clamp(np.asarray(x0, dtype=int).copy())
        seen: dict[State, int] = {}
        traj: list[State] = []
        for _ in range(max_steps):
            t = tuple(int(v) for v in x)
            if t in seen:
                start = seen[t]
                return traj[:start], traj[start:]
            seen[t] = len(traj)
            traj.append(t)
            x = self.step_sync(x)
        return traj, []  # non convergé dans max_steps

    def attractor_sync(self, x0, max_steps: int = 1000) -> list[State]:
        """L'attracteur atteint depuis x0 (liste des états du cycle)."""
        _, cycle = self.run_sync(x0, max_steps)
        return cycle


def random_sbn(n: int, wb_max: int, rng: np.random.Generator | None = None,
               symmetric: bool = False, theta: np.ndarray | None = None) -> SBN:
    """Tire un SBN aléatoire (poids entiers dans [-wb_max, wb_max]).

    Brique de base du régime C (réservoir tiré au hasard) — l'équivalent
    Python léger de MCSBN pour prototyper. symmetric=True pour la mémoire
    associative (garantit Goles-Olivos : points fixes ou 2-cycles)."""
    rng = rng or np.random.default_rng()
    W = rng.integers(-wb_max, wb_max + 1, size=(n, n))
    if symmetric:
        W = np.triu(W)
        W = W + W.T - np.diag(np.diag(W))
    return SBN(W=W, theta=theta)
