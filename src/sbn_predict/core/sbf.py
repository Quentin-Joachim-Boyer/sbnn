"""Signed Boolean Function (SBF) — l'unité de calcul.

Définition (fidèle à `sbf.py` / `SBFStatTable.java` du projet PluripotentSBN) :

    f : {0,1}^k -> {0,1},   f(x) = 1  <=>  sum_i w_i * x_i > theta

- Poids entiers bornés : w_i in [-wb(k), +wb(k)].
- Seuil `theta` : 0 dans le formalisme d'origine. On l'expose ici comme
  paramètre entier borné, car theta != 0 équivaut EXACTEMENT à un nœud de biais
  clampé à 1 de poids -theta (cf. docs/synthese §0.2). Le garder explicite
  permet de trancher la question "biais" sans nœud spécial.

Passer à theta != 0 fait sortir des SBF homogènes (séparables par l'origine)
vers les fonctions à seuil générales (LTF) : on récupère AND, OR, etc., et
l'état nul cesse d'être un point fixe forcé.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Iterator


def wb(k: int) -> int:
    """Borne des poids en dimension k (table exacte du projet)."""
    table = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5}
    if k <= 0:
        raise ValueError("k doit être >= 1")
    return table.get(k, k)  # wb(d) = d pour d > 5


@dataclass(frozen=True)
class SBF:
    """Une fonction booléenne à signe de dimension k.

    weights : tuple d'entiers de longueur k, chacun dans [-wb(k), wb(k)].
    theta   : seuil entier (0 = formalisme d'origine).
    """

    weights: tuple[int, ...]
    theta: int = 0

    def __post_init__(self) -> None:
        k = len(self.weights)
        b = wb(k)
        if any(abs(w) > b for w in self.weights):
            raise ValueError(f"poids hors de [-{b}, {b}] pour k={k}")

    @property
    def k(self) -> int:
        return len(self.weights)

    def __call__(self, x: tuple[int, ...]) -> int:
        if len(x) != self.k:
            raise ValueError("dimension de x incompatible")
        s = sum(w * xi for w, xi in zip(self.weights, x))
        return 1 if s > self.theta else 0

    def truth_table(self) -> tuple[int, ...]:
        """Vecteur de sortie sur les 2^k entrées (ordre lexicographique)."""
        return tuple(self(x) for x in product((0, 1), repeat=self.k))


def _inputs(k: int) -> list[tuple[int, ...]]:
    return list(product((0, 1), repeat=k))


@lru_cache(maxsize=None)
def enumerate_sbf_tables(k: int, allow_theta: bool = False) -> dict[tuple[int, ...], SBF]:
    """Énumère les SBF DISTINCTES (par table de vérité) en dimension k.

    Renvoie {table_de_vérité: un SBF représentatif}. C'est le levier
    "ajustement exact par énumération" (synthese §4.3) : l'ensemble est fini.

    allow_theta=False -> theta=0 uniquement (compte d'origine : 3->? ...).
    allow_theta=True  -> balaie aussi theta dans [-k*wb, k*wb] (LTF générales).
    """
    b = wb(k)
    xs = _inputs(k)
    weight_ranges = product(range(-b, b + 1), repeat=k)
    thetas = range(-k * b, k * b + 1) if allow_theta else (0,)

    tables: dict[tuple[int, ...], SBF] = {}
    for w in weight_ranges:
        for th in thetas:
            f = SBF(weights=w, theta=th)
            tt = f.truth_table()
            tables.setdefault(tt, f)  # premier représentant gardé
    return tables


def count_distinct_sbf(k: int, allow_theta: bool = False) -> int:
    """Nombre de fonctions distinctes réalisables (utile pour mesurer
    l'explosion combinatoire induite par theta != 0)."""
    return len(enumerate_sbf_tables(k, allow_theta))


def iter_sbf(k: int, allow_theta: bool = False) -> Iterator[SBF]:
    yield from enumerate_sbf_tables(k, allow_theta).values()
