"""Jalon 2 (synthese §6) — Mémoire associative : rappel sous bruit.

Stocke C motifs, mesure le taux de rappel vs niveau de bruit. Relie capacité
de rappel <-> taille de bassin (robustesse). C'est aussi le test de l'attention
hard par récurrence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbn_predict.regimes.associative import make_memory, recall  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    n, P = 40, 4
    patterns = rng.integers(0, 2, size=(P, n))
    net = make_memory(patterns)

    for flips in (0, 2, 5, 10):
        ok = 0
        for _ in range(50):
            p = patterns[rng.integers(P)].copy()
            idx = rng.choice(n, size=flips, replace=False)
            p[idx] ^= 1
            out = recall(net, p, async_order=list(range(n)))
            if any(np.array_equal(out, pat) for pat in patterns):
                ok += 1
        print(f"bruit={flips:2d} bits | rappel correct = {ok/50:.2f}")


if __name__ == "__main__":
    main()
