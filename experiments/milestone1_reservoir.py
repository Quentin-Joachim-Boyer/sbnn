"""Jalon 1 (synthese §6) — Reservoir sur un jeu jouet (parité).

Test go/no-go CENTRAL : le readout sur états SBN doit battre le readout sur
l'entrée brute seule. Sinon le réservoir n'apporte rien.

Usage : python experiments/milestone1_reservoir.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbn_predict.core.network import random_sbn          # noqa: E402
from sbn_predict.regimes.reservoir import reservoir_features  # noqa: E402


def parity_dataset(k: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(list(itertools.product((0, 1), repeat=k)))
    y = X.sum(axis=1) % 2
    return X, y


def main() -> None:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("scikit-learn requis : pip install scikit-learn")
        return

    rng = np.random.default_rng(0)
    k, n, T = 4, 12, 6
    X, y = parity_dataset(k)

    net = random_sbn(n, wb_max=5, rng=rng)
    feats = reservoir_features(net, X, T=T, input_nodes=list(range(k)))

    clf_raw = LogisticRegression(max_iter=1000).fit(X, y)
    clf_res = LogisticRegression(max_iter=1000).fit(feats, y)

    print(f"parité k={k} | baseline (entrée brute) acc = {clf_raw.score(X, y):.2f}")
    print(f"parité k={k} | réservoir SBN       acc = {clf_res.score(feats, y):.2f}")
    print("Go/no-go : le réservoir doit dépasser la baseline.")


if __name__ == "__main__":
    main()
