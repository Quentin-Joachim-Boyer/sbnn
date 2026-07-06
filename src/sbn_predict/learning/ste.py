"""§4.4 — Straight-Through Estimator / gradients de substitution (BNN).

Poids-ombres continus entraînés par backprop (seuil ~ identité au backward),
projetés sur [-wb, wb] ∩ ℤ au forward. Permet le multicouche (régime A).
Attention : quantification plus agressive que les BNN usuels (wb petit).

STUB : nécessite un framework autograd (torch/jax) — porté ultérieurement si
la voie performance est retenue (cf. synthese §7 Q5). Non prioritaire.
"""
from __future__ import annotations


def train_ste(*args, **kwargs):  # noqa: D401
    raise NotImplementedError(
        "Straight-through estimator : à implémenter avec torch/jax si la voie "
        "'performance' est retenue. Voir docs/synthese §4.4."
    )
