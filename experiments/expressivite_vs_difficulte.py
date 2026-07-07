"""Expressivite vs difficulte de calcul (Q1/Q2), et approximation par selection
stochastique (Q3), sur des reservoirs SBN + readout lineaire.

Q1/Q2 : n minimal d'un reservoir pour REALISER exactement une tache (readout
        lineaire, 100% sur toute la table 2^k), en fonction de la DIFFICULTE :
        parite (max. non structuree) vs bande (moderee) vs seuil/majorite (separable).
Q3   : la selection stochastique (meilleur de K reservoirs tires) realise les
        taches dures avec un reseau plus petit que le tirage typique.

Usage : python experiments/expressivite_vs_difficulte.py   (~1-2 min)
Necessite : pip install -e . + scikit-learn, matplotlib.
"""
import sys
import itertools
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbn_predict.core.network import random_sbn                     # noqa: E402
from sbn_predict.regimes.reservoir import reservoir_features        # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
T, WB = 4, 5


def all_inputs(k):
    return np.array(list(itertools.product((0, 1), repeat=k)))


def realizes(X, y, n, seed):
    from sklearn.linear_model import LogisticRegression
    net = random_sbn(n, wb_max=WB, rng=np.random.default_rng(seed))
    F = reservoir_features(net, X, T=T, input_nodes=list(range(X.shape[1])))
    return LogisticRegression(max_iter=2000).fit(F, y).score(F, y) == 1.0


def n_median(X, y, R=10):
    k = X.shape[1]
    for n in range(k, 41):
        if np.mean([realizes(X, y, n, 300 + r) for r in range(R)]) >= 0.5:
            return n
    return None


def n_best_of_k(X, y, K=20):
    k = X.shape[1]
    for n in range(k, 41):
        if any(realizes(X, y, n, 700 + r) for r in range(K)):
            return n
    return None


def families(k):
    X = all_inputs(k); s = X.sum(1)
    return X, {
        "parite": s % 2,
        "bande": np.isin(s, [k // 2, k // 2 + 1]).astype(int),
        "seuil": (s >= (k + 1) // 2).astype(int),
    }


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)

    ks = [2, 3, 4, 5, 6]
    req = {"parite": [], "bande": [], "seuil": []}
    for k in ks:
        X, tg = families(k)
        for name, y in tg.items():
            req[name].append(n_median(X, y))
    print("Q1/Q2 n_required:", req)

    pk = [2, 3, 4, 5]; nmed = []; nbest = []
    for k in pk:
        X = all_inputs(k); y = X.sum(1) % 2
        nmed.append(n_median(X, y)); nbest.append(n_best_of_k(X, y))
    print("Q3 parity median vs best-of-20:", list(zip(pk, nmed, nbest)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    a = ax[0]
    a.plot(ks, [2 ** k for k in ks], ":", c="grey", label="2^k (table complete)")
    a.plot(ks, req["parite"], "o-", c="#c0392b", label="parite (max. non structuree)")
    bx = [k for k, v in zip(ks, req["bande"]) if v is not None]
    bv = [v for v in req["bande"] if v is not None]
    a.plot(bx, bv, "^-", c="#e67e22", label="bande sum in {milieu} (moderee)")
    a.plot(ks, req["seuil"], "s-", c="#27ae60", label="seuil/majorite (separable)")
    a.set_xlabel("nb de bits d'entree k"); a.set_ylabel("n minimal du reseau (readout exact)")
    a.set_title("Q1-Q2 : la TAILLE requise suit la DIFFICULTE de la tache,\npas le nb d'entrees")
    a.legend(fontsize=8); a.set_xticks(ks)
    b = ax[1]
    b.plot(pk, nmed, "o-", c="#7f8c8d", label="tirage typique (mediane)")
    b.plot(pk, nbest, "o-", c="#2471a3", label="meilleur de 20 (selection stochastique)")
    b.set_xlabel("bits de parite k"); b.set_ylabel("n minimal pour realiser la parite")
    b.set_title("Q3 : la SELECTION stochastique realise le dur\navec un reseau plus petit")
    b.legend(fontsize=8); b.set_xticks(pk)
    fig.suptitle("Expressivite vs difficulte de calcul, et approximation par selection stochastique "
                 "(reservoir SBN, T=4, wb=5)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "expressivite_vs_difficulte.png"; fig.savefig(out, dpi=130); plt.close(fig)
    print("Figure:", out)


if __name__ == "__main__":
    main()
