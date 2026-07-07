"""Suivi decisif section 5 : A ACCURACY DE TRAIN EGALE, la robustesse
predit-elle le test ? (Test que la tache 1 a fait emerger.)

Lit les res_*.json produits par hypothesis_robustness_generalization.py (K reservoirs,
acc_train / acc_test / robustness deja calcules) -- aucun recalcul lourd.

Prerequis : lancer d'abord
  python experiments/hypothesis_robustness_generalization.py --only bool_struct --K 200
  (idem bool_band, moons)   puis   ... --aggregate
Ensuite :
  python experiments/followup_equal_train.py

Sortie : partial Spearman(test, rob | train_acc) + p (permutation), regression
standardisee test ~ train + rob (beta_rob = effet a train egal), experience de
selection (top-rob vs bottom-rob dans des bandes de train), figure de residus.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

FIG_DIR = (Path(sys.argv[1]) if len(sys.argv) > 1
           else Path(__file__).resolve().parents[1] / "figures")
NAMES = ["bool_struct", "bool_band", "moons"]


def rank(a):
    a = np.asarray(a, float)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); st = cs - c
    return ((st + cs - 1) / 2.0)[inv]


def pear(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def resid(y, z):
    A = np.column_stack([np.ones_like(z), z])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ c


def partial(x, y, z):
    return pear(resid(rank(x), rank(z)), resid(rank(y), rank(z)))


def pval_partial(x, y, z, obs, n=5000, seed=0):
    rng = np.random.default_rng(seed); c = 0
    for _ in range(n):
        p = rng.permutation(len(y))
        if abs(partial(x, y[p], z)) >= abs(obs) - 1e-12:
            c += 1
    return (c + 1) / (n + 1)


def zc(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / s if s > 0 else a - a.mean()


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = []
    for name in NAMES:
        fp = FIG_DIR / f"res_{name}.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())["arr"]
        tr = np.array(d["acc_train"]); te = np.array(d["acc_test"]); rob = np.array(d["robustness"])
        pc = partial(rob, te, tr); pp = pval_partial(rob, te, tr, pc)
        A = np.column_stack([np.ones(len(te)), zc(tr), zc(rob)])
        coef, *_ = np.linalg.lstsq(A, zc(te), rcond=None)
        order = np.argsort(tr); diffs = []
        for b in np.array_split(order, 3):
            r = rob[b]; t = te[b]; m = np.median(r)
            diffs.append(float(t[r >= m].mean() - t[r < m].mean()))
        items.append((name, tr, te, rob, pc, pp, float(coef[1]), float(coef[2]), diffs))
        print(f"{name:12s} partial(rob,test|train)={pc:+.2f} (p={pp:.3f}) | "
              f"beta_train={coef[1]:+.2f} beta_rob(a train egal)={coef[2]:+.2f} | "
              f"selection top-bottom rob par bande train={[round(x, 3) for x in diffs]} "
              f"moy={np.mean(diffs):+.3f}")

    if not items:
        print("Aucun res_*.json trouve. Lancer d'abord hypothesis_robustness_generalization.py.")
        return
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.0), squeeze=False)
    for ax, (name, tr, te, rob, pc, pp, bt, br, diffs) in zip(axes[0], items):
        rr = resid(rank(rob), rank(tr)); rt = resid(rank(te), rank(tr))
        ax.scatter(rr, rt, s=24, c="#34495e", alpha=0.7, edgecolor="w", linewidth=0.3)
        b1 = np.polyfit(rr, rt, 1)
        xs = np.array([rr.min(), rr.max()])
        ax.plot(xs, np.polyval(b1, xs), c="#e74c3c", lw=1.6)
        ax.axhline(0, c="grey", lw=0.6); ax.axvline(0, c="grey", lw=0.6)
        ax.set_xlabel("robustesse | train (residu de rang)")
        ax.set_ylabel("accuracy test | train (residu de rang)")
        sig = "significatif" if pp < 0.05 else "n.s."
        ax.set_title(f"{name}\npartial(test, rob | train)={pc:+.2f} (p={pp:.3f}, {sig})\n"
                     f"beta_rob a train egal={br:+.2f} | gain selection moy={np.mean(diffs):+.3f}",
                     fontsize=9)
    fig.suptitle("Test decisif section 5 : a TRAIN EGAL, la robustesse aide-t-elle le test ?  "
                 "-> effet petit et de SIGNE INSTABLE selon la tache", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "followup_equal_train.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("Figure:", out)


if __name__ == "__main__":
    main()
