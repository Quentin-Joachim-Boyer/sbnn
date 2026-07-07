"""Tache 1 (synthese section 5) -- la robustesse predit-elle la generalisation ?

Hypothese directrice : les reservoirs les plus robustes generalisent mieux.
Protocole : K reservoirs SBN aleatoires FIXES ; readout logistique sur vrai
split train/test -> acc_test ; robustesse = readout_robustness (stabilite de la
decision sous flip d'1 bit d'entree, mesuree sur le train, label-free) ;
controle "richesse dynamique" = rang du noyau, etats distincts, entropie.
Stats : Spearman + correlation PARTIELLE controlant la richesse + OLS.

Datasets :
  bool_struct : y=1 si somme(x) in {2,3} sur 6 bits (symetrique non-monotone).
  bool_band   : y=1 si somme(x) in {1,4} sur 6 bits (bande non-contigue, structure
                independante, plus dure).
  moons       : make_moons binarise (thermometre) -- CONTROLE "tache facile"
                (quasi separable ; on s'attend a peu d'effet reservoir).

Usage (1 dataset par appel, contrainte de temps) :
  python experiments/hypothesis_robustness_generalization.py --only bool_struct --K 200
  python experiments/hypothesis_robustness_generalization.py --aggregate   # combine
Necessite : pip install -e . (numpy, scikit-learn, matplotlib).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbn_predict.core.network import random_sbn                      # noqa: E402
from sbn_predict.regimes.reservoir import reservoir_features         # noqa: E402
from sbn_predict.metrics import readout_robustness                   # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


# --------------------------------------------------------------------------- #
def dataset_bool_struct(n_bits: int, S: set[int]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(list(itertools.product((0, 1), repeat=n_bits)))
    s = X.sum(axis=1)
    y = np.isin(s, list(S)).astype(int)
    return X, y


def _thermometer_row(values, lo, hi, levels):
    frac = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    ks = np.round(frac * levels).astype(int)
    bits = []
    for k in ks:
        v = np.zeros(levels, dtype=int); v[:k] = 1; bits.append(v)
    return np.concatenate(bits)


def dataset_moons(n_samples=240, levels=4, noise=0.30, seed=0):
    from sklearn.datasets import make_moons
    Xc, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    lo, hi = Xc.min(axis=0), Xc.max(axis=0)
    X = np.array([_thermometer_row(r, lo, hi, levels) for r in Xc])
    return X.astype(int), y.astype(int)


DATASETS = {
    "bool_struct": (lambda seed: dataset_bool_struct(6, {2, 3}),
                    dict(n_nodes=16, wb_max=3, T=6, test_frac=0.35)),
    "bool_band":   (lambda seed: dataset_bool_struct(6, {1, 4}),
                    dict(n_nodes=16, wb_max=3, T=6, test_frac=0.35)),
    "moons":       (lambda seed: dataset_moons(levels=4, noise=0.30, seed=seed),
                    dict(n_nodes=12, wb_max=3, T=4, test_frac=0.35)),
}


# --------------------------------------------------------------------------- #
def kernel_rank(feats):
    return int(np.linalg.matrix_rank(feats.astype(float)))


def distinct_states(feats, n_nodes, T):
    flat = feats.reshape(feats.shape[0], T + 1, n_nodes).reshape(-1, n_nodes)
    return len(np.unique(flat, axis=0))


def activation_entropy(feats):
    p = np.clip(feats.mean(axis=0), 1e-9, 1 - 1e-9)
    return float((-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).mean())


# --------------------------------------------------------------------------- #
def _rankdata(a):
    a = np.asarray(a, float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); start = csum - counts
    return ((start + csum - 1) / 2.0)[inv]


def _pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y))


def _residuals(y, z):
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def partial_spearman(x, y, z):
    rx, ry, rz = _rankdata(x), _rankdata(y), _rankdata(z)
    return _pearson(_residuals(rx, rz), _residuals(ry, rz))


def perm_pvalue(fn, x, y, obs, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed); c = 0
    for _ in range(n_perm):
        if abs(fn(x, rng.permutation(y))) >= abs(obs) - 1e-12:
            c += 1
    return (c + 1) / (n_perm + 1)


def perm_pvalue_partial(x, y, z, obs, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed); c = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(y))
        if abs(partial_spearman(x, y[perm], z)) >= abs(obs) - 1e-12:
            c += 1
    return (c + 1) / (n_perm + 1)


def ols_standardized(y, predictors):
    def z(a):
        a = np.asarray(a, float); s = a.std()
        return (a - a.mean()) / s if s > 0 else a - a.mean()
    names = list(predictors)
    A = np.column_stack([np.ones(len(y))] + [z(predictors[k]) for k in names])
    coef, *_ = np.linalg.lstsq(A, z(y), rcond=None)
    return {n: float(c) for n, c in zip(names, coef[1:])}


# --------------------------------------------------------------------------- #
def run_dataset(name, seed, K):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    make, cfg = DATASETS[name]
    X, y = make(seed)
    n_nodes, wb_max, T, test_frac = cfg["n_nodes"], cfg["wb_max"], cfg["T"], cfg["test_frac"]
    n_input = X.shape[1]
    input_nodes = list(range(n_input))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac,
                                          random_state=seed, stratify=y)
    master = np.random.default_rng(seed)
    rows = []
    for k in range(K):
        rng = np.random.default_rng(master.integers(1 << 32))
        net = random_sbn(n_nodes, wb_max=wb_max, rng=rng)
        ftr = reservoir_features(net, Xtr, T=T, input_nodes=input_nodes)
        fte = reservoir_features(net, Xte, T=T, input_nodes=input_nodes)
        clf = LogisticRegression(max_iter=2000).fit(ftr, ytr)
        rows.append(dict(
            k=k, acc_train=float(clf.score(ftr, ytr)), acc_test=float(clf.score(fte, yte)),
            robustness=readout_robustness(net, Xtr, clf, T=T, input_nodes=input_nodes),
            kernel_rank=kernel_rank(ftr),
            distinct_states=distinct_states(ftr, n_nodes, T),
            act_entropy=activation_entropy(ftr)))
    arr = {key: [r[key] for r in rows] for key in rows[0]}
    arr["_name"] = name
    arr["_meta"] = dict(n_nodes=n_nodes, wb_max=wb_max, T=T, K=K,
                        n_train=len(ytr), n_test=len(yte), n_input=n_input,
                        base_rate=float(max(y.mean(), 1 - y.mean())))
    return arr


def analyse(arr, seed=0):
    acc = np.array(arr["acc_test"]); rob = np.array(arr["robustness"])
    rich = np.array(arr["kernel_rank"])
    acctr = np.array(arr["acc_train"]); gap = acctr - acc
    r_ar = spearman(acc, rob)
    r_partial = partial_spearman(rob, acc, rich)
    return dict(
        acc_mean=float(acc.mean()), acc_std=float(acc.std()),
        acc_min=float(acc.min()), acc_max=float(acc.max()),
        acctr_mean=float(acctr.mean()), gap_mean=float(gap.mean()),
        rob_mean=float(rob.mean()), rob_std=float(rob.std()),
        spearman_acc_rob=r_ar, p_acc_rob=perm_pvalue(spearman, rob, acc, r_ar, seed=seed),
        spearman_acc_rich=spearman(acc, rich), spearman_rob_rich=spearman(rob, rich),
        spearman_rob_acctrain=spearman(rob, acctr), spearman_rob_gap=spearman(rob, gap),
        partial_acc_rob_given_rich=r_partial,
        p_partial=perm_pvalue_partial(rob, acc, rich, r_partial, seed=seed),
        **{f"beta_{k}": v for k, v in
           ols_standardized(acc, {"robustness": rob, "kernel_rank": rich}).items()})


def verdict_line(name, s):
    strong = s["partial_acc_rob_given_rich"] > 0 and s["p_partial"] < 0.05
    tag = "GO  " if strong else "no-go"
    return (f"[{tag}] {name}: acc {s['acc_mean']:.3f}+-{s['acc_std']:.3f} "
            f"[{s['acc_min']:.2f},{s['acc_max']:.2f}] | rho(acc,rob)={s['spearman_acc_rob']:+.2f} "
            f"(p={s['p_acc_rob']:.3f}) | partiel|richesse={s['partial_acc_rob_given_rich']:+.2f} "
            f"(p={s['p_partial']:.3f}) | beta_rob={s['beta_robustness']:+.2f} vs "
            f"beta_rich={s['beta_kernel_rank']:+.2f} | rho(rob,rich)={s['spearman_rob_rich']:+.2f} "
            f"|| rho(rob,acc_train)={s['spearman_rob_acctrain']:+.2f} rho(rob,ecart)={s['spearman_rob_gap']:+.2f}")


# --------------------------------------------------------------------------- #
def make_figure(items, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 9.4), squeeze=False)
    for j, (arr, s) in enumerate(items):
        rob = np.array(arr["robustness"]); acc = np.array(arr["acc_test"])
        acctr = np.array(arr["acc_train"]); rich = np.array(arr["kernel_rank"])
        base = arr["_meta"]["base_rate"]
        ax = axes[0][j]
        sc = ax.scatter(rob, acc, c=rich, cmap="viridis", s=26,
                        edgecolor="k", linewidth=0.3, alpha=0.85)
        ax.axhline(base, ls="--", c="grey", lw=1, label=f"taux de base = {base:.2f}")
        ax.set_xlabel("robustesse readout"); ax.set_ylabel("accuracy TEST")
        ax.set_title(f"{arr['_name']} (n={arr['_meta']['n_nodes']}, T={arr['_meta']['T']}, "
                     f"K={arr['_meta']['K']})\nrho(acc_test, rob)={s['spearman_acc_rob']:+.2f} "
                     f"(p={s['p_acc_rob']:.3f}) | partiel|rich={s['partial_acc_rob_given_rich']:+.2f} "
                     f"(p={s['p_partial']:.3f})", fontsize=9)
        fig.colorbar(sc, ax=ax).set_label("richesse : rang du noyau")
        ax.legend(loc="best", fontsize=8)
        ax2 = axes[1][j]
        ax2.scatter(rob, acctr, s=18, c="#c0392b", alpha=0.7, label="train")
        ax2.scatter(rob, acc, s=18, c="#2471a3", alpha=0.7, label="test")
        ax2.axhline(base, ls="--", c="grey", lw=1)
        ax2.set_xlabel("robustesse readout"); ax2.set_ylabel("accuracy")
        ax2.set_title(f"ciseau capacite : rho(rob, acc_train)={s['spearman_rob_acctrain']:+.2f} | "
                      f"rho(rob, ecart)={s['spearman_rob_gap']:+.2f}", fontsize=9)
        ax2.legend(loc="best", fontsize=8)
    fig.suptitle("Robustesse = axe de REGULARISATION (train baisse, ecart se resserre), "
                 "pas de performance (test plat)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(path, dpi=130); plt.close(fig)


def save_csv(items, path):
    import csv
    cols = ["dataset", "k", "acc_train", "acc_test", "robustness",
            "kernel_rank", "distinct_states", "act_entropy"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for arr, _ in items:
            for i in range(len(arr["k"])):
                w.writerow([arr["_name"]] + [arr[c][i] for c in cols[1:]])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(DATASETS))
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    FIG_DIR.mkdir(exist_ok=True)

    if args.aggregate:
        items = []
        for name in DATASETS:
            fp = FIG_DIR / f"res_{name}.json"
            if fp.exists():
                d = json.loads(fp.read_text())
                items.append((d["arr"], analyse(d["arr"], seed=args.seed)))
        make_figure(items, FIG_DIR / "hypo_robustness_generalization.png")
        save_csv(items, FIG_DIR / "hypo_reservoirs.csv")
        print("--- VERDICT hypothese directrice section 5 ---")
        for arr, s in items:
            print("  " + verdict_line(arr["_name"], s))
        return

    names = [args.only] if args.only else list(DATASETS)
    for name in names:
        arr = run_dataset(name, seed=args.seed, K=args.K)
        stats = analyse(arr, seed=args.seed)
        (FIG_DIR / f"res_{name}.json").write_text(json.dumps({"arr": arr, "stats": stats}))
        print(verdict_line(name, stats))


if __name__ == "__main__":
    main()
