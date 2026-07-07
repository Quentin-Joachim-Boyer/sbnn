"""Showcase sur un jeu CLASSIQUE : chiffres manuscrits `digits` (mini-MNIST 8x8).

Montre la machinerie SBN de bout en bout sur un benchmark reconnu :
  - regime C : reservoir aleatoire + readout logistique ;
  - regime B : le SBN comme classifieur par ATTRACTEURS / attention (1 prototype
    par classe stocke comme point fixe ; on converge, on lit la classe du plus
    proche prototype), cablage HEBBIEN vs APPRIS A MARGE ;
  - baselines : logistique sur pixels bruts, et le plafond 'plus proche prototype'.

Demo CONCEPTUELLE (la machinerie tourne sur du vrai), pas une course a la perf.

Usage : python experiments/showcase_digits.py
Necessite : pip install -e . + scikit-learn, matplotlib.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbn_predict.core.network import random_sbn, SBN                 # noqa: E402
from sbn_predict.regimes.reservoir import reservoir_features         # noqa: E402
from sbn_predict.regimes.associative import hebbian_weights          # noqa: E402
from sbn_predict.learning.perceptron import train_perceptron_margin  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    FIG_DIR.mkdir(exist_ok=True)

    d = load_digits()
    imgs = d.images; X = (imgs.reshape(len(imgs), -1) > 6).astype(int); y = d.target
    Xtr, Xte, ytr, yte, Itr, Ite = train_test_split(X, y, imgs, test_size=0.3,
                                                    random_state=0, stratify=y)
    acc_raw = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)

    net = random_sbn(96, wb_max=5, rng=np.random.default_rng(0))
    Ftr = reservoir_features(net, Xtr, T=3, input_nodes=list(range(64)))
    Fte = reservoir_features(net, Xte, T=3, input_nodes=list(range(64)))
    clf = LogisticRegression(max_iter=2000).fit(Ftr, ytr)
    acc_res = clf.score(Fte, yte); pred_res = clf.predict(Fte)

    protos = np.array([(Xtr[ytr == c].mean(0) > 0.5).astype(int) for c in range(10)])

    def classify(netB, x, max_steps=60):
        s = x.copy(); order = list(range(64)); prev = None
        for _ in range(max_steps):
            s = netB.step_async(s, order)
            if prev is not None and np.array_equal(s, prev):
                break
            prev = s.copy()
        return int(np.argmin((protos != s).sum(1)))

    netH = SBN(W=hebbian_weights(protos))
    accH = np.mean([classify(netH, Xte[i]) == yte[i] for i in range(len(yte))])
    n = 64; W = np.zeros((n, n), int); th = np.zeros(n, int)
    for j in range(n):
        w, b = train_perceptron_margin(protos, protos[:, j], wb_max=8, kappa=6, epochs=200)
        W[:, j] = w; th[j] = -b
    netL = SBN(W=W, theta=th)
    accL = np.mean([classify(netL, Xte[i]) == yte[i] for i in range(len(yte))])
    acc_tmpl = np.mean([np.argmin((protos != Xte[i]).sum(1)) == yte[i] for i in range(len(yte))])

    print(f"raw={acc_raw:.3f} res={acc_res:.3f} heb={accH:.3f} learn={accL:.3f} tmpl={acc_tmpl:.3f}")

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("Chiffres 'digits' : original (haut) -> binarise (bas)", fontsize=10)
    sub = fig.add_gridspec(2, 8, left=0.055, right=0.48, top=0.9, bottom=0.56, wspace=0.1, hspace=0.1)
    for c in range(8):
        i = np.where(yte == c)[0][0]
        a1 = fig.add_subplot(sub[0, c]); a1.imshow(Ite[i], cmap="gray_r"); a1.axis("off")
        a2 = fig.add_subplot(sub[1, c]); a2.imshow(Xte[i].reshape(8, 8), cmap="gray_r"); a2.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_title("Les 10 prototypes stockes = attracteurs du SBN", fontsize=10)
    sub2 = fig.add_gridspec(2, 5, left=0.55, right=0.97, top=0.9, bottom=0.56, wspace=0.1, hspace=0.15)
    for c in range(10):
        a = fig.add_subplot(sub2[c // 5, c % 5]); a.imshow(protos[c].reshape(8, 8), cmap="gray_r")
        a.set_title(str(c), fontsize=8); a.axis("off")
    ax = fig.add_subplot(gs[1, 0])
    names = ["logistique\npixels bruts", "reservoir\n+ readout", "attracteurs\nhebbien",
             "attracteurs\nappris marge", "ideal\n(+proche proto)"]
    vals = [acc_raw, acc_res, accH, accL, acc_tmpl]
    cols = ["#95a5a6", "#2471a3", "#c0392b", "#2980b9", "#27ae60"]
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.1, ls=":", c="grey"); ax.text(4.3, 0.11, "hasard", color="grey", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy test (10 classes)")
    ax.set_title("Deux regimes SBN sur digits + le role du CABLAGE", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(yte, pred_res); ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Matrice de confusion -- reservoir+readout ({acc_res:.2f})", fontsize=10)
    ax.set_xlabel("predit"); ax.set_ylabel("vrai"); ax.set_xticks(range(10)); ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    fig.suptitle("Showcase SBN sur un jeu classique (digits, mini-MNIST) : reservoir competent, "
                 "attention par attracteurs -- et le cablage a marge fait tout", fontsize=12)
    out = FIG_DIR / "showcase_digits.png"; fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("Figure:", out)


if __name__ == "__main__":
    main()
