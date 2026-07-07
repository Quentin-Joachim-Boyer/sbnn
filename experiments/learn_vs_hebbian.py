"""Fabriquer W par APPRENTISSAGE (perceptron a marge, section 4.2) plutot que par
superposition hebbienne, pour la memoire associative pluripotente.

  A. Capacite de points fixes EXACTS : appris >> hebbien (bien au-dela de 0.14 n).
  B. fit != robustesse : sans marge, bassins minuscules ; la MARGE kappa
     (= "robustesse = marge", section 5, devenue objectif d'apprentissage)
     restaure les bassins.
  C. Sous bruit aussi, l'appris a marge domine l'hebbien.
  D. Le budget de POIDS BORNES (wb, signature du projet) fixe la frontiere
     capacite x robustesse.

Usage :
  python experiments/learn_vs_hebbian.py                 # tout + figure
  python experiments/learn_vs_hebbian.py --panel AC|B|D   # 1 panneau (checkpoint json)
  python experiments/learn_vs_hebbian.py --plot           # figure depuis les json
Necessite : pip install -e . (numpy, matplotlib).
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbn_predict.regimes.associative import hebbian_weights, recall      # noqa: E402
from sbn_predict.learning.perceptron import train_perceptron_margin      # noqa: E402
from sbn_predict.core.network import SBN                                 # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
N_MEM, C, G = 64, 16, 3
P_LIST = [2, 4, 6, 8, 10, 12]
TOTALS = [G * P for P in P_LIST]
SEEDS = 3


def build(P_each, seed):
    rng = np.random.default_rng(seed); ctx = rng.integers(0, 2, size=(G, C)); pats = {}; aug = []
    for g in range(G):
        pats[g] = rng.integers(0, 2, size=(P_each, N_MEM))
        for p in pats[g]:
            aug.append(np.concatenate([ctx[g], p]))
    return pats, ctx, np.array(aug)


def learn_W(aug, wb_max, kappa):
    N = aug.shape[1]; W = np.zeros((N, N), int); th = np.zeros(N, int)
    for i in range(C, N):                       # noeuds memoire (contexte clampe)
        w, b = train_perceptron_margin(aug, aug[:, i], wb_max=wb_max, kappa=kappa, epochs=150)
        W[:, i] = w; th[i] = -b
    return SBN(W=W, theta=th)


def recall_mem(net, cue, ctx_vec):
    x = np.concatenate([ctx_vec, cue]); net.clamped = {i: int(ctx_vec[i]) for i in range(C)}
    return np.asarray(recall(net, x, async_order=list(range(C + N_MEM)), max_steps=150))[C:]


def flip(p, f, rng):
    q = p.copy(); idx = rng.choice(len(p), f, replace=False); q[idx] ^= 1; return q


def clean_fp(net, pats, ctx):
    ok = tot = 0
    for g in range(G):
        for p in pats[g]:
            ok += int(np.array_equal(recall_mem(net, p.copy(), ctx[g]), p)); tot += 1
    return ok / tot


def noisy(net, pats, ctx, rng, f=6, tries=6):
    ok = tot = 0
    for g in range(G):
        for p in pats[g]:
            for _ in range(tries):
                ok += int(np.array_equal(recall_mem(net, flip(p, f, rng), ctx[g]), p)); tot += 1
    return ok / tot


def compute_AC(rng):
    hc, lc, hn, ln = [], [], [], []
    for P in P_LIST:
        hcs = lcs = hns = lns = 0.0
        for sd in range(SEEDS):
            pats, ctx, aug = build(P, sd)
            hnet = SBN(W=hebbian_weights(aug)); lnet = learn_W(aug, 60, 16)
            hcs += clean_fp(hnet, pats, ctx); lcs += clean_fp(lnet, pats, ctx)
            hns += noisy(hnet, pats, ctx, rng); lns += noisy(lnet, pats, ctx, rng)
        hc.append(hcs / SEEDS); lc.append(lcs / SEEDS); hn.append(hns / SEEDS); ln.append(lns / SEEDS)
    d = dict(totals=TOTALS, heb_clean=hc, lrn_clean=lc, heb_noisy=hn, lrn_noisy=ln)
    (FIG_DIR / "res_AC.json").write_text(json.dumps(d)); return d


def compute_B(rng):
    kappas = [1, 3, 8, 16, 24, 32]; cur = []
    for k in kappas:
        v = 0.0
        for sd in range(SEEDS):
            pats, ctx, aug = build(4, sd); v += noisy(learn_W(aug, 60, k), pats, ctx, rng)
        cur.append(v / SEEDS)
    d = dict(kappas=kappas, noise=cur)
    (FIG_DIR / "res_B.json").write_text(json.dumps(d)); return d


def compute_D(rng):
    wbs = [3, 6, 12]; loads = [2, 6, 10]; sd_n = 3; curves = {}
    for wb in wbs:
        c = []
        for P in loads:
            v = 0.0
            for sd in range(sd_n):
                pats, ctx, aug = build(P, sd); v += noisy(learn_W(aug, wb, 16), pats, ctx, rng)
            c.append(v / sd_n)
        curves[wb] = c
    d = dict(loads=[G * P for P in loads], curves=curves)
    (FIG_DIR / "res_D.json").write_text(json.dumps(d)); return d


def plot(AC, B, D):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    totals = AC["totals"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0][0]
    a.plot(totals, AC["heb_clean"], "s--", c="#7f8c8d", label="hebbien")
    a.plot(totals, AC["lrn_clean"], "o-", c="#2471a3", label="appris a marge (k=16)")
    a.axvline(0.14 * (C + N_MEM), ls=":", c="grey"); a.text(0.14 * (C + N_MEM) + 0.3, 0.05, "0.14(c+n)", color="grey")
    a.set_xlabel("nb total de motifs"); a.set_ylabel("points fixes EXACTS (indice propre)")
    a.set_title("A. Capacite exacte : l'apprentissage depasse largement l'hebbien")
    a.legend(fontsize=8); a.set_ylim(-0.02, 1.02)
    b = ax[0][1]
    b.plot(B["kappas"], B["noise"], "o-", c="#8e44ad")
    b.set_xlabel("marge d'apprentissage kappa"); b.set_ylabel("rappel sous bruit (6/64), charge=12")
    b.set_title("B. La MARGE = robustesse : restaure les bassins\n(robustesse = marge, section 5, apprise)")
    b.set_ylim(-0.02, 1.02)
    c = ax[1][0]
    c.plot(totals, AC["heb_noisy"], "s--", c="#7f8c8d", label="hebbien")
    c.plot(totals, AC["lrn_noisy"], "o-", c="#2471a3", label="appris a marge (k=16)")
    c.set_xlabel("nb total de motifs"); c.set_ylabel("rappel sous bruit (6/64)")
    c.set_title("C. Sous bruit aussi, l'appris a marge domine")
    c.legend(fontsize=8); c.set_ylim(-0.02, 1.02)
    d = ax[1][1]
    cols = {"3": "#f1c40f", "6": "#e67e22", "12": "#c0392b"}
    for wb, cur in D["curves"].items():
        d.plot(D["loads"], cur, "o-", c=cols[str(wb)], label=f"wb={wb}")
    d.set_xlabel("nb total de motifs"); d.set_ylabel("rappel sous bruit (6/64), k=16")
    d.set_title("D. Le budget de POIDS BORNES fixe la frontiere\ncapacite x robustesse (recule avec wb)")
    d.legend(fontsize=8, title="borne poids"); d.set_ylim(-0.02, 1.02)
    fig.suptitle("Apprendre W (perceptron a marge, section 4.2) vs hebbien -- capacite ET robustesse",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "learn_vs_hebbian.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("Figure:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["AC", "B", "D"])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    if args.panel == "AC":
        print("AC:", compute_AC(rng))
    elif args.panel == "B":
        print("B:", compute_B(rng))
    elif args.panel == "D":
        print("D:", compute_D(rng))
    elif args.plot:
        plot(json.loads((FIG_DIR / "res_AC.json").read_text()),
             json.loads((FIG_DIR / "res_B.json").read_text()),
             json.loads((FIG_DIR / "res_D.json").read_text()))
    else:                                   # tout d'un coup (machine sans contrainte de temps)
        AC = compute_AC(rng); B = compute_B(rng); D = compute_D(rng)
        plot(AC, B, D)


if __name__ == "__main__":
    main()
