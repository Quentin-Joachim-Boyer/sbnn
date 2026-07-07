"""ASP 'attracteur contraint' (section 4.6, suite) -- conception EXACTE de bassins.

Demonstration honnete en trois temps :
  1. LE PROBLEME : le stockage par-noeud fait INTERFERER les bassins de cibles
     proches (un voisin de y1 converge vers y2). Le bassin correct chute avec le
     nombre de cibles proches.
  2. CE QUE L'ASP APPORTE : il conçoit le bassin d'une cible EXACTEMENT (verifie
     par simulation = 1.0) en contraignant la dynamique deroulee.
  3. LA LIMITE : le probleme couple (deroulement L pas) heurte le mur NP tres
     vite -- deja intraitable pour 2 cibles proches.

Usage : python experiments/asp_attractor_basins.py
Necessite : pip install -e ".[asp]" + numpy, matplotlib.
"""
import sys
import time
import itertools  # noqa: F401  (utilise indirectement via close_targets)
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbn_predict.learning.asp_attractor import learn_attractor          # noqa: E402
from sbn_predict.core.network import SBN                                # noqa: E402
from sbn_predict.learning.perceptron import train_perceptron_margin     # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def correct_basin(W, th, targets, L=8):
    net = SBN(W=W, theta=th); n = len(targets[0]); ok = tot = 0
    for y in targets:
        for j in range(n):
            c = y.copy(); c[j] ^= 1
            cyc = net.attractor_sync(c, max_steps=L + 5)
            ok += int(len(cyc) == 1 and np.array_equal(np.array(cyc[0]), y)); tot += 1
    return ok / tot


def pernode_W(targets, kappa, wb=2):
    n = len(targets[0]); W = np.zeros((n, n), int); th = np.zeros(n, int); X = np.array(targets)
    for j in range(n):
        w, b = train_perceptron_margin(X, X[:, j], wb_max=wb, kappa=kappa, epochs=300)
        W[:, j] = w; th[j] = -b
    return W, th


def close_targets(P, seed, n=6, dist=2):
    rng = np.random.default_rng(seed); y0 = rng.integers(0, 2, size=n); ts = [y0]; g = 0
    while len(ts) < P and g < 500:
        g += 1; c = y0.copy(); idx = rng.choice(n, dist, replace=False); c[idx] ^= 1
        if not any(np.array_equal(c, t) for t in ts):
            ts.append(c)
    return ts


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)

    # A) interference par-noeud : bassin correct vs nb de cibles proches
    P_list = [1, 2, 3, 4]
    pn = [float(np.mean([max(correct_basin(*pernode_W(close_targets(P, sd), kap), close_targets(P, sd))
                             for kap in [1, 2, 3]) for sd in range(3)])) for P in P_list]

    # B) mur NP : ASP cible unique rayon 1, temps vs n (verifie = 1.0)
    ns = [4, 5, 6, 7, 8]; times = []; verified = []
    rng = np.random.default_rng(3)
    for n in ns:
        y = rng.integers(0, 2, size=n); t0 = time.time()
        W, th, sat, to, nc = learn_attractor([y], radius=1, wb=2, L=2, time_limit=10)
        times.append(time.time() - t0)
        verified.append(correct_basin(W, th, [y]) if sat and W is not None else 0.0)

    print("A) par-noeud bassin correct", dict(zip(P_list, [round(x, 2) for x in pn])))
    print("B) ASP temps vs n", dict(zip(ns, [round(t, 2) for t in times])), "verifie", verified)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    a = ax[0]
    a.plot(P_list, pn, "s--", c="#7f8c8d", label="par-noeud (point fixe + marge)")
    a.axhline(1.0, ls=":", c="#2471a3")
    a.plot([1], [1.0], "o", c="#2471a3", ms=9, label="ASP (bassin conçu, 1 cible, verifie)")
    a.set_xlabel("nb de cibles PROCHES stockees (Hamming 2, n=6)")
    a.set_ylabel("indices -> BON attracteur (rayon 1)")
    a.set_title("Le probleme : les bassins par-noeud INTERFERENT\n(l'ASP conçoit un bassin exact, mais...)")
    a.legend(fontsize=8); a.set_ylim(0, 1.05); a.set_xticks(P_list)

    b = ax[1]
    b.semilogy(ns, times, "o-", c="#c0392b", label="1 cible, rayon 1 (SAT, verifie 1.0)")
    b.axhline(15, ls="--", c="grey")
    b.text(4.1, 17, "2 cibles proches : indecis > 15 s (mur)", color="grey", fontsize=8)
    b.set_xlabel("taille du reseau n"); b.set_ylabel("temps clingo (s, log)")
    b.set_title("... le mur NP arrive vite\n(dynamique deroulee = probleme couple)")
    b.legend(fontsize=8); b.set_xticks(ns)

    fig.suptitle("ASP 'attracteur contraint' : concevoir des bassins exacts que le par-noeud ne "
                 "garantit pas -- concept valide, mais borne par le cout NP", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "asp_attractor_basins.png"; fig.savefig(out, dpi=130); plt.close(fig)
    print("Figure:", out)


if __name__ == "__main__":
    main()
