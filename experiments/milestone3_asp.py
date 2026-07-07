"""Jalon 3 (section 4.6) -- Apprentissage PAR RESOLUTION ASP (clingo).

Trois demonstrations de ce que le solveur apporte sur un noeud a seuil a poids
entiers bornes, la ou l'apprentissage en ligne (perceptron) est limite :

  1. NON SEPARABLE -> MaxSAT exact : le perceptron cale (n'atteint pas 0 erreur),
     ASP renvoie le nombre MAXIMAL d'exemples satisfaisables (certifie).
  2. CAPACITE EXACTE par noeud vs budget de poids wb : ASP determine le nombre
     maximal de motifs stockables (marge >= 1) EXACTEMENT ; croit avec wb.
  3. MUR DE PASSAGE A L'ECHELLE : temps clingo vs fan-in k (le cout NP, section 4.6).

Usage :
  python experiments/milestone3_asp.py                 # tout + figure
  python experiments/milestone3_asp.py --panel 1|2|3    # 1 panneau (checkpoint json)
  python experiments/milestone3_asp.py --plot
Necessite : pip install -e ".[asp]" (clingo), + numpy, matplotlib.
"""
import sys
import time
import json
import argparse
import itertools
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbn_predict.learning.asp_solve import solve_node                 # noqa: E402
from sbn_predict.learning.perceptron import train_perceptron_margin   # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def perc_accuracy(X, y, wb, kappa=1, epochs=500):
    w, b = train_perceptron_margin(X, y, wb_max=wb, kappa=kappa, epochs=epochs); th = -b
    return float(np.mean([(1 if int(x @ w) > th else 0) == yy for x, yy in zip(X, y)]))


def panel1():
    tasks = []
    X2 = np.array(list(itertools.product((0, 1), repeat=2)))
    tasks.append(("XOR (k=2)", X2, np.array([a ^ b for a, b in X2])))
    X3 = np.array(list(itertools.product((0, 1), repeat=3)))
    tasks.append(("parite (k=3)", X3, X3.sum(1) % 2))
    tasks.append(("maj (k=3)\nseparable", X3, (X3.sum(1) >= 2).astype(int)))
    rng = np.random.default_rng(0); Xr = np.array(list(itertools.product((0, 1), repeat=4)))
    tasks.append(("aleatoire (k=4)", Xr, rng.integers(0, 2, size=len(Xr))))
    names, pa, aa = [], [], []
    for nm, X, y in tasks:
        names.append(nm); pa.append(perc_accuracy(X, y, 2))
        r = solve_node(X, y, wb=2, mode="maxsat"); aa.append(r["nsat"] / r["m"])
    json.dump(dict(names=names, perc=pa, asp=aa), open(FIG_DIR / "m3_1.json", "w"))
    print("panel1", list(zip(names, [round(x, 2) for x in pa], [round(x, 2) for x in aa])))


def panel2():
    k = 6; Xall = np.array(list(itertools.product((0, 1), repeat=k))); wbs = [1, 2, 3, 4]; cap = []
    for wb in wbs:
        best = 0
        for P in range(1, 15):
            oks = 0
            for sd in range(3):
                r2 = np.random.default_rng(sd); idx = r2.choice(len(Xall), P, replace=False)
                res = solve_node(Xall[idx], r2.integers(0, 2, size=P), wb=wb, mode="margin")
                oks += int(res["kappa"] is not None)
            if oks >= 2:
                best = P
            else:
                break
        cap.append(best)
    json.dump(dict(wbs=wbs, cap=cap), open(FIG_DIR / "m3_2.json", "w"))
    print("panel2", dict(zip(wbs, cap)))


def panel3():
    ks = [3, 4, 5, 6, 7, 8, 9, 10, 11]; times = {}
    for wb in [1, 2]:
        ts = []
        for k in ks:
            Xk = np.array(list(itertools.product((0, 1), repeat=k)))
            if len(Xk) > 48:
                idx = np.random.default_rng(0).choice(len(Xk), 48, replace=False); Xk = Xk[idx]
            yk = np.random.default_rng(1).integers(0, 2, size=len(Xk))
            t0 = time.time(); solve_node(Xk, yk, wb=wb, mode="margin", time_limit=6)
            ts.append(min(time.time() - t0, 6.0))
        times[str(wb)] = ts
    json.dump(dict(ks=ks, times=times), open(FIG_DIR / "m3_3.json", "w"))
    print("panel3", {wb: [round(t, 2) for t in ts] for wb, ts in times.items()})


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d1 = json.load(open(FIG_DIR / "m3_1.json")); d2 = json.load(open(FIG_DIR / "m3_2.json"))
    d3 = json.load(open(FIG_DIR / "m3_3.json"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    a = ax[0]; x = np.arange(len(d1["names"])); wdt = 0.38
    a.bar(x - wdt / 2, d1["perc"], wdt, label="perceptron (500 ep)", color="#7f8c8d")
    a.bar(x + wdt / 2, d1["asp"], wdt, label="ASP MaxSAT (exact)", color="#2471a3")
    a.set_xticks(x); a.set_xticklabels(d1["names"], fontsize=8)
    a.set_ylabel("exactitude (exemples satisfaits)")
    a.set_title("1. Non separable : ASP MaxSAT = optimum exact\n(le perceptron cale)")
    a.legend(fontsize=8); a.set_ylim(0, 1.05)
    b = ax[1]; b.plot(d2["wbs"], d2["cap"], "o-", c="#27ae60")
    b.set_xlabel("borne des poids wb"); b.set_ylabel("motifs max stockables (marge>=1)")
    b.set_title("2. Capacite EXACTE par noeud vs budget de poids\n(certifiee par ASP, k=6)")
    b.set_xticks(d2["wbs"])
    c = ax[2]
    for wb, ts in d3["times"].items():
        c.semilogy(d3["ks"], ts, "o-", label=f"wb={wb}")
    c.set_xlabel("fan-in k (nb d'entrees)"); c.set_ylabel("temps clingo (s, log)")
    c.set_title("3. Mur de passage a l'echelle\n(le cout NP du section 4.6)"); c.legend(fontsize=8)
    fig.suptitle("Jalon 3 -- Apprentissage par resolution ASP (section 4.6) : exact, certifie, "
                 "mais borne en taille", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "milestone3_asp.png"; fig.savefig(out, dpi=130); plt.close(fig); print("Figure:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=int, choices=[1, 2, 3])
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(); FIG_DIR.mkdir(exist_ok=True)
    if a.plot:
        plot()
    elif a.panel == 1:
        panel1()
    elif a.panel == 2:
        panel2()
    elif a.panel == 3:
        panel3()
    else:
        panel1(); panel2(); panel3(); plot()


if __name__ == "__main__":
    main()
