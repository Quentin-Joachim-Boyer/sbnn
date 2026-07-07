"""Jalon 2 (synthese section 2.B / section 6.2) -- Memoire associative (regime B).

On teste quatre choses :
  1. Rappel sous bruit : depuis un indice bruite, converge-t-on vers le motif stocke ?
  2. Capacite <-> charge : rappel vs P/n ; on cherche la falaise ~0.14 n (Hopfield).
  3. Capacite <-> taille de BASSIN : agregat (bassin moyen vs charge) et par motif
     (profondeur de bassin = tolerance au bruit vers le motif exact vs rappel).
  4. Attention HARD par recurrence : la convergence = selection du motif stocke le
     plus proche par le CONTENU = argmax d'overlap (version discrete de l'attention
     softmax, cf. Ramsauer 2020). On mesure l'accord recurrence <-> argmax-overlap.
Compare SYNCHRONE (Goles-Olivos : points fixes ou 2-cycles) vs ASYNCHRONE
(convergence type Hopfield vers points fixes).

Usage : python experiments/milestone2_associative.py
Necessite : pip install -e . (numpy, matplotlib).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbn_predict.regimes.associative import make_memory, recall  # noqa: E402
from sbn_predict.metrics import basin_robustness                 # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def random_patterns(P, n, rng):
    return rng.integers(0, 2, size=(P, n))


def flip_cue(pattern, n_flips, rng):
    cue = pattern.copy()
    idx = rng.choice(len(pattern), size=n_flips, replace=False)
    cue[idx] ^= 1
    return cue


def recall_success(patterns, n_flips, trials, mode, rng, max_steps=200):
    """Fraction des indices bruites (n_flips bits) qui reviennent EXACTEMENT au
    motif d'origine. mode = 'sync' | 'async'."""
    n = patterns.shape[1]
    net = make_memory(patterns)
    order = list(range(n))
    ok = 0; tot = 0
    for p in patterns:
        for _ in range(trials):
            cue = flip_cue(p, n_flips, rng)
            if mode == "async":
                rng.shuffle(order)
                out = recall(net, cue, async_order=list(order), max_steps=max_steps)
            else:
                out = recall(net, cue, max_steps=max_steps)
            ok += int(np.array_equal(out, p)); tot += 1
    return ok / tot


def attention_agreement(patterns, n_flips, trials, mode, rng, max_steps=200):
    """Accord entre la recurrence et l'attention HARD (argmax d'overlap en +-1)."""
    n = patterns.shape[1]
    net = make_memory(patterns)
    Pm = np.where(patterns > 0, 1, -1)     # motifs en +-1
    order = list(range(n)); agree = 0; tot = 0
    for p in patterns:
        for _ in range(trials):
            cue = flip_cue(p, n_flips, rng)
            cue_pm = np.where(cue > 0, 1, -1)
            hard = int(np.argmax(Pm @ cue_pm))          # attention hard : plus proche par contenu
            if mode == "async":
                rng.shuffle(order)
                out = recall(net, cue, async_order=list(order), max_steps=max_steps)
            else:
                out = recall(net, cue, max_steps=max_steps)
            rec = int(np.argmax(Pm @ np.where(out > 0, 1, -1)))  # motif recurrent le plus proche
            agree += int(rec == hard); tot += 1
    return agree / tot


def mean_basin(patterns, rng_seed):
    net = make_memory(patterns)
    return float(np.mean([basin_robustness(net, p, n_flips=4, trials=30,
                          rng=np.random.default_rng(rng_seed + i))
                          for i, p in enumerate(patterns)]))


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    n = 64
    n_flips = int(round(0.10 * n))
    loads = list(range(1, 15))
    trials, seeds = 25, 4

    def avg(fn):
        out = []
        for P in loads:
            vals = []
            for sd in range(seeds):
                pats = random_patterns(P, n, np.random.default_rng(1000 * sd + P))
                vals.append(fn(pats))
            out.append(float(np.mean(vals)))
        return out

    succ_sync = avg(lambda pats: recall_success(pats, n_flips, trials, "sync", rng))
    succ_async = avg(lambda pats: recall_success(pats, n_flips, trials, "async", rng))
    basin_load = avg(lambda pats: mean_basin(pats, 50))
    agree_async = avg(lambda pats: attention_agreement(pats, n_flips, 15, "async", rng))
    cap = 0.14 * n

    # bassin<->rappel par motif, a une charge PRES de la capacite (variance non nulle)
    P_near = 9
    pats = random_patterns(P_near, n, np.random.default_rng(7))
    net = make_memory(pats)

    # "profondeur de bassin" par motif = tolerance au bruit VERS LE MOTIF EXACT,
    # mesuree a un rayon (4 flips) DIFFERENT du test de rappel (6 flips) -> non circulaire.
    def exact_recall_frac(pp, flips, tries):
        return sum(int(np.array_equal(recall(net, flip_cue(pp, flips, rng),
                   async_order=list(range(n))), pp)) for _ in range(tries)) / tries
    basins = [exact_recall_frac(pp, 4, 50) for pp in pats]
    per_pat_recall = [exact_recall_frac(pp, n_flips, 60) for pp in pats]

    print(f"n={n}, bruit={n_flips}/{n}, capacite ~0.14n={cap:.1f}")
    print("rappel sync :", [round(x, 2) for x in succ_sync])
    print("rappel async:", [round(x, 2) for x in succ_async])
    print("bassin moyen:", [round(x, 2) for x in basin_load])
    print("attention accord async:", [round(x, 2) for x in agree_async])
    print(f"P={P_near} bassins:", [round(b, 2) for b in basins],
          "recall:", [round(r, 2) for r in per_pat_recall])

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0][0]
    a.plot(loads, succ_sync, "o-", label="synchrone", c="#c0392b")
    a.plot(loads, succ_async, "s-", label="asynchrone", c="#2471a3")
    a.axvline(cap, ls="--", c="grey"); a.text(cap + 0.1, 0.05, "~0.14 n", color="grey")
    a.set_xlabel("charge P (nb de motifs)"); a.set_ylabel(f"rappel exact (bruit {n_flips}/{n})")
    a.set_title("1-2. Rappel vs charge : falaise de capacite (moy. 4 tirages)")
    a.legend(); a.set_ylim(-0.02, 1.02)

    b = ax[0][1]
    ln1 = b.plot(loads, succ_async, "s-", c="#2471a3", label="rappel (async)")
    b.set_xlabel("charge P"); b.set_ylabel("rappel exact", color="#2471a3")
    b2 = b.twinx()
    ln2 = b2.plot(loads, basin_load, "^-", c="#27ae60", label="bassin moyen")
    b2.set_ylabel("taille de bassin moyenne", color="#27ae60")
    b.axvline(cap, ls="--", c="grey")
    b.set_title("3. Capacite <-> BASSIN : les deux s'effondrent ensemble")
    lns = ln1 + ln2; b.legend(lns, [l.get_label() for l in lns], loc="upper right", fontsize=8)

    c = ax[1][0]
    c.scatter(basins, per_pat_recall, s=70, c="#27ae60", edgecolor="k")
    c.set_xlabel("profondeur de bassin : rappel exact a 4/64 bits (P=9)")
    c.set_ylabel("succes de rappel du motif")
    c.set_title("3b. Par motif : bassin plus profond -> meilleur rappel a fort bruit")
    if np.std(basins) > 1e-9:
        c.text(0.05, 0.9, f"correlation = {np.corrcoef(basins, per_pat_recall)[0, 1]:+.2f}",
               transform=c.transAxes)

    d = ax[1][1]
    d.plot(loads, agree_async, "s-", c="#8e44ad")
    d.axvline(cap, ls="--", c="grey"); d.text(cap + 0.1, 0.05, "~0.14 n", color="grey")
    d.set_xlabel("charge P"); d.set_ylabel("accord recurrence <-> argmax overlap")
    d.set_title("4. Attention HARD par recurrence\n(fidele SOUS la capacite, se degrade au-dela)")
    d.set_ylim(-0.02, 1.02)

    fig.suptitle("Jalon 2 -- Memoire associative SBN : rappel, capacite, bassins, attention hard",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "milestone2_associative.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("Figure:", out)


if __name__ == "__main__":
    main()
