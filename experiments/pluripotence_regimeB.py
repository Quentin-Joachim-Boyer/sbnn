"""Pluripotence en regime B -- un seul SBN encode plusieurs memoires selon un
CONTEXTE clampe (noeuds de controle = "prompts materiels", synthese section 5).

Mecanisme : SBN = c noeuds de contexte (clampes, code distribue) + n noeuds
memoire. Chaque motif p est stocke sous un contexte g par Hebbien sur [ctx_g, p],
tout dans une SEULE matrice W. Au rappel : clamp du contexte -> l'attracteur
accessible depend du contexte = un reseau, plusieurs sous-reseaux.

Quatre mesures (moyennes sur plusieurs graines) :
  1. Quand le CONTEXTE decide : vs l'ambiguite de l'indice. Quand le contenu est
     clair, il gagne ; quand l'indice devient ambigu, le contexte clampe l'emporte
     (croisement) -- comportement de "prompt materiel".
  2. ROUTAGE > capacite de rappel exact : atteindre le bon banc memoire survit
     bien au-dela de la falaise de rappel exact.
  3. Remodelage de bassin : meme indice, contexte A vs B -> attracteur different.
  4. Interference vs nombre de contextes G.

Usage : python experiments/pluripotence_regimeB.py
Necessite : pip install -e . (numpy, matplotlib).
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbn_predict.regimes.associative import hebbian_weights, recall  # noqa: E402
from sbn_predict.core.network import SBN                             # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
N_MEM, C = 64, 16


def build(G, P_each, n_mem, c, seed):
    rng = np.random.default_rng(seed)
    ctx = rng.integers(0, 2, size=(G, c))       # code de contexte distribue
    pats, aug = {}, []
    for g in range(G):
        pats[g] = rng.integers(0, 2, size=(P_each, n_mem))
        for p in pats[g]:
            aug.append(np.concatenate([ctx[g], p]))
    return SBN(W=hebbian_weights(np.array(aug))), pats, ctx


def recall_mem(net, cue_mem, ctx_vec, c, n_mem):
    x = np.concatenate([ctx_vec, cue_mem])
    net.clamped = {i: int(ctx_vec[i]) for i in range(c)}
    return np.asarray(recall(net, x, async_order=list(range(c + n_mem)), max_steps=300))[c:]


def flip(p, f, rng):
    q = p.copy(); idx = rng.choice(len(p), f, replace=False); q[idx] ^= 1; return q


def nearest(out, patset):
    o = np.where(out > 0, 1, -1); P = np.where(patset > 0, 1, -1)
    return int(np.argmax(P @ o))


def context_dominance(noise_fracs, seeds, rng):
    """Quand le contexte l'emporte-t-il sur le contenu ? On abime un motif p0 de
    ctx0 a differents niveaux de bruit, puis :
      - CORRECT : rappel sous ctx0 -> P(retour a p0) (le contenu porte).
      - PULL    : rappel sous ctx1 (mauvais) -> P(attracteur dans le banc ctx1)
                  (le contexte tire ailleurs). Croise le contenu quand l'indice
                  devient ambigu."""
    correct, pull = [], []
    for nf in noise_fracs:
        cc = pp = tt = 0
        f = max(1, int(round(nf * N_MEM)))
        for sd in range(seeds):
            net, pats, ctx = build(2, 2, N_MEM, C, seed=900 + 7 * sd)
            allp = np.concatenate([pats[0], pats[1]])
            for p0 in pats[0]:
                for _ in range(8):
                    cue = flip(p0, f, rng)
                    o_ok = recall_mem(net, cue, ctx[0], C, N_MEM)
                    cc += int(np.array_equal(o_ok, p0))
                    o_pull = recall_mem(net, cue, ctx[1], C, N_MEM)
                    j = nearest(o_pull, allp); pp += int(j >= 2)   # banc ctx1 = indices 2,3
                    tt += 1
        correct.append(cc / tt); pull.append(pp / tt)
    return correct, pull


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    f = 6
    SEEDS = 5

    noise_fracs = [0.03, 0.08, 0.15, 0.25, 0.35, 0.45]
    dom_correct, dom_pull = context_dominance(noise_fracs, SEEDS, rng)

    # routage + rappel, sweep charge (G=3)
    G = 3
    P_list = [1, 2, 3, 4, 5, 6]
    right, incontext, wrong = [], [], []
    for P_each in P_list:
        rr = ii = ww = tt = 0
        for sd in range(SEEDS):
            net, pats, ctx = build(G, P_each, N_MEM, C, seed=100 * sd + P_each)
            allp = np.concatenate([pats[g] for g in range(G)])
            for g in range(G):
                for pi, p in enumerate(pats[g]):
                    for _ in range(8):
                        cue = flip(p, f, rng)
                        out = recall_mem(net, cue, ctx[g], C, N_MEM)
                        rr += int(np.array_equal(out, p))
                        j = nearest(out, allp); ii += int(g * P_each <= j < (g + 1) * P_each)
                        outw = recall_mem(net, cue, ctx[(g + 1) % G], C, N_MEM)
                        ww += int(np.array_equal(outw, p)); tt += 1
        right.append(rr / tt); incontext.append(ii / tt); wrong.append(ww / tt)
    totals = [G * P for P in P_list]

    # remodelage de bassin (meme indice, contexte A vs B)
    changed = tot = 0
    for sd in range(SEEDS):
        net, pats, ctx = build(3, 2, N_MEM, C, seed=500 + sd)
        for _ in range(60):
            cue = rng.integers(0, 2, size=N_MEM)
            oA = recall_mem(net, cue, ctx[0], C, N_MEM)
            oB = recall_mem(net, cue, ctx[1], C, N_MEM)
            changed += int(not np.array_equal(oA, oB)); tot += 1
    reshape_frac = changed / tot

    # interference vs nombre de contextes
    Gs = [1, 2, 3, 4, 6, 8]
    gright, gwrong = [], []
    for G in Gs:
        r = wr = tt = 0
        for sd in range(SEEDS):
            net, pats, ctx = build(G, 2, N_MEM, C, seed=700 + 10 * sd + G)
            for g in range(G):
                for p in pats[g]:
                    for _ in range(6):
                        cue = flip(p, f, rng)
                        r += int(np.array_equal(recall_mem(net, cue, ctx[g], C, N_MEM), p))
                        wr += int(np.array_equal(recall_mem(net, cue, ctx[(g + 1) % G], C, N_MEM), p))
                        tt += 1
        gright.append(r / tt); gwrong.append(wr / tt)

    print("dominance ctx: noise", noise_fracs, "correct", [round(x, 2) for x in dom_correct],
          "pull", [round(x, 2) for x in dom_pull])
    print("charge totale:", totals)
    print("rappel bon contexte:", [round(x, 2) for x in right])
    print("routage (attracteur bon contexte):", [round(x, 2) for x in incontext])
    print("rappel sous mauvais contexte:", [round(x, 2) for x in wrong])
    print("remodelage bassin:", round(reshape_frac, 2))
    print("G:", Gs, "rappel:", [round(x, 2) for x in gright], "faux ctx:", [round(x, 2) for x in gwrong])

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0][0]
    a.plot(noise_fracs, dom_correct, "o-", c="#2471a3", label="contenu gagne : rappel p0 (bon ctx)")
    a.plot(noise_fracs, dom_pull, "^-", c="#8e44ad", label="contexte gagne : tire vers l'autre banc (mauvais ctx)")
    a.set_xlabel("ambiguite de l'indice (fraction de bits bruites)"); a.set_ylabel("taux")
    a.set_title("1. Quand le CONTEXTE decide : quand l'indice devient ambigu")
    a.legend(fontsize=7); a.set_ylim(-0.02, 1.02)

    b = ax[0][1]
    b.plot(totals, incontext, "^-", c="#27ae60", label="ROUTAGE (bon banc memoire)")
    b.plot(totals, right, "o-", c="#2471a3", label="rappel EXACT")
    b.set_xlabel("nb total de motifs"); b.set_ylabel("taux")
    b.set_title("2. Le ROUTAGE survit bien au-dela\nde la capacite de rappel exact")
    b.legend(fontsize=8); b.set_ylim(-0.02, 1.02)

    c = ax[1][0]
    c.bar(["meme indice\ncontexte A vs B"], [reshape_frac], color="#8e44ad", width=0.5)
    c.set_ylim(0, 1); c.set_ylabel("fraction d'attracteurs differents")
    c.set_title("3. Remodelage de bassin par le contexte\n(un reseau, plusieurs sous-reseaux)")
    c.text(0, reshape_frac + 0.03, f"{reshape_frac:.2f}", ha="center")

    d = ax[1][1]
    d.plot(Gs, gright, "o-", c="#2471a3", label="rappel bon contexte")
    d.plot(Gs, gwrong, "x--", c="#c0392b", label="rappel faux contexte")
    d.set_xlabel("nombre de contextes G (P=2 chacun)"); d.set_ylabel("taux")
    d.set_title("4. Interference vs nombre de contextes"); d.legend(fontsize=8); d.set_ylim(-0.02, 1.02)

    fig.suptitle("Pluripotence en regime B -- un SBN, plusieurs memoires selectionnees par contexte clampe",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / "pluripotence_regimeB.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("Figure:", out)


if __name__ == "__main__":
    main()
