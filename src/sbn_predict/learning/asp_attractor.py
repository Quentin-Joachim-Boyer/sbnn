"""Section 4.6 (suite) -- ASP 'ATTRACTEUR CONTRAINT' : apprendre W borne tel que
des motifs cibles soient des points fixes AVEC un bassin prescrit (les indices a
<= `radius` flips convergent vers leur cible en <= L pas synchrones).

La dynamique deroulee sur L pas COUPLE les pas de temps et les colonnes : le
probleme ne se decompose PAS par noeud (contrairement au stockage de points
fixes seul). C'est donc un probleme reellement propre au solveur -- et le plus
original de la feuille de route ASP (contraindre l'attracteur atteint, regime B).

Attention : NP-difficile, et le deroulement fait exploser la taille -- ne passe
a l'echelle que pour de tres petits reseaux (n <= ~8 pour un attracteur, et deja
intraitable pour plusieurs cibles proches). Voir experiments/asp_attractor_basins.py.

Necessite `clingo` (extra [asp]).
"""
from __future__ import annotations
import itertools
import numpy as np

_ENC = """
node(0..n-1).
weight(-wb..wb).
thetaval(-tmax..tmax).
time(0..steps-1).

1 { w(I,J,V) : weight(V) } 1 :- node(I), node(J).      % matrice W complete
1 { th(J,T) : thetaval(T) } 1 :- node(J).              % un seuil par noeud

st(C,0,J,X) :- init(C,J,X).                            % etat initial de l'indice C
sum(C,T,J,S) :- cue(C), time(T), node(J),
                S = #sum { X*V,I : st(C,T,I,X), w(I,J,V) }.
st(C,T+1,J,1) :- sum(C,T,J,S), th(J,TH), S > TH.       % dynamique synchrone deroulee
st(C,T+1,J,0) :- sum(C,T,J,S), th(J,TH), S <= TH.

% chaque indice C doit atteindre sa cible goal(C,.) au pas lgoal(C)
:- goal(C,J,Y), lgoal(C,L2), st(C,L2,J,SV), SV != Y.

#show w/3.
#show th/2.
"""


def learn_attractor(targets, radius=1, wb=1, L=2, tmax=None, time_limit=20):
    """Cherche (W, theta), poids entiers dans [-wb, wb], tels que chaque motif de
    `targets` soit un point fixe et que tous ses voisins a <= `radius` flips
    convergent vers lui en <= L pas synchrones.

    Renvoie (W, theta, satisfiable, timeout, n_cues). W/theta = None si non trouve.
    """
    import clingo
    targets = [np.asarray(t, int) for t in targets]
    n = len(targets[0]); tmax = tmax if tmax is not None else n * wb
    facts = [f"#const n={n}.", f"#const wb={wb}.", f"#const tmax={tmax}.", f"#const steps={L}."]
    cues = []; cid = 0
    for t in targets:
        cues.append((cid, t, t, 1)); cid += 1          # le motif : point fixe (1 pas)
        for r in range(1, radius + 1):
            for combo in itertools.combinations(range(n), r):
                c = t.copy()
                for j in combo:
                    c[j] ^= 1
                cues.append((cid, c, t, L)); cid += 1   # voisin : converge en <= L pas
    lines = list(facts)
    for c, init, goal, lg in cues:
        lines.append(f"cue({c}).")
        lines.append(f"lgoal({c},{lg}).")
        for j in range(n):
            lines.append(f"init({c},{j},{int(init[j])}).")
            lines.append(f"goal({c},{j},{int(goal[j])}).")
    prog = "\n".join(lines) + _ENC
    ctl = clingo.Control()
    ctl.add("base", [], prog); ctl.ground([("base", [])])
    got = {}

    def on_model(m):
        W = np.zeros((n, n), int); th = np.zeros(n, int)
        for a in m.symbols(shown=True):
            if a.name == "w":
                W[a.arguments[0].number, a.arguments[1].number] = a.arguments[2].number
            elif a.name == "th":
                th[a.arguments[0].number] = a.arguments[1].number
        got["W"] = W; got["theta"] = th

    with ctl.solve(on_model=on_model, async_=True) as h:
        fin = h.wait(time_limit)
        if not fin:
            h.cancel()
        res = h.get()
    return (got.get("W"), got.get("theta"), bool(res.satisfiable), (not fin), len(cues))
