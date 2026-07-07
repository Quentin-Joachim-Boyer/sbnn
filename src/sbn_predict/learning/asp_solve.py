"""Section 4.6 -- Apprentissage PAR RESOLUTION (ASP / clingo) d'un noeud a seuil.

f(x) = 1  <=>  sum_i w_i x_i > theta,  w_i in [-wb, wb] entiers, theta entier.

Deux modes :
  - "margin"  : trouve (w, theta) qui realise TOUS les exemples avec la MARGE de
                stabilite symetrique |S - theta| >= kappa MAXIMALE (Hopfield).
                UNSAT (kappa=None) si non separable par un noeud a seuil borne.
  - "maxsat"  : maximise le NOMBRE d'exemples satisfaits (marge >= 1) -- repli
                exact quand le jeu n'est pas parfaitement realisable.

Exact (poids entiers = domaine fini) et CERTIFIE (optimum prouve), au prix d'un
cout NP : petits noeuds / petits jeux (cf. section 4.6). Necessite `clingo`
(pip install clingo ; extra [asp]).
"""
from __future__ import annotations
import numpy as np

_ENCODING = """
weight(-wb..wb).
thetaval(-tmax..tmax).
kappaval(1..kmax).

1 { w(I,V) : weight(V) } 1 :- inp(I).      % un poids par entree
1 { th(T) : thetaval(T) } 1.               % un seuil

sum(E,S) :- ex(E), S = #sum { X*V,I : xval(E,I,X), w(I,V) }.

#show w/2.
#show th/1.
"""

_MARGIN = """
1 { k(K) : kappaval(K) } 1.
% marge de stabilite symetrique |champ - theta| >= K (K>=1) ; champ = S - theta.
:- tgt(E,1), sum(E,S), th(T), k(K), S - T < K.
:- tgt(E,0), sum(E,S), th(T), k(K), T - S < K.
#maximize { K : k(K) }.
#show k/1.
"""

_MAXSAT = """
% marge >= 1 ; on maximise le nombre d'exemples satisfaits (SBF : S>theta -> 1).
sat(E) :- tgt(E,1), sum(E,S), th(T), S - T >= 1.
sat(E) :- tgt(E,0), sum(E,S), th(T), T - S >= 0.
#maximize { 1,E : sat(E) }.
#show sat/1.
"""


def _instance(X, y):
    lines = []
    _, k = X.shape
    for i in range(k):
        lines.append(f"inp({i}).")
    for e in range(X.shape[0]):
        lines.append(f"ex({e}).")
        for i in range(k):
            lines.append(f"xval({e},{i},{int(X[e, i])}).")
        lines.append(f"tgt({e},{int(y[e])}).")
    return "\n".join(lines)


def solve_node(X, y, wb=2, tmax=None, kmax=None, mode="margin", time_limit=10):
    """Apprend un noeud a seuil par ASP. Renvoie un dict :
      weights (np.ndarray|None), theta (int|None), kappa (int|None, mode margin),
      nsat (int, mode maxsat), m (nb exemples), satisfiable (bool), timeout (bool).
    """
    import clingo
    X = np.asarray(X, int); y = np.asarray(y, int)
    k = X.shape[1]
    tmax = tmax if tmax is not None else k * wb
    kmax = kmax if kmax is not None else k * wb
    prog = (f"#const wb={wb}.\n#const tmax={tmax}.\n#const kmax={kmax}.\n"
            + _ENCODING + (_MARGIN if mode == "margin" else _MAXSAT)
            + "\n" + _instance(X, y))
    ctl = clingo.Control(["--opt-mode=opt"])
    ctl.add("base", [], prog)
    ctl.ground([("base", [])])
    best = {}

    def on_model(m):
        w = {}; th = None; kap = None; sat = 0
        for a in m.symbols(shown=True):
            if a.name == "w":
                w[a.arguments[0].number] = a.arguments[1].number
            elif a.name == "th":
                th = a.arguments[0].number
            elif a.name == "k":
                kap = a.arguments[0].number
            elif a.name == "sat":
                sat += 1
        best.update(dict(w=w, theta=th, kappa=kap, nsat=sat))

    with ctl.solve(on_model=on_model, async_=True) as handle:
        finished = handle.wait(time_limit)
        if not finished:
            handle.cancel()
        res = handle.get()
    weights = (np.array([best.get("w", {}).get(i, 0) for i in range(k)], int)
               if "w" in best else None)
    return dict(weights=weights, theta=best.get("theta"), kappa=best.get("kappa"),
                nsat=best.get("nsat", 0), m=X.shape[0],
                satisfiable=bool(res.satisfiable), timeout=not finished)


def node_output(w, theta, x):
    return 1 if int(np.dot(w, x)) > theta else 0
