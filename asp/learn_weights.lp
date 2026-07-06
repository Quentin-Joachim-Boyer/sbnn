% ============================================================================
% §4.6 — Apprentissage PAR RÉSOLUTION (ASP) : trouver W tel que la dynamique
% réalise les paires (x -> y) du jeu d'entraînement.
%
% Angle le plus original du projet (apprentissage-par-solveur de réseaux à
% seuil, peu exploré). Exact car poids entiers = domaine fini. Limite : NP,
% donc petits réseaux / petits jeux (d = 3-4).
%
% Prototype "1 pas" (feedforward / couche de sortie). Étape suivante :
% dérouler L pas et/ou contraindre l'attracteur (régime B).
%
% Lancer :  clingo asp/learn_weights.lp asp/instance_example.lp
% (clingcon si l'on veut de l'arithmétique bornée native.)
% ============================================================================

#const wb = 2.          % borne des poids (= wb(k))
#const theta = 0.       % seuil

node(0..n-1).
weight(-wb..wb).

% Choisir un poids w(I,J) pour chaque arête I->J (colonne J = SBF du nœud J).
1 { w(I,J,V) : weight(V) } 1 :- node(I), node(J).

% Somme pondérée entrante au nœud J pour l'exemple E :
sum(E,J,S) :- example(E), node(J),
              S = #sum { X*V, I : input(E,I,X), w(I,J,V) }.

% Sortie prédite (seuil strict > theta) :
pred(E,J,1) :- sum(E,J,S), S > theta.
pred(E,J,0) :- sum(E,J,S), S <= theta.

% Contrainte : la prédiction doit égaler la cible sur tous les exemples.
:- target(E,J,Y), pred(E,J,P), Y != P.

#show w/3.

% ---------------------------------------------------------------------------
% Une instance fournit : #const n=... .  input(E,I,X).  target(E,J,Y).
% Voir asp/instance_example.lp pour un gabarit (fonction AND avec biais).
% ---------------------------------------------------------------------------
