% ============================================================================
% §4.6 -- Apprentissage PAR RESOLUTION d'un noeud a seuil, AVEC MARGE.
%
% Trouve (w, theta), poids entiers dans [-wb, wb], seuil entier, tels que chaque
% exemple soit realise avec une marge de stabilite symetrique |S - theta| >= K,
% et MAXIMISE K (deepest basin). UNSAT si non separable par un noeud borne.
%
% Semantique du noeud : f(x) = 1  <=>  S = sum_i w_i x_i > theta.
% Marge (Hopfield) : cible 1 -> S - theta >= K ; cible 0 -> theta - S >= K.
%
% Lancer :  clingo asp/learn_margin.lp asp/instance_example.lp \
%             -c wb=2 -c tmax=8 -c kmax=8
% (Variante MaxSAT quand non realisable : remplacer les contraintes dures par
%  #maximize { 1,E : sat(E) } -- voir src/sbn_predict/learning/asp_solve.py.)
%
% Format d'instance attendu (voir asp/instance_example.lp) :
%   #const n = ... .            % nb d'entrees
%   input(E, I, X).             % valeur X de l'entree I dans l'exemple E
%   target(E, Y).               % cible Y (0/1) de l'exemple E pour le noeud appris
% ============================================================================

#const wb = 2.
#const tmax = 8.
#const kmax = 8.

inp(0..n-1).
ex(E) :- input(E,_,_).

weight(-wb..wb).
thetaval(-tmax..tmax).
kappaval(1..kmax).

% un poids par entree, un seuil, une marge
1 { w(I,V) : weight(V) } 1 :- inp(I).
1 { th(T) : thetaval(T) } 1.
1 { k(K) : kappaval(K) } 1.

% somme ponderee entrante pour l'exemple E
sum(E,S) :- ex(E), S = #sum { X*V,I : input(E,I,X), w(I,V) }.

% contraintes de marge : chaque exemple realise avec |S - theta| >= K
:- target(E,1), sum(E,S), th(T), k(K), S - T < K.
:- target(E,0), sum(E,S), th(T), k(K), T - S < K.

% maximiser la marge (bassins les plus profonds)
#maximize { K : k(K) }.

#show w/2.
#show th/1.
#show k/1.
