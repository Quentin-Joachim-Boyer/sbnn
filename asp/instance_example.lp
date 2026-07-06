% Gabarit d'instance : apprendre la fonction AND sur 2 entrées + 1 nœud de biais
% clampé à 1 (nœud 2). Cible : nœud de sortie = AND(x0, x1).
#const n = 3.

% input(Exemple, Noeud, Valeur) — le nœud 2 (biais) vaut toujours 1.
input(0,0,0). input(0,1,0). input(0,2,1).
input(1,0,0). input(1,1,1). input(1,2,1).
input(2,0,1). input(2,1,0). input(2,2,1).
input(3,0,1). input(3,1,1). input(3,2,1).

% target(Exemple, NoeudSortie, Valeur) — ici on apprend le nœud 0 = AND.
target(0,0,0).
target(1,0,0).
target(2,0,0).
target(3,0,1).
