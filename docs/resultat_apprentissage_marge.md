# Apprendre W (perceptron à marge, §4.2) vs superposition hebbienne

**Verdict : c'est le résultat qui unifie tout le fil.** Fabriquer W par
apprentissage exact au lieu de le superposer (Hebb) fait sauter la limite de
capacité de Hopfield ; et la **marge d'apprentissage** est exactement la
« robustesse = marge » du §5, désormais réglable comme un objectif — ce qui donne
à la fois la capacité *et* la tolérance au bruit, sous la contrainte des poids
entiers bornés (signature du projet).

On reprend la mémoire pluripotente (16 nœuds de contexte clampés + 64 nœuds
mémoire) et on remplace l'hebbien par un **perceptron à marge par nœud**
(`learning.train_perceptron_margin`, ajouté au repo). Voir
`experiments/learn_vs_hebbian.py`, `figures/learn_vs_hebbian.png`.

## Résultats

**A. Capacité de points fixes exacts : appris ≫ hebbien.** L'hebbien cesse de
stocker des points fixes exacts dès ~12 motifs (la falaise 0.14·n habituelle) ;
l'apprentissage à marge garde **100 % de points fixes exacts jusqu'à 36 motifs**
(et au-delà), tant que les motifs restent linéairement séparables par nœud. Le
levier §4.2/§4.3 bat frontalement la règle de Hopfield sur la capacité.

**B. fit ≠ robustesse — et la marge répare ça.** Un perceptron sans marge fabrique
des points fixes à **bassins minuscules** : rappel exact parfait sur indice propre,
mais ~0 sous bruit. En augmentant la marge de stabilité κ, le rappel sous bruit
(6/64 bits) remonte de façon monotone : κ=1 → 0.15, κ=8 → 0.78, κ=16 → 1.0. La
« robustesse = marge = taille de bassin » du §5 n'est plus une observation *a
posteriori* mais un **objectif d'apprentissage** : on creuse les bassins à la
demande.

**C. Sous bruit aussi, l'appris à marge domine.** À κ=16, le rappel sous bruit
reste ~1.0 jusqu'à 12 motifs, ~0.74 à 18, décroît ensuite — là où l'hebbien est
déjà à 0. L'apprentissage à marge domine l'hebbien **sur les deux axes** à la fois
(capacité et robustesse), au lieu du compromis hebbien (petite capacité, bons
bassins *dans* la capacité).

**D. Les poids bornés sont la ressource limitante.** Atteindre une grande marge
exige de gros poids ; comme le projet **borne les poids entiers** (wb), le budget
de poids fixe la frontière capacité × robustesse : à κ=16, wb=3 s'effondre dès 18
motifs, wb=6 tient jusqu'à ~18, wb=12 jusqu'à ~30. La frontière recule
proprement quand on augmente wb. (Au-delà de wb≈12 le gain sature, et de très gros
poids peuvent même durcir les bassins — à creuser.)

## Ce que ça apporte au projet

C'est la synthèse des trois lots précédents :

- Tâche 1 (régime C) : robustesse ≠ performance sur réservoir *fixe*.
- Jalon 2 (régime B) : robustesse = taille de bassin quand la cible est un attracteur.
- Pluripotence : le contexte clampé sélectionne la mémoire (prompt matériel).
- **Ici** : dès qu'on *apprend* W (au lieu de le tirer/superposer), la robustesse
  devient un **paramètre de conception** (la marge κ), et les grandeurs du projet
  (bassins, robustesse, poids bornés) se lisent directement comme capacité, marge
  et budget — exactement le programme « recycler la machinerie pluripotence/
  robustesse en théorie de l'apprentissage » du §5/TL;DR.

Le perceptron à marge est le premier pas concret vers les leviers §4.3
(énumération exacte de SBF) et §4.6 (apprentissage ASP) : ici la marge est réglée
par une règle en ligne ; l'ASP permettrait de la **maximiser exactement** sous
contrainte de poids bornés (problème d'optimisation entière naturel pour clingo).

## Limites et suites

- **Séparabilité par nœud.** Le perceptron n'est exact que si les cibles par nœud
  sont linéairement séparables ; au-delà, il ne converge pas. Mesurer où ça casse
  et basculer sur ASP/MaxSAT (§4.6) pour les cas non séparables.
- **Marge vs bornes.** Formaliser le front de Pareto capacité × robustesse × wb
  (le panneau D en est une coupe) ; c'est un résultat quantifiable et publiable.
- **Non-monotonie à très gros poids** (wb≳32) : vérifier si de très grands poids
  dégradent réellement les bassins (durcissement) ou si c'est du bruit d'échantillon.
- **Comparaison propre à Hopfield** : refaire A–C en codage {−1,+1} avec θ = biais,
  pour situer la capacité par rapport à la borne de Gardner (~2n).
