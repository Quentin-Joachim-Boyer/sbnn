# Jalon 3 — Apprentissage par résolution ASP (§4.6)

**Verdict : le troisième jalon de la feuille de route est franchi.** L'angle « le
plus publiable » du §4.6 — apprendre les poids d'un nœud à seuil par un solveur —
fonctionne : exact, certifié, avec repli MaxSAT là où le perceptron cale, et un
mur de passage à l'échelle quantifié. clingo 5.8 via l'API Python.

Nouveau code : `src/sbn_predict/learning/asp_solve.py` (`solve_node`, modes
`margin`/`maxsat`), encodage autonome `asp/learn_margin.lp`, expérience
`experiments/milestone3_asp.py`. Figure `figures/milestone3_asp.png`.

Le nœud appris : `f(x)=1 ⟺ Σ wᵢxᵢ > θ`, poids entiers dans [−wb, wb], seuil θ.
Deux modes : **margin** (réalise tous les exemples en maximisant la marge de
stabilité symétrique |S−θ|≥κ = bassins les plus profonds ; UNSAT si non
séparable) et **maxsat** (maximise le nombre d'exemples satisfaits).

## Résultats

**1. Non séparable → MaxSAT exact (le perceptron cale).** Sur XOR (k=2) et
parité (k=3), un nœud à seuil ne peut pas tout réaliser : le perceptron reste à
**0.50** (il oscille, ne converge jamais), tandis qu'ASP-MaxSAT renvoie l'optimum
**exact et certifié** : 3/4 = **0.75**. Sur une cible aléatoire (k=4) : perceptron
0.62 vs ASP **0.88**. Contrôle séparable (majorité k=3) : les deux à 1.0. ASP
donne donc la *meilleure réponse possible* même quand la tâche est irréalisable,
et le certifie — ce qu'aucune méthode en ligne ne fait.

**2. Capacité exacte par nœud vs budget de poids (certifiée).** Nombre maximal de
motifs stockables (marge ≥ 1) par un nœud à k=6 entrées : wb=1 → 5, wb=2 → 9,
puis saturation. ASP détermine **exactement** cette frontière capacité × budget —
la version certifiée du compromis empirique observé au jalon précédent (panneau D
de `learn_vs_hebbian`).

**3. Mur de passage à l'échelle.** Le temps clingo croît exponentiellement avec
le fan-in k (échelle log) : ~10 ms à k=4, ~3 s à k=11 pour wb=2, et bien plus
vite pour wb grand (l'espace des poids est (2wb+1)ᵏ). C'est la limite annoncée
§4.6 : l'ASP est réservé aux **petits nœuds / petits jeux**. Pour les nœuds à
grand fan-in, on garde le perceptron à marge (jalon précédent, approché mais
scalable) — les deux méthodes sont complémentaires.

## Ce que ça apporte au projet

Le jalon 3 complète les deux leviers spécifiques du §4 :

- **§4.3 (énumération exacte de SBF)** et **§4.2 (perceptron à marge)** : exacts
  ou scalables, mais l'un limité à petit k, l'autre approché et sans certificat.
- **§4.6 (ASP, ce jalon)** : exact *et* certifié *et* dégradant proprement
  (MaxSAT) — au prix d'un coût NP borné en taille.

Ensemble ils dessinent une **hiérarchie d'apprentissage** pour les nœuds à seuil
à poids bornés : énumération (k≤5) → ASP (petit k, exact/certifié) → perceptron à
marge (grand k, approché). Et surtout, ASP rend **calculables exactement** les
grandeurs du projet (marge maximale = robustesse, capacité par nœud) qui étaient
jusqu'ici mesurées empiriquement — c'est le sens du programme §5 « recycler la
machinerie en théorie de l'apprentissage ».

## Limites et suites

- **Un seul pas / un seul nœud.** L'encodage actuel apprend un nœud (ou une
  couche de sortie) sur 1 pas. Suites de la feuille de route ASP : dérouler L pas
  (régime A profond, couples les pas de temps) et **contraindre l'attracteur
  atteint** (régime B) — « le plus original » (asp/README §4).
- **Mur NP.** Repousser via symétries, bornes serrées sur θ/κ, ou clingcon
  (arithmétique bornée native) ; sinon rester sur les petits nœuds.
- **Comparaison MaxSAT.** Comparer à un vrai solveur MaxSAT dédié et à la
  binarisation-après-continu (§4.7) comme baselines.
