# ASP « attracteur contraint » — concevoir des bassins exacts (§4.6, suite)

**Verdict : concept validé, et cadré honnêtement — c'est le premier pas vers
l'item « le plus original » de la feuille de route ASP, mais le mur NP est proche.**

L'idée (note de cadrage §7.1) : apprendre W par solveur non pas pour un seul pas,
mais pour que des motifs soient des **points fixes avec un bassin prescrit** — les
indices proches convergent vers leur cible sur L pas synchrones. En déroulant la
dynamique, le problème **couple les pas de temps et les colonnes** : il ne se
décompose plus par nœud (contrairement au stockage de points fixes seul), ce qui
en fait un vrai problème de solveur. Nouveau code :
`src/sbn_predict/learning/asp_attractor.py` (`learn_attractor`), expérience
`experiments/asp_attractor_basins.py`, figure `figures/asp_attractor_basins.png`.

## Résultats

**1. Le problème est réel (motivation).** Le stockage **par nœud** (chaque motif
rendu point fixe avec marge) fait **interférer les bassins** de cibles proches :
la fraction d'indices qui convergent vers le *bon* attracteur (rayon 1) chute avec
le nombre de cibles proches (Hamming 2, n=6) — **1.0 → 0.72 → 0.61 → 0.38** pour
1, 2, 3, 4 cibles. Rendre chaque motif stable localement ne garantit pas des
bassins non-interférents.

**2. L'ASP conçoit un bassin exactement.** Pour une cible, `learn_attractor`
trouve W (poids entiers bornés) tel que **tous** les voisins à 1 flip convergent
vers elle en ≤ 2 pas ; vérifié par simulation de la dynamique = **1.0**. Ça
fonctionne aussi au rayon 2 (voisins à 2 flips), là encore vérifié. Le solveur
*conçoit* donc le bassin, ce que ni l'hebbien ni le par-nœud ne garantissent.

**3. Le mur NP arrive vite.** Le déroulement fait exploser la taille. Pour une
cible et un bassin de rayon 1, clingo passe de ~0.2 s (n=4) à ~4.5 s (n=8) —
croissance exponentielle. Et pour **2 cibles proches**, le problème devient
**indécidable en pratique** (> 15 s sans réponse, dès n=5). C'est exactement
l'avertissement §4.6 : l'apprentissage contraint par l'attracteur est le plus
original *et* le plus dur.

## Lecture

Le résultat est une **tension féconde**, pas un échec : le régime où l'ASP est
*nécessaire* (cibles proches aux bassins interférents, que le par-nœud rate) est
précisément celui où le problème est *le plus dur*. Autrement dit, la conception
exacte de bassins par solveur est un objet bien défini et démontré, mais dont la
frontière de tractabilité est étroite avec l'encodage naïf. C'est un énoncé
difficile à qualifier de « déjà connu » — et donc un bon candidat pour la
contribution centrale, à condition de repousser le mur.

## Suites (comment rendre ça exploitable)

- **Encodage plus malin** pour repousser le mur : imposer **W symétrique**
  (moitié moins de variables, cohérent Goles-Olivos/Hopfield) ; bornes serrées
  sur θ ; **clingcon** (arithmétique bornée native) au lieu du choix explicite de
  poids ; casser les symétries.
- **Objectif souple** : passer de SAT dur à **MaxSAT** (maximiser la fraction de
  voisins bien routés) — donne un résultat même quand le bassin parfait est
  infaisable, et évite les longues preuves d'UNSAT.
- **Pluripotence apprise** : combiner avec des nœuds de contexte clampés pour
  concevoir *exactement* des bassins **dépendant du contexte** (le lien direct
  avec le résultat pluripotence, et l'affirmation centrale de la note de cadrage).
- **Théorie** : caractériser quand des bassins non-interférents existent pour des
  cibles à distance de Hamming donnée sous poids bornés (frontière de faisabilité
  exacte) — un énoncé quantifiable, à la portée d'une petite preuve.
