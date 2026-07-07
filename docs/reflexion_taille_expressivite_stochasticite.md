# Taille, expressivité, difficulté, stochasticité — trois questions

Réponses, appuyées sur `experiments/expressivite_vs_difficulte.py` et
`figures/expressivite_vs_difficulte.png` (reservoir SBN, T=4, wb=5, readout linéaire).

## Q1 — Des petits réseaux (n<10) suffisent-ils pour des tâches un peu complexes ?

**Oui, si la complexité est *structurelle* (peu de bits pertinents, régularité) ;
non, si la tâche est *non structurée* (haute entropie sur beaucoup de bits).** Ce
n'est pas le nombre d'entrées qui décide, c'est la difficulté intrinsèque.

Mesuré (n minimal pour réaliser exactement la tâche via readout linéaire) :

| k (bits) | parité (dure) | bande (modérée) | seuil/majorité (séparable) |
|---|---|---|---|
| 3 | 7 | 10 | 3 |
| 4 | 14 | 21 | 4 |
| 5 | 24 | 22 | 5 |
| 6 | 35 | — (>40) | 6 |

La fonction **séparable** (majorité) se réalise avec **n = k** (donc n<10 tient
largement, même pour beaucoup d'entrées). La **parité** — la fonction la plus non
structurée — exige n ≈ 0.5·2^k : il faut essentiellement **mémoriser la table**,
donc n explose. Distinction clé : un petit réseau troque *le temps* (récurrence,
T pas) contre *la profondeur*, mais il ne dépasse pas ses **2^n états** de mémoire.
Il excelle sur les tâches *séquentiellement* complexes mais *étroites* en mémoire,
et échoue sur celles qui demandent beaucoup de bits parallèles.

## Q2 — L'expressivité scale-t-elle comme la difficulté de calcul ?

**Non — et l'écart est le fait central.** Trois échelles distinctes :

1. **Capacité de représentation** par nœud : super-exponentielle en fan-in
   (32 SBF en d=3, 370 en d=4, 11 292 en d=5). Représenter est *bon marché*.
2. **Coût d'apprentissage EXACT** (trouver W) : recherche dans (2wb+1)^(n²),
   NP-difficile — l'ASP heurte le mur dès n≈8–11 (`docs/resultat_jalon3_asp.md`,
   `resultat_asp_attracteur_contraint.md`). Trouver est *cher*, bien plus vite que
   la représentation ne devient riche.
3. **Coût d'exécution/analyse** : dynamique sur 2^n états.

Conclusion : un petit réseau peut *représenter* une fonction complexe bien avant
qu'on puisse *trouver exactement* ses poids. « Facile à exprimer, dur à trouver »
— c'est précisément la tension qui motive les méthodes approchées (Q3). En régime
reservoir, on la contourne : le réservoir est aléatoire (pas cherché), seul le
readout linéaire est appris (facile) ; le prix est alors une taille n plus grande
pour les tâches dures (Q1) plutôt qu'un coût de recherche.

## Q3 — Peut-on introduire de la stochasticité pour approcher les calculs durs ?

**Oui, c'est la sortie naturelle du mur NP**, sur deux plans, cohérente avec la
règle du projet (« doser la continuité sur les ÉTATS, jamais sur les poids »).

**a. Recherche stochastique des poids** (approxime l'apprentissage dur). Déjà
mesuré : la simple **sélection stochastique** — meilleur de K=20 réservoirs tirés —
réalise la parité avec un réseau bien plus petit que le tirage typique :

| parité k | n médian (typique) | n meilleur-de-20 |
|---|---|---|
| 3 | 8 | 5 |
| 4 | 12 | **6** |
| 5 | 21 | 16 |

On troque du **calcul** (K tirages, ou recuit / évolution — `learning.hill_climb`,
§4.5, MCSBN) contre de la **taille**. C'est l'approximation Monte-Carlo de la
conception exacte que l'ASP ne peut pas passer à l'échelle.

**b. Dynamique stochastique** (approxime l'inférence dure). Mise à jour
asynchrone aléatoire, ou bruit/température type Glauber–Boltzmann : le réseau
**échantillonne** les bassins au lieu de converger de façon déterministe. C'est le
recuit simulé appliqué à la minimisation d'énergie du réseau (un problème
combinatoire dur), et c'est aussi la **relaxation continue** qui relie l'attention
*hard* du régime B à l'attention *soft* (Boltzmann/softmax, Ramsauer 2020) — donc
le pont direct vers la version douce annoncée §2.B, et vers un mouvement
*probabiliste* à l'intérieur d'un **support stable partagé** (cf.
`conception_pluripotence.md`).

La stochasticité ne rend pas le NP facile ; elle transforme une **décision
exacte** en **approximation avec un bouton de température/confiance**, à dégradation
douce. C'est le bon compromis pour ce projet, et une grande partie de l'outillage
existe déjà (asynchrone, `random_sbn`/MCSBN, `hill_climb`).

## À retenir (boussole)

- Petit n suffit pour la difficulté *structurelle*, pas pour l'entropie *brute*.
- Représenter ≪ trouver exactement : l'écart est le vrai problème, pas l'expressivité.
- La stochasticité (recherche + dynamique) est la voie d'approximation, et elle
  coïncide avec le pont hard→soft et avec la navigation des supports stables.
