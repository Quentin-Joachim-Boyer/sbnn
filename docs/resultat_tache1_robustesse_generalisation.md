# Tâche 1 — La robustesse prédit-elle la généralisation ? (hypothèse directrice §5)

**Verdict : hypothèse NON confirmée dans sa forme naïve, mais un résultat plus
intéressant émerge.** La robustesse d'un réservoir SBN ne prédit **pas** sa
précision en test. En revanche, elle contrôle fortement la **capacité** du
réservoir : plus un réservoir est robuste, moins son readout surajuste. La
robustesse se comporte comme un **axe de régularisation**, pas de performance.

Fait à K=200 réservoirs par jeu, seed=0. Voir
`experiments/hypothesis_robustness_generalization.py`, figure
`figures/hypo_robustness_generalization.png`, données `figures/hypo_reservoirs.csv`.

## Protocole

Pour chaque réservoir SBN aléatoire fixe (tiré par `random_sbn`) : features
réservoir sur un vrai split train/test, readout logistique entraîné sur le train,
mesure de `acc_test`. Robustesse = `metrics.readout_robustness` : fraction des
flips d'un bit d'entrée qui laissent inchangée la décision du readout (mesurée
sur le train, **sans étiquette de test** — donc utilisable pour *prédire* la
généralisation). Contrôle de la richesse dynamique = rang du noyau (dimension
effective des features), états distincts visités, entropie d'activation.

Trois jeux, tous à structure généralisable (la parité est exclue : rien à
généraliser, tout point retiré est imprévisible) :

| Jeu | Cible | Nature |
|---|---|---|
| `bool_struct` | somme(x) ∈ {2,3} sur 6 bits | symétrique non-monotone, non séparable |
| `bool_band` | somme(x) ∈ {1,4} sur 6 bits | bande non-contiguë, plus dure, structure indépendante |
| `moons` | make_moons binarisé (thermomètre) | contrôle « tâche facile », quasi séparable |

Statistiques sans dépendance externe : Spearman + p-value par permutation (5000),
**corrélation partielle** contrôlant la richesse, régression OLS standardisée.

## Résultats (K=200)

| Jeu | acc_test | ρ(acc, rob) | ρ partiel \| richesse | ρ(rob, acc_train) | ρ(rob, écart) | ρ(rob, richesse) |
|---|---|---|---|---|---|---|
| bool_struct | 0.58 ± 0.08 | +0.09 (p=0.22) | +0.08 (p=0.29) | **−0.83** | −0.42 | −0.12 |
| bool_band | 0.51 ± 0.07 | −0.12 (p=0.09) | −0.08 (p=0.28) | **−0.85** | −0.43 | −0.45 |
| moons | 0.89 ± 0.01 | +0.12 (p=0.11) | +0.09 (p=0.21) | −0.00 | −0.12 | −0.17 |

Lecture :

1. **Robustesse ↮ accuracy test.** Aucune corrélation significative, sur aucun
   jeu, avant ou après contrôle de la richesse. L'hypothèse « les réservoirs
   robustes généralisent mieux » est infirmée au sens de la performance brute.
   *(Un ρ=+0.50 apparaissait à K=20 : c'était un artefact de petit échantillon,
   dissous à K=200. Leçon de puissance statistique.)*

2. **Robustesse ↔ capacité (le vrai signal).** Sur les deux tâches booléennes
   dures, ρ(robustesse, acc_**train**) ≈ **−0.84**. Les réservoirs robustes
   fournissent des features trop lisses pour que le readout surajuste : le train
   chute fort, le test reste plat, donc l'**écart de généralisation se resserre**
   (ρ(rob, écart) ≈ −0.43). C'est le « ciseau » de la rangée du bas de la figure.
   La robustesse agit exactement comme un **régularisateur** (contrôle de
   capacité), pas comme un levier de performance.

3. **Ce n'est pas non plus la richesse.** Le contrôle demandé est doublement
   satisfait : ni la robustesse ni la richesse (rang du noyau, états distincts,
   entropie) ne prédisent l'accuracy test. Pour des réservoirs purement
   aléatoires, la performance en test est surtout un tirage.

4. **Compromis robustesse ↔ richesse confirmé (edge of chaos).** ρ(rob, richesse)
   systématiquement négatif : les deux grandeurs sont des axes opposés, comme le
   prédit la théorie du reservoir computing.

5. **Contrôle « tâche facile ».** Sur moons (quasi séparable), acc ≈ 0.89 sans
   variance : la qualité du réservoir n'y change rien (test ≈ train), résultat
   attendu — quand le readout n'a pas besoin de surajuster, la robustesse ne joue
   plus.

## Interprétation

La reformulation §5 « robustesse ≈ marge ≈ généralisation » est vraie **du côté
de l'écart train/test**, mais fausse du côté de l'accuracy test, parce que le
réservoir est **fixe et non entraîné** : augmenter la robustesse réduit
l'overfitting par le haut (train ↓) sans tirer le test vers le haut. La marge ne
peut améliorer le test que si la capacité était le facteur limitant — or ici le
readout linéaire sur un réservoir aléatoire est déjà limité par la qualité (non
contrôlée) des features, pas par le surajustement.

Conséquence pour le projet : la robustesse (machinerie pluripotence/bassins) est
un bon **outil de sélection de capacité** — utile si un jour on *entraîne* le
réservoir (régimes 4.3/4.5/4.6), là où réduire l'écart de généralisation à
performance de train égale deviendrait un vrai gain.

## Test décisif : « à train égal » (suivi §5)

Le résultat 2 laisse une porte ouverte : si la robustesse ne fait que réduire la
capacité, alors *à accuracy de train égale* elle pourrait quand même aider le
test. C'est le vrai test de l'hypothèse. On isole l'effet de la robustesse en
contrôlant l'accuracy train (corrélation partielle sur rangs, régression
`test ~ train + rob`, et sélection top-robustesse vs bottom-robustesse à
l'intérieur de bandes de train). Voir `experiments/followup_equal_train.py` et
`figures/followup_equal_train.png`.

| Jeu | partiel(test, rob \| train) | β_rob à train égal | gain de sélection (moy.) |
|---|---|---|---|
| bool_struct | **+0.15** (p=0.040) | +0.24 | +0.017 |
| bool_band | **−0.18** (p=0.008) | −0.25 | −0.004 |
| moons | +0.12 (p=0.105, n.s.) | −0.06 | +0.009 |

**Conclusion : effet petit et de signe INSTABLE selon la tâche.** À train égal, la
robustesse aide légèrement sur `bool_struct` (cible = bande *contiguë*, lisse dans
l'espace d'entrée) mais **nuit** sur `bool_band` (cible = bande *non-contiguë*,
haute fréquence), et ne fait rien sur moons. Le gain de sélection maximal est de
~1–2 points d'accuracy, et il change de signe. Interprétation : la robustesse
*lisse* les features — bénéfique quand la cible est lisse, pénalisant quand elle
est irrégulière. Ce n'est donc **pas** un levier de généralisation universel,
même à capacité contrôlée. L'hypothèse §5 ne tient pas dans sa forme forte.

## Limites et suites

- **Métrique de robustesse.** Ici `readout_robustness` (propre au régime C). La
  métrique nommée en §5, `basin_robustness` (dynamique libre), n'a pas été
  testée en corrélation — à ajouter comme colonne secondaire pour vérifier si la
  robustesse *de bassin* raconte la même histoire.
- **Réservoirs non entraînés.** Le test décisif de §5 devrait porter sur des
  réservoirs *sélectionnés/appris* (par robustesse), pas seulement tirés : est-ce
  qu'à performance de train égale, sélectionner le plus robuste améliore le test ?
- **bool_band est trop dure** (test souvent sous le taux de base) : à recalibrer
  (réservoir plus grand ou tâche moins extrême) si on veut y lire un effet fin.
- **Questions ouvertes §7 non encore tranchées ici** : synchrone (utilisé) vs
  asynchrone ; horizon de lecture T fixe (utilisé) vs à convergence ; baseline
  BNN/MLP (le milestone 1 garde la baseline « entrée brute »).
