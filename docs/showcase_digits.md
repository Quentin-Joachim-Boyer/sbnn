# Showcase — SBN sur un jeu classique (digits / mini-MNIST)

Démonstration de bout en bout de la machinerie du projet sur un benchmark reconnu
(sklearn `digits` : 1797 images 8×8, 10 classes). Pixels binarisés (seuil), split
train/test 70/30. Voir `experiments/showcase_digits.py` et
`figures/showcase_digits.png`. **Démo conceptuelle, pas une course à la performance.**

## Résultats (accuracy test, 10 classes)

| Méthode | Régime | Accuracy |
|---|---|---|
| Logistique sur pixels bruts | baseline | **0.93** |
| Réservoir SBN + readout | C | **0.93** |
| Attracteurs, câblage **hebbien** | B | 0.10 (≈ hasard) |
| Attracteurs, câblage **appris à marge** | B | 0.59 |
| Idéal « plus proche prototype » | plafond | 0.78 |

## Lecture — c'est la session en miniature

- **Le réservoir SBN classe les chiffres au niveau de la baseline** (0.93) : la
  machinerie tourne sur du vrai. Il n'améliore pas la baseline (les chiffres sont
  déjà quasi séparables en pixels) — cohérent avec la tâche 1 : le réservoir
  apporte de l'expressivité, pas magiquement de la performance.
- **Le SBN comme classifieur par attention/attracteurs** : on stocke un prototype
  par classe (attracteurs, cf. images), on laisse converger une image test, on lit
  la classe du prototype atteint. Le câblage **hebbien s'effondre** (0.10 ≈ hasard :
  10 prototypes corrélés dépassent la capacité 0.14·n ≈ 9 → interférence totale),
  tandis que le câblage **appris à marge remonte à 0.59** (plafond « plus proche
  prototype » = 0.78). **Le câblage fait tout** — exactement le résultat du test
  d'attention (`docs/test_attention.md`) et de l'apprentissage à marge
  (`docs/resultat_apprentissage_marge.md`).
- L'écart appris-à-marge (0.59) au plafond template-matching (0.78) = ce qu'il
  reste d'états parasites / bassins interférents sur des prototypes corrélés — le
  levier restant est ASP / bassins non-interférents (`resultat_asp_attracteur_contraint.md`)
  et le chevauchement des supports stables (`conception_pluripotence.md`).

## En une phrase

Sur un jeu classique, le SBN prédictif fonctionne : réservoir compétent au niveau
d'une baseline linéaire, et attention native par attracteurs — dont l'efficacité
n'est réelle qu'avec un câblage à marge. La performance brute n'est pas le point ;
la cohérence de bout en bout des concepts du projet, si.
