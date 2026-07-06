# sbn-predict

Réseaux Booléens à Signe (SBN) comme **réseaux de neurones prédictifs**.
Volet *connexe* à *PluripotentSBN* : mêmes unités (fonctions à seuil, poids
entiers bornés), mais un but différent — **prédiction** (classification /
régression) plutôt qu'exploration du paysage dynamique.

> Un nœud de SBN **est** un neurone à seuil (McCulloch–Pitts). La prédiction est
> abordée sur trois régimes ; les deux leviers spécifiques du projet sont
> l'**ajustement exact par énumération de SBF** et l'**apprentissage-par-solveur
> ASP**. Détails complets : [`docs/SBN_pour_prediction__synthese.md`](docs/SBN_pour_prediction__synthese.md).

## Le formalisme

`f(x) = 1 ⟺ Σᵢ wᵢ·xᵢ > θ`, poids entiers `wᵢ ∈ [−wb(k), wb(k)]`, états `{0,1}`.
Le seuil `θ` (0 dans le formalisme d'origine) est exposé comme paramètre : `θ ≠ 0`
équivaut exactement à un nœud de biais, et fait passer aux fonctions à seuil
générales (LTF). Coût mesuré : d=3 passe de 32 à 104 fonctions, d=4 de 370 à 1882.

## Structure

| Chemin | Rôle |
|---|---|
| `src/sbn_predict/core/` | SBF (seuil, bornes, énumération), dynamique du réseau, encodage E/S |
| `src/sbn_predict/regimes/` | A feedforward · B mémoire associative/attention · C reservoir |
| `src/sbn_predict/learning/` | énumération exacte (§4.3), perceptron, STE, recherche |
| `asp/` | apprentissage-par-résolution en clingo (§4.6) |
| `experiments/` | jalons §6 (go/no-go chiffrés) |
| `tests/` | vérifications exactes du cœur |
| `docs/` | synthèse technique |

## Démarrage

```bash
pip install -e ".[dev]"      # + [asp] pour le pont clingo
pytest                       # 10 tests exacts (comptes SBF, Goles-Olivos, rappel…)
python experiments/milestone1_reservoir.py   # premier go/no-go : reservoir vs baseline
```

## Feuille de route (jalons)

1. **Reservoir** (régime C) — le readout SBN doit battre l'entrée brute. *Test central.*
2. **Mémoire associative** (régime B) — rappel sous bruit ↔ taille de bassin.
3. **Apprentissage ASP** (§4.6) — jusqu'où la résolution exacte passe à l'échelle.

Portage vers un langage plus efficace (Java/Rust) prévu si les jalons Python
valident le pont et que la taille l'exige.

## Statut

Prototype de recherche — v0.0.1. Cœur implémenté et testé ; régimes A/C et STE
en squelettes documentés.
