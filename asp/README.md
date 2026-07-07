# Apprentissage par résolution (ASP) — §4.6

Encode l'apprentissage des poids d'un SBN comme un problème de satisfaction :
« trouver W entier tel que la dynamique réalise les paires (x → y) ».

## Lancer
```bash
clingo learn_weights.lp instance_example.lp        # SAT -> un W valide
# variante MaxSAT (maximiser les contraintes satisfaites) : à ajouter via #maximize
```

## Fichiers
- `learn_weights.lp` — encodage générique (prototype 1 pas, couche de sortie).
- `learn_margin.lp` — **avec marge** : maximise la marge de stabilité |S−θ|≥K
  (bassins les plus profonds), UNSAT si non séparable. (§4.6, jalon 3.)
- `instance_example.lp` — gabarit : apprendre AND(x0,x1) avec nœud de biais.
- Pont Python : `src/sbn_predict/learning/asp_solve.py` (`solve_node`, modes
  `margin` / `maxsat`), utilisé par `experiments/milestone3_asp.py`.

## Ce que le solveur apporte (mesuré, jalon 3)
- **Exact + certifié** : marge maximale prouvée ; certifie l'inséparabilité.
- **MaxSAT** : jeu non réalisable → nombre MAXIMAL d'exemples satisfaits, là où
  le perceptron cale (XOR/parité : 0.75 vs 0.50).
- **Coût NP** : le temps clingo explose avec le fan-in k (mur mesuré).

## Feuille de route
1. 1 pas / couche de sortie (fait).
2. Marge maximisée + MaxSAT (fait — `learn_margin.lp`, jalon 3).
3. Dérouler L pas (régime A profond).
4. Contraindre l'attracteur atteint (régime B) — le plus original.
