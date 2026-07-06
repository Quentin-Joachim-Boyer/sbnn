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
- `instance_example.lp` — gabarit : apprendre AND(x0,x1) avec nœud de biais.

## Feuille de route
1. 1 pas / couche de sortie (fait).
2. Dérouler L pas (régime A profond).
3. Contraindre l'attracteur atteint (régime B) — le plus original.
4. Passage à MaxSAT quand le jeu n'est pas parfaitement réalisable.
