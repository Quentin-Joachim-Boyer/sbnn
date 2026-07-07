# Jalon 2 — Mémoire associative (régime B) : rappel, capacité, bassins, attention hard

**Verdict : VALIDÉ.** Le régime B fonctionne comme attendu et matérialise
proprement l'attention *hard* par récurrence. Stockage hebbien one-shot de motifs
comme attracteurs, rappel adressable par le contenu, falaise de capacité au bon
endroit, et un lien direct capacité ↔ taille de bassin.

Setup : n=64 nœuds, motifs binaires aléatoires, bruit d'indice 6/64 bits (~10 %),
courbes moyennées sur 4 tirages. Voir `experiments/milestone2_associative.py` et
`figures/milestone2_associative.png`.

## Résultats

1. **Rappel sous bruit (panneau 1).** Rappel exact ≈ 1.0 jusqu'à P≈4 motifs,
   ~0.8–0.9 à P=5–6, puis chute à 0.44 (P=7) et ~0.15 (P=9). **Falaise de
   capacité** située juste sous la borne de Hopfield 0.14·n ≈ 9 — cohérent avec
   la théorie (le seuil de rappel *exact* est un peu plus bas que la capacité de
   stabilité brute).

2. **Synchrone ≈ asynchrone.** Les deux régimes rappellent de façon quasi
   identique. Point important vis-à-vis de la question ouverte §7.1 : pour le
   *rappel de motifs stockés* (codage {0,1}, W hebbien symétrique), le piège des
   2-cycles synchrones (Goles–Olivos) **ne se déclenche pas** en pratique — les
   motifs propres sont des points fixes atteints aussi bien en synchrone. Le choix
   sync/async n'est donc pas critique pour ce régime ; il le redeviendra si l'on
   stocke des cibles qui ne sont pas des points fixes.

3. **Capacité ↔ taille de bassin (panneaux 3 et 3b).** À l'agrégat, le rappel et
   la taille de bassin moyenne (`metrics.basin_robustness`, 4 flips) **s'effondrent
   ensemble** quand la charge augmente. Par motif, près de la capacité (P=9), la
   *profondeur de bassin* (tolérance au bruit vers le motif exact, mesurée à un
   rayon 4/64 différent du test de rappel 6/64) prédit le rappel avec une
   **corrélation +0.99**. La reformulation §5 « robustesse = taille de bassin =
   tolérance au bruit » est donc **exacte dans le régime B** — contrairement au
   régime C (tâche 1), où le lien à la performance ne tenait pas. La différence
   est nette : en régime B la cible *est* l'attracteur, donc la taille de bassin
   mesure directement la marge de rappel.

   *Subtilité relevée :* au-delà de la capacité, certains motifs cessent d'être
   des points fixes exacts (rappel exact = 0) tout en gardant un attracteur voisin
   robuste — d'où la nécessité de mesurer la profondeur de bassin *vers le motif
   exact* et non vers l'attracteur, pour un proxy fidèle.

4. **Attention HARD par récurrence (panneau 4).** On formalise l'attention hard
   comme argmaxⱼ ⟨indice, motifⱼ⟩ (adressage par contenu, ±1). L'accord entre le
   motif atteint par la **récurrence** et ce **argmax d'overlap** est ≈ 1.0 sous
   la capacité, puis se dégrade au-delà (états parasites). Autrement dit : **sous
   la capacité, la dynamique SBN implémente exactement une attention hard** —
   sélection du motif stocké le plus proche par le contenu. C'est la version
   discrète, native (sans relaxation) de l'attention softmax des Modern Hopfield
   Networks (Ramsauer 2020) : la relaxation continue de cet argmax *est* l'attention
   transformer.

## Ce que ça donne au projet

Le régime B est le pont conceptuel le plus solide : il réutilise directement
bassins/robustesse (qui prédisent ici *vraiment* la tolérance au bruit), relie la
capacité mémoire à la théorie de Hopfield, et fournit une définition matérielle,
énumérable, de l'attention hard. C'est cohérent avec la recommandation §2.B/§7.5
(contribution conceptuelle plutôt que performance brute).

## Limites et suites

- **Codage {0,1} + θ=0.** L'état nul reste point fixe et le codage asymétrique
  déplace le biais (synthese §0.3). Refaire avec θ = nœud de biais pour comparer
  proprement à la littérature Hopfield {−1,+1}, et voir si la capacité se rapproche
  de 0.14·n.
- **Motifs aléatoires non corrélés.** Tester la diaphonie sur motifs corrélés et
  le lien avec la **pluripotence** (plusieurs « mémoires » / contextes dans un même
  réseau, nœuds de contrôle = prompts matériels, §5).
- **Attention hard → soft.** Prochaine étape naturelle : doser la continuité sur
  les ÉTATS (softmax type Ramsauer) tout en gardant les poids discrets, pour
  matérialiser le passage attention hard → attention transformer annoncé §2.B.
- **Capacité de rappel exact vs stabilité.** La falaise du rappel exact est un peu
  avant 0.14·n ; quantifier l'écart entre « motif stable » et « motif rappelé
  exactement sous bruit ».
