# Pluripotence en régime B — un SBN, plusieurs mémoires sélectionnées par contexte

**Verdict : le pont conceptuel le plus fort du projet tient.** Un unique SBN
encode plusieurs mémoires ; des nœuds de **contexte clampés** (le « prompt
matériel » du §5) sélectionnent laquelle est active. La dynamique route par le
contenu, le contexte reshape le paysage d'attracteurs, et l'ensemble se lit
directement en langage multi-tâche / modularité.

Setup : SBN = 16 nœuds de contexte (code distribué, clampés) + 64 nœuds mémoire.
Stockage hebbien one-shot des motifs augmentés [contexte_g, motif] dans **une seule**
matrice W. Courbes moyennées sur 5 graines. Voir
`experiments/pluripotence_regimeB.py`, `figures/pluripotence_regimeB.png`.

## Résultats

1. **Le contexte décide quand l'indice est ambigu (panneau 1).** En partant d'un
   motif p₀ et en le bruitant : à faible bruit, le *contenu* gagne (rappel de p₀
   ≈ 0.97 même sous mauvais contexte — l'indice suffit) ; à fort bruit, le
   *contexte clampé* l'emporte et tire l'état vers le banc mémoire qu'il désigne.
   **Croisement net vers 25 % de bits bruités.** C'est exactement le comportement
   d'un prompt : il tranche quand le contenu seul est ambigu, pas quand il est
   déjà déterminé.

2. **Le routage survit bien au-delà de la capacité de rappel exact (panneau 2).**
   Le rappel *exact* s'effondre avec la charge (falaise habituelle), mais
   « atteindre le bon banc mémoire » (routage par contexte) reste ≈ 0.8–0.9
   jusqu'à 18 motifs. Autrement dit : même quand le réseau ne peut plus restituer
   le motif précis, le contexte identifie toujours correctement *quelle* mémoire /
   *quelle* tâche est active. La capacité de **routage** ≫ capacité de **rappel exact**.

3. **Le contexte reshape les bassins (panneau 3).** Pour des indices aléatoires
   (maximalement ambigus), l'attracteur atteint diffère selon le contexte clampé
   dans **97 %** des cas. Le même réseau, avec deux contextes clampés différents,
   se comporte comme deux sous-réseaux distincts — la définition opérationnelle de
   la pluripotence/décomposition du §5, matérialisée.

4. **Interférence vs nombre de contextes (panneau 4).** À P=2 motifs par contexte,
   le rappel du bon contexte reste ≈ 1.0 jusqu'à G=3–4 contextes puis chute — la
   limite de capacité de la superposition hebbienne unique.

## Lien avec la consigne (§5, §7.5)

C'est le résultat qui réconcilie tout le reste. La tâche 1 avait montré que
robustesse/bassins ne prédisent pas la performance d'un readout en régime C ; le
jalon 2 avait montré que bassins = tolérance au bruit en régime B. Ici on va plus
loin : les **nœuds de contrôle = prompts matériels** transforment un même SBN en
plusieurs fonctions selon un contexte figé — le « pont conceptuel le plus riche et
le plus spécifique » annoncé §5. La robustesse (taille de bassin) devient une
grandeur de **modularité/multi-tâche** : chaque contexte creuse ses propres bassins.

## Limites et suites

- **Superposition hebbienne unique.** Le gating achète du routage, pas de la
  capacité de rappel exact (crosstalk résiduel entre contextes dans W). Tester un
  stockage moins interférent (motifs orthogonalisés, ou apprentissage exact par
  énumération SBF §4.3 / ASP §4.6 pour *fabriquer* W au lieu de le superposer).
- **Codage {0,1} + θ=0.** Refaire avec θ = nœud de biais pour rapprocher du cadre
  Hopfield {−1,+1} et mesurer proprement la capacité par contexte.
- **Contexte continu / soft.** Interpoler entre contextes (états continus, poids
  discrets) : le croisement du panneau 1 deviendrait une transition douce — le
  lien vers l'attention soft / transformer (§2.B).
- **Décodage explicite du contexte.** Ici le contexte est *fourni* (clampé) ;
  étape suivante : *inférer* le contexte depuis l'indice (le réseau choisit son
  propre prompt), ce qui rapproche de l'attention par adressage complet.
