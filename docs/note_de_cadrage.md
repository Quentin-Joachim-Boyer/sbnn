# Note de cadrage — positionnement, affirmation centrale, plan d'attaque

> But de ce document : figer, après le premier cycle de développement, *où est la
> valeur du projet*, *quelle affirmation défendre*, et *comment l'exploiter*. À
> relire avant chaque nouveau lot pour éviter la dispersion.

## 1. Thèse en une phrase

Dans les réseaux à seuil à **poids entiers bornés**, les grandeurs « pluripotence
/ robustesse / taille de bassin » du projet ne sont pas des métaphores mais **un
même objet — la marge — exactement calculable et apprenable** ; ce qui permet de
recycler la machinerie du projet en théorie de l'apprentissage, *à condition de
ne pas viser la performance brute mais la structure exacte et certifiée*.

## 2. Ce que le développement a établi (rappel condensé)

Quatre lots, un fil conducteur (détails dans les `docs/resultat_*.md`) :

- **Régime C (réservoir fixe)** — la robustesse **ne prédit pas** la performance
  test ; à train égal l'effet est petit et de signe instable. C'est un axe de
  **régularisation/capacité**, pas de performance (ρ(rob, acc_train) ≈ −0.84).
  *Résultat négatif honnête, et utile : il borne les prétentions.*
- **Régime B (mémoire associative)** — l'identité « robustesse = taille de bassin
  = tolérance au bruit » tient **exactement** (bassin ↔ rappel +0.99), parce que
  la cible *est* l'attracteur. Falaise de capacité à 0.14·n, sync ≈ async,
  attention *hard* = argmax d'overlap matérialisé par la récurrence.
- **Pluripotence** — un seul SBN à poids figés, plusieurs mémoires sélectionnées
  par des **nœuds de contexte clampés** (prompts matériels). Le contexte reshape
  les bassins (97 %), tranche quand l'indice est ambigu (croisement contenu/contexte),
  et **le routage survit bien au-delà de la capacité de rappel exact**.
- **Apprentissage (perceptron à marge, puis ASP §4.6)** — apprendre W fait sauter
  la capacité de Hopfield ; la **marge κ devient un bouton** qui arbitre budget de
  poids ↔ profondeur de bassin ; l'ASP rend marge et capacité **exactes et
  certifiées** (MaxSAT quand irréalisable), au prix d'un mur NP.

## 3. Ce qui est distinctif (vs. ce qui est déjà connu)

Séparer clairement, car c'est le nerf de la crédibilité.

**Déjà connu (à citer, pas à revendiquer)** : capacité de Hopfield ~0.14·n ;
limite XOR d'un seul seuil (Minsky–Papert) ; saveur de la borne de Gardner ;
compromis richesse/robustesse « edge of chaos » du reservoir computing ; attention
softmax = Modern Hopfield (Ramsauer 2020) ; expressivité TC⁰.

**Réellement distinctif (à défendre)** :
1. **Pluripotence = prompts matériels en régime exact** : un réseau figé dont des
   nœuds de contrôle clampés sélectionnent la fonction, avec la thèse mesurée
   « **capacité de routage ≫ capacité de rappel exact** » et le croisement
   contenu/contexte. Angle non traité ainsi dans la littérature attention/Hopfield.
2. **Apprentissage-par-solveur certifié de nœuds à seuil bornés** : marge maximale
   *prouvée*, inséparabilité *certifiée*, repli MaxSAT exact — là où le perceptron
   cale. Le solveur rend **exactement calculables** des grandeurs (marge = robustesse,
   capacité par nœud) jusque-là seulement mesurées.
3. **La hiérarchie unifiée** énumération SBF (k≤5) → ASP (petit k, exact/certifié)
   → perceptron à marge (grand k, approché), le tout pour le *même* format de nœud
   à poids entiers bornés, avec la marge comme fil.

## 4. Évaluation lucide du potentiel

- **Nature** : contribution **conceptuelle / théorique / neurosymbolique**, pas une
  méthode d'apprentissage compétitive. Ne pas la vendre en performance.
- **Points forts** : histoire cohérente et *honnête* (le contraste B vs C la rend
  crédible), objets originaux (prompts matériels, solveur certifié), reproductible
  et exact.
- **Points faibles** : petite échelle (n≤64, k≤11 ASP) ; capacités modestes ;
  recouvre partiellement du connu ; l'hypothèse directrice §5 n'est vraie qu'en
  régime B. **Risque principal** : reproche « Hopfield/Gardner avec des étapes en
  plus ». Parade : isoler *une* affirmation neuve et la prouver ou la rendre
  indéniablement utile.

Verdict : **potentiel réel mais ciblé** — viser un workshop (mémoire associative /
attention / neurosymbolique) avec un noyau net, pas une grande conférence
généraliste ni une prétention de performance.

## 5. L'affirmation centrale à défendre (choisir UNE)

Deux candidates, par ordre de préférence.

**A. (recommandée) Pluripotence = aggrégation de comportements similaires via
supports stables chevauchants.**
Formulation *large* (cf. `docs/conception_pluripotence.md`) : « un réseau à poids
bornés dont le **jeu stable** se décompose en **familles à cœur commun** (supports
chevauchants), navigables par quelques variables/nœuds ». Le *routage par contexte
au-delà de la capacité de rappel exact* (banques disjointes) en est le **cas
particulier** déjà mesuré, mais la thèse à défendre est la plus générale : la
pluripotence est une propriété **intrinsèque de la structure des supports
stables**, pas une sélection par nœud de contrôle. Le plus singulier ; demande un
petit théorème / une mesure de chevauchement (voir §7 et `conception_pluripotence.md`).
*Attention : ne pas re-rétrécir A à « un nœud de contrôle → mémoires disjointes ».*

**B. Marge exacte et certifiée comme robustesse apprenable.**
« Pour un nœud à seuil à poids entiers bornés, la marge de stabilité maximale (=
profondeur de bassin) et la capacité par nœud sont exactement calculables et
certifiables par ASP/MaxSAT ; la marge est un paramètre d'apprentissage qui
arbitre capacité contre robustesse sous budget de poids fixe. » Le plus « propre »
côté méthode, très neurosymbolique.

Idéalement, **A comme cœur, B comme outil** qui la rend exacte.

## 6. Positionnement et baselines à produire

Avant toute soumission, ancrer par des comparaisons standard :

- **{−1,+1} + biais θ** : refaire mémoire/pluripotence en codage symétrique pour
  se situer vs la **borne de Gardner (~2n)** et la littérature Hopfield classique.
- **Modern Hopfield / attention (Ramsauer 2020)** : montrer explicitement que la
  version *soft* (continuité sur les états, poids gardés discrets) du régime B
  redonne l'attention softmax → le hard du projet en est la limite discrète.
- **TC⁰ / circuits à seuil** : situer l'expressivité par nœud/profondeur.
- **BNN entraîné + petit MLP continu** : baselines de performance honnêtes (pour
  dire « on n'est pas là pour ça », chiffres à l'appui).
- **Solveur MaxSAT dédié** et **binarisation-après-continu (§4.7)** : baselines de
  la brique d'apprentissage exact.

## 7. Plan d'attaque (priorisé)

1. **Le pari le plus payant — ASP « attracteur contraint » (régime B par solveur).**
   Étendre l'encodage §4.6 de « 1 pas » à « W tel que ces motifs soient les
   points fixes/2-cycles voulus », éventuellement avec contexte (pluripotence
   apprise exactement). C'est ce qu'aucune méthode en ligne ne fait, et ça marie
   ASP + mémoire + pluripotence en un énoncé. *Difficile à qualifier de déjà connu.*
2. **Formaliser l'affirmation A** : énoncer et tenter de prouver, même sous
   hypothèses simplificatrices, « routage-capacité > rappel-capacité » (p. ex. via
   un argument de séparation des sous-espaces induits par le contexte clampé).
3. **Baselines §6** : {−1,+1}/Gardner, soft-Hopfield, BNN/MLP. Indispensable pour
   la crédibilité.
4. **Passage à l'échelle** : repousser le mur ASP (symétries, bornes serrées sur
   θ/κ, clingcon) ; sinon assumer le petit régime et le documenter.
5. **Rédaction** : une note de 6–8 pages autour de A (cœur) + B (outil), avec le
   contraste B vs C comme garde-fou d'honnêteté et les figures déjà produites.

## 8. Ce qui tuerait le projet (à surveiller)

- Rester sur des résultats qui *redérivent* Hopfield/Gardner sans énoncé neuf → le
  §7.1 (attracteur contraint) et §7.2 (théorème de routage) sont les antidotes.
- Glisser vers une prétention de performance → la contredit frontalement (régime C).
- Éparpillement sur les trois régimes → **choisir A, tout y ramener.**

---
*Synthèse : le projet a un vrai noyau original (prompts matériels + apprentissage
exact certifié de la marge) et une histoire honnête. Il vaut une contribution
ciblée si l'on prouve/instancie UNE affirmation nette et qu'on l'ancre proprement
dans la littérature — pas si on cherche la performance ou on multiplie les pistes.*
