# SBN comme réseaux de neurones prédictifs — synthèse technique (v0, à itérer)

> Document de travail destiné à être fourni à un autre modèle (Fable) pour
> itération. Objectif : recenser **tout ce qu'il faut** pour adapter les Réseaux
> Booléens à Signe (SBN) du projet *PluripotentSBN* à des tâches de **prédiction**
> (classification / régression), en réutilisant au maximum la pile existante
> (énumération des SBF, attracteurs, robustesse/évolutivité, ASP, Monte-Carlo).
> Les points marqués **[À trancher]** sont des choix de conception ouverts.

---

## 0. Le formalisme, tel qu'il est réellement implémenté

À ne pas idéaliser : on part de la définition *exacte* du code (`sbf.py`,
`SBFStatTable.java`), pas d'une version générique.

- **Unité (SBF de dimension k).** `f : {0,1}^k → {0,1}`, avec
  `f(x) = 1  ⟺  Σᵢ wᵢ·xᵢ > 0`.
- **Poids entiers bornés.** `wᵢ ∈ [−wb(k), +wb(k)]` avec
  `wb(1)=1, wb(2)=1, wb(3)=2, wb(4)=3, wb(5)=5, wb(d)=d` pour `d>5`.
- **Réseau (SBN de dimension n).** État `x(t) ∈ {0,1}ⁿ`. Chaque nœud `j` a sa
  fonction `f_j` = SBF de la **colonne j** de la matrice de poids `W`. Les
  colonnes sont indépendantes → l'espace des dynamiques distinctes est
  exactement `(SBF_n)ⁿ`, et **tout** n-uplet de SBF est réalisable.
- **Mise à jour synchrone** (tous les nœuds en parallèle) →
  `x(t+1) = F_W(x(t))`, dynamique **déterministe** sur l'hypercube. Attracteurs =
  points fixes **ou cycles** (d'où `NumAttractors`, `CycleLenMSQ`).

### Trois conséquences directes pour la prédiction

1. **Un nœud de SBN = un neurone de McCulloch–Pitts (1943) / un perceptron à
   seuil.** Le pont n'est pas analogique, il est définitionnel. Un SBN est donc un
   *réseau de neurones récurrent binaire à poids entiers bornés*, dans la lignée
   Little (1974) / Hopfield (1982).
2. **Pas de biais natif ⇒ `f(0,…,0) = 0` pour tout nœud** (somme vide = 0, pas
   `> 0`). L'état nul est **toujours** un point fixe. Aucune fonction ne peut
   « s'allumer » sur l'entrée nulle. Pour toute tâche de prédiction, il faudra un
   **nœud de biais clampé à 1** (entrée constante), sinon on perd la moitié de
   l'expressivité utile. *C'est le premier détail d'ingénierie non négociable.*
3. **Seuil strict `> 0` + états `{0,1}` (codage asymétrique).** Différent du
   codage `{−1,+1}` de Hopfield : passer d'un codage à l'autre déplace
   implicitement le biais. À fixer explicitement avant toute comparaison avec la
   littérature Hopfield/BNN.

### Un atout sous-estimé : l'énumération exacte des SBF

Le projet **énumère déjà** toutes les SBF distinctes par dimension
(32 en d=3, 370 en d=4, 11 292 en d=5). Conséquence pour l'apprentissage :
un **ajustement supervisé par nœud** (« quelle SBF reproduit le mieux le
comportement voulu du nœud j sur les données ? ») se réduit à un **argmin sur un
ensemble fini énuméré** — exact, sans gradient. C'est le levier le plus
spécifique au projet et il faut le mettre au centre (cf. §3 et §4.4).

---

## 1. Ce qui distingue un SBN d'un NN prédictif « classique »

| Axe | NN prédictif standard | SBN (ce projet) | Enjeu |
|---|---|---|---|
| États / activations | réels, continus | binaires `{0,1}` | quantification, perte d'info |
| Activation | différentiable (ReLU, sigmoïde) | seuil (Heaviside) | **pas de gradient direct** |
| Poids | réels | entiers **fortement bornés** | capacité par unité limitée |
| Topologie | *feedforward* (couches) | récurrent (dynamique) | « quand lire la sortie ? » |
| Calcul | une passe avant | **trajectoire vers un attracteur** | convergence non garantie |
| Biais | libre | absent (à ajouter) | cf. §0.2 |

Deux obstacles durs à garder en tête : **(a) non-différentiabilité** du seuil et
**(b) nature dynamique/récurrente** (il faut définir l'instant de lecture). Les
trois régimes ci-dessous répondent différemment à ces deux obstacles.

---

## 2. Trois régimes de prédiction (à explorer séparément)

### Régime A — Classifieur *feedforward* (SBN déplié en MLP de portes à seuil)

On impose une structure en couches : nœuds d'entrée clampés, couches cachées,
nœuds de sortie. Un SBN acyclique déplié = un **perceptron multicouche à
activation seuil** (deep threshold network).

- **Entrée** : clamper les nœuds d'entrée à l'encodage binaire du vecteur
  d'exemple pendant `L` pas (L = profondeur).
- **Sortie** : lire les nœuds de sortie après `L` pas.
- **Poids** : c'est là que ça coince — apprentissage multicouche = gradient, or
  seuil non différentiable ⇒ voir §4.4 (straight-through, recherche, ASP).
- **Capacité** : une seule porte à seuil ne calcule pas XOR (Minsky–Papert 1969) ;
  il **faut** des couches cachées. Théorie de référence : circuits à seuil
  (classe TC⁰). Expressivité réelle mais gouvernée par profondeur × largeur.
- **Fit projet** : moyen. On perd le côté « dynamique » qui fait l'intérêt du
  projet ; on retombe sur un BNN classique.

### Régime B — Mémoire associative / attracteurs (le plus *natif*)

Le mode de prédiction naturel d'un réseau booléen dynamique : **encoder les
cibles comme attracteurs**, présenter une entrée partielle/bruitée comme état
initial, laisser converger → l'attracteur atteint = la prédiction (mémoire
adressable par le contenu, façon Hopfield).

- **Entrée** : état initial `x(0)` = motif d'entrée (éventuellement partiel).
- **Sortie** : attracteur atteint (point fixe ou cycle) → décodé en classe.
- **Poids** : stockage **hebbien** / one-shot des motifs cibles ; pas
  d'entraînement itératif nécessaire pour une première version.
- **Réutilise directement** : le calcul d'attracteurs, de bassins, la
  **robustesse** (⇒ taille de bassin ⇒ tolérance au bruit), la notion de
  **pluripotence/décomposition** (⇒ multi-tâche / plusieurs « mémoires » dans un
  même réseau).
- **Piège technique majeur** : la mise à jour **synchrone** + poids symétriques
  donne des **cycles de période 2** (théorie de Goles–Olivos : les réseaux à
  seuil synchrones convergent vers points fixes **ou** 2-cycles). Deux options
  **[À trancher]** : (i) passer en mise à jour **asynchrone** pour retrouver la
  convergence Hopfield vers points fixes ; (ii) garder le synchrone et
  **traiter les 2-cycles comme des attracteurs-classes** à part entière.
- **Fit projet** : excellent. C'est le régime que je recommanderais en premier
  pour un lien conceptuel fort avec la pluripotence.

### Régime C — *Reservoir computing* (le moins cher à tester)

Garder le SBN comme **réservoir dynamique aléatoire FIXE** (jamais entraîné) et
n'entraîner qu'un **lecteur linéaire** sur les états visités.

- **Entrée** : injectée dans le réservoir (clamp de quelques nœuds, ou biais
  d'entrée sur la somme seuil).
- **État de calcul** : concaténation des états `x(0),…,x(T)` (ou statistiques :
  fréquence d'activation par nœud, attracteur atteint…) → vecteur de features.
- **Sortie** : `y = readout(features)`, avec **readout entraîné** (régression
  logistique / linéaire — là, le gradient marche, on est hors du réservoir).
- **Poids du réservoir** : non entraînés ⇒ **sidestep total** de la
  non-différentiabilité. On **tire** le réservoir avec MCSBN.
- **Réutilise directement** : le **générateur Monte-Carlo** (MCSBN) produit
  gratuitement des réservoirs ; on peut *sélectionner* les réservoirs selon
  robustesse/évolutivité/richesse dynamique (lien avec « edge of chaos » en
  reservoir computing).
- **Fit projet** : très bon rapport effort/résultat. **Prototype recommandé en
  premier jalon** (cf. §6).

---

## 3. Détails transverses — l'encodage entrée/sortie

### 3.1 Encodage des entrées (features → bits)

Les entrées continues doivent être binarisées ; le choix influence fortement la
séparabilité par des portes à seuil.

- **One-hot** (catégoriel) : robuste, coûteux en nœuds.
- **Thermomètre / unaire** (ordinal, continu discrétisé en seuils) :
  `v ↦ 1^k 0^{m−k}` — **préserve l'ordre**, très adapté aux seuils signés,
  recommandé pour les features numériques.
- **Binaire naturel** : compact mais crée des frontières non linéaires
  difficiles pour un seuil ⇒ à éviter comme entrée directe.
- **Nœud de biais** clampé à 1 : **obligatoire** (§0.2).
- **Clamp vs injection** **[À trancher]** : soit on *fige* des nœuds d'entrée
  (ils gardent leur valeur), soit on *ajoute* un terme d'entrée à la somme
  seuil `Σ wᵢxᵢ + bᵢ·uⱼ` (plus proche du reservoir computing).

### 3.2 Encodage des sorties (bits → prédiction)

- **One-hot + argmax** sur `C` nœuds de sortie : classification `C` classes.
- **Identité d'attracteur** (régime B) : chaque classe = un attracteur
  pré-assigné ; prédiction = attracteur atteint (nécessite un dictionnaire
  attracteur→classe).
- **Readout linéaire** (régime C) : sortie continue possible ⇒ régression.
- **Régression** en sortie binaire pure : passer par un code
  thermomètre inverse (décodage) — imprécis, réserver aux petites plages.

### 3.3 Quand lire la sortie ? (le nerf du régime récurrent)

- **Horizon fixe `T`** : lire à `x(T)`. Simple, mais `T` est un hyperparamètre
  **[À trancher]**.
- **À convergence** : lire l'attracteur. Nécessite de **gérer la
  non-convergence / les cycles** (borne d'itérations, politique de repli, ou
  lecture d'une statistique moyenne sur le cycle).
- **Lecture intégrée sur la trajectoire** (régime C) : plus riche, plus robuste.

---

## 4. Détails transverses — l'ajustement des poids

C'est le cœur de la question, et là où le seuil non différentiable + poids
entiers bornés changent tout. Panorama des méthodes, de la plus native à la plus
« importée » :

### 4.1 Hebbien / one-shot (mémoire associative, régime B)

Stockage direct des motifs cibles dans `W` (règle de Hopfield adaptée au codage
`{0,1}` et au seuil strict). **Aucune itération.** Capacité limitée
(≈ 0,14·n motifs pour Hopfield), diaphonie entre motifs. Bon point de départ,
faible performance attendue mais **exact et interprétable**.

### 4.2 Règle du perceptron, **par nœud**

Pour un nœud dont on connaît la sortie désirée sur chaque exemple : la règle du
perceptron converge **si** les données sont linéairement séparables (dans le
codage binaire). Applicable au **readout** (régime C) et à la **couche de
sortie** (régime A). Ne résout pas l'affectation de crédit des couches cachées.

### 4.3 Ajustement **exact par énumération de SBF** (spécifique au projet)

Puisque toutes les SBF de dimension n sont énumérées : pour un nœud dont on a les
paires (entrée du nœud → sortie voulue), **choisir la SBF qui minimise l'erreur**
= argmin sur la table `SBFTable`. Exact, sans gradient, trivial pour la couche de
sortie et le readout. Pour les couches cachées, il faut d'abord *fabriquer* des
cibles cachées (cf. 4.4). **À exploiter en priorité** : c'est ce que le projet
sait déjà faire mieux que quiconque.

### 4.4 *Straight-Through Estimator* / gradients de substitution (BNN)

Approche standard des réseaux binarisés (Courbariaux, Hubara, Bengio 2016) :
entraîner des **poids-ombres continus** par backprop en traitant le seuil comme
l'identité (ou une sigmoïde raide) au *backward*, puis **projeter** sur
`[−wb, wb] ∩ ℤ` au *forward*. Permet le multicouche (régime A). Attention : la
**forte quantification** des poids (wb petit) est plus agressive que les BNN
usuels ⇒ dégradation à surveiller.

### 4.5 Apprentissage **par recherche** (réutilise MCSBN)

Traiter l'apprentissage comme une **optimisation dans l'espace discret** des
poids/SBF : recherche locale, algorithmes évolutionnaires, ou Monte-Carlo, avec
**fitness = précision de prédiction** (+ régularisation par robustesse). Le
générateur MCSBN fournit déjà l'échantillonnage ; il « suffit » d'ajouter la
fonction de fitness et une pression de sélection. **Très cohérent avec l'esprit
du projet** (exploration du paysage).

### 4.6 Apprentissage **par résolution** ASP / MaxSAT (l'angle original)

Encoder l'apprentissage comme un **problème de satisfaction/optimisation** :
« ∃ W tel que ∀ (x,y) du jeu d'entraînement, `dynamique(W, x)` atteigne `y` (ou
maximiser le nombre de contraintes satisfaites) ». Résolu par **clingo/clingcon**
(déjà dans la pile !) ou un solveur MaxSAT.
- **Exact** (poids entiers = domaine fini, naturel pour l'ASP).
- **Nouveau** : l'apprentissage-par-solveur de réseaux à seuil est peu exploré,
  et le projet a déjà les encodages `.lp` et le savoir-faire ASP.
- **Limite** : passage à l'échelle (NP-difficile) ⇒ petits réseaux / petits jeux.
- **C'est l'angle le plus publiable** : il connecte apprentissage, pluripotence
  et model counting.

### 4.7 Relaxation continue → binarisation

Entraîner un réseau continu classique, puis **binariser** poids et activations
en projetant sur le format SBN. Simple, mais l'écart continu→binaire dégrade, et
on perd le lien conceptuel. Utile surtout comme *baseline* de comparaison.

---

## 5. Ce que les grandeurs du projet deviennent en langage « apprentissage »

Réinterprétation qui rend le lien fécond (et fournit des hypothèses testables) :

- **Robustesse** (taille de bassin / stabilité aux mutations d'état) ≈ **marge /
  tolérance au bruit d'entrée** ≈ proxy de **généralisation**.
- **Évolutivité** (nb de dynamiques distinctes atteintes par mutation d'un poids)
  ≈ **entraînabilité / navigabilité** du paysage d'apprentissage.
- **Pluripotence / décomposition** (un réseau « simule » plusieurs sous-réseaux)
  ≈ **modularité / apprentissage multi-tâche** : un même SBN encodant plusieurs
  fonctions selon un contexte figé (les nœuds de contrôle = des « prompts »
  matériels). C'est le pont conceptuel le plus riche et le plus spécifique.
- **Métagraphe** (dynamiques voisines par mutation) ≈ **paysage de perte
  discret** : la trajectoire d'apprentissage 4.5 s'y lit littéralement.

Hypothèse directrice proposée : *les réseaux les plus robustes/pluripotents
généralisent mieux et sont plus faciles à apprendre.* Testable avec l'outillage
existant.

---

## 6. Prototype minimal proposé (premier jalon pour Fable)

Ordre de montée en complexité, du plus sûr au plus ambitieux :

1. **Reservoir (régime C) sur un jeu binaire jouet** (p. ex. parité, ou MNIST
   binarisé downsamplé) : tirer K réservoirs SBN via MCSBN (d = 5–6), collecter
   les états sur T pas, entraîner un readout logistique, mesurer l'accuracy vs
   robustesse/évolutivité du réservoir. *Coût faible, valide le pont, réutilise
   tout MCSBN.*
2. **Mémoire associative (régime B)** : stockage hebbien de C motifs-classes,
   test de rappel sous bruit, en tranchant synchrone-vs-asynchrone. Relier
   capacité de rappel ↔ taille de bassin (robustesse déjà calculée).
3. **Apprentissage ASP (régime 4.6)** sur d = 3–4 : encoder « trouver W tel que
   ces paires (x→y) tiennent » en `.lp`, mesurer jusqu'où ça passe à l'échelle,
   comparer au straight-through (4.4).

Chaque jalon produit une figure comparable au reste du rapport
(scatter accuracy × robustesse, métagraphe coloré par accuracy…).

---

## 7. Questions ouvertes à trancher avec Fable

1. Synchrone (2-cycles à gérer) **ou** asynchrone (convergence type Hopfield) ?
2. Clamp d'entrée **ou** injection additive dans la somme seuil ?
3. Lecture à horizon fixe `T` **ou** à convergence ?
4. Nœud de biais unique **ou** biais par nœud (change la classe de fonctions) ?
5. Objectif prioritaire : **performance** (alors : régime C + STE) **ou**
   **contribution conceptuelle** (alors : régime B/pluripotence + apprentissage
   ASP) ? *Recommandation : le second, c'est là que le projet est singulier.*
6. Comparer à quel *baseline* (BNN entraîné classiquement, petit MLP continu) ?

---

## 8. Ancrages bibliographiques (pour situer et itérer)

- McCulloch & Pitts (1943) — neurone à seuil ; Rosenblatt (1958) — perceptron ;
  Minsky & Papert (1969) — limite XOR d'une porte seuil.
- Little (1974), Hopfield (1982) — réseaux récurrents binaires, mémoire
  associative ; Goles & Olivos — convergence des réseaux à seuil **synchrones**
  (points fixes ou 2-cycles).
- Théorie des **circuits à seuil** (classe TC⁰) — expressivité des réseaux de
  portes à seuil bornées.
- Courbariaux, Hubara, Bengio (2016) — *Binarized Neural Networks* /
  straight-through estimator.
- **Reservoir computing** — echo state networks (Jaeger), liquid state machines
  (Maass) ; « edge of chaos » et richesse dynamique.
- Wagner (2007/2008) — robustesse & évolutivité (déjà dans la biblio du projet),
  à relire cette fois sous l'angle généralisation.

---

### TL;DR pour Fable

Un nœud de SBN **est** un neurone à seuil sans biais ⇒ la prédiction est
possible, sur trois régimes distincts : **A** (classifieur feedforward, obstacle
= gradient), **B** (mémoire à attracteurs, le plus natif, obstacle = 2-cycles
synchrones), **C** (reservoir, le moins cher, obstacle = aucun côté récurrent).
Les deux leviers spécifiques et sous-exploités du projet sont **l'ajustement
exact par énumération de SBF** et **l'apprentissage-par-résolution ASP/MaxSAT**.
La valeur n'est pas la performance brute mais le recyclage de la machinerie
pluripotence/robustesse en théorie de l'apprentissage. Premier jalon conseillé :
un **reservoir** (régime C) sur jeu binaire jouet via MCSBN.
