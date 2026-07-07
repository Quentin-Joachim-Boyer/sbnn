# Conception de la pluripotence — un concept souple (à garder au centre)

> Rappel de cadrage conceptuel, à ne pas oublier : la **pluripotence** est LE
> cœur du projet — l'**aggrégation, dans un même réseau, de comportements
> *similaires***. C'est un concept *souple*, qui se lit de plusieurs manières.
> Ne pas le réduire à sa forme la plus étroite (un nœud de contrôle sélectionnant
> des mémoires disjointes) : ce n'est qu'une instanciation parmi d'autres.

## Trois lentilles, de la plus étroite à la plus fidèle

**1. Un nœud de contrôle (le plus étroit — déjà fait).**
Un nœud de contexte clampé sélectionne un comportement. Sélection *extrinsèque*
parmi des comportements *disjoints* (banques de mémoire séparées). Utile comme
preuve de concept (`experiments/pluripotence_regimeB.py`), mais réducteur : la
pluripotence n'est pas la sélection, c'est la co-existence structurée.

**2. Plusieurs nœuds de contrôle — indexation vs composition.**
- *Indexation* : k nœuds → 2^k contextes, mais toujours des banques disjointes.
  Peu de gain conceptuel.
- *Composition (riche)* : les nœuds de contrôle sont des **boutons indépendants
  dont les effets se composent**. Le réseau réalise alors une **famille
  structurée / factorisée** de comportements qui **partagent de la structure**
  (contrôle démêlé, composition de traits). C'est déjà de l'aggrégation de
  comportements *similaires*, pas une simple table de correspondance.

**3. Supports stables plutôt que disjoints (le recentrage clé).**
Définir la pluripotence **intrinsèquement**, par la géométrie du **jeu stable** du
réseau — pas par un sélecteur externe. On regarde les **supports** des états
stables (attracteurs : points fixes, 2-cycles) : quels nœuds sont actifs/figés, et
**comment ces supports se chevauchent**.
- Supports **disjoints** → vraies mémoires séparées (la lentille 1).
- Supports **chevauchants / emboîtés** → **aggrégation de comportements
  *similaires*** : un **cœur stable commun** partagé par toute une famille, avec
  variation en **périphérie**. *C'est ça, la pluripotence au sens fort.*

Dans cette lentille, le clamp d'un nœud n'est qu'**un moyen de se déplacer dans le
support partagé** (choisir un membre de la famille) ; le phénomène central est la
**structure de chevauchement des supports stables**, pas la porte de contrôle.

## Ce que ça change pour l'affirmation centrale (note de cadrage §5)

Déplacer le cœur : de
« *banques disjointes sélectionnées par contexte + routage au-delà de la
capacité* »
vers
« ***aggrégation de comportements similaires via des supports stables
chevauchants*** — un réseau à poids bornés dont le jeu stable se décompose en
familles à cœur commun, navigables par quelques nœuds/variables ».
Plus fidèle à *PluripotentSBN*, plus original, et ça englobe le routage comme cas
particulier (familles à supports disjoints).

## Comment l'opérationnaliser (avec l'outillage existant)

Le cœur `network.py` (attracteurs, bassins) suffit à démarrer :

1. **Répertoire stable** : énumérer/échantillonner les attracteurs d'un réseau
   (déjà : `run_sync`, `attractor_sync`).
2. **Support** de chaque état stable : nœuds actifs (ou, pour une famille, le
   **cœur figé** = nœuds constants sur toute la famille) vs **périphérie** variable.
3. **Géométrie de chevauchement** : treillis/poset des supports ; mesurer le
   partage (cœur commun) vs la disjonction. Une **mesure de pluripotence** =
   degré de décomposition en familles à support partagé (vs mémoires disjointes).
4. **Lien au contrôle** : identifier quels nœuds, clampés, déplacent *dans* une
   famille (périphérie) vs *entre* familles (cœur) — le contrôle devient une
   *lecture* de la structure des supports, non sa définition.
5. **Apprentissage** : concevoir W (perceptron à marge / ASP §4.6) pour obtenir un
   **cœur stable partagé + périphérie variable** prescrits — la version « supports
   chevauchants » de l'attracteur contraint (`asp_attractor.py`).

## Garde-fou

Chaque fois qu'on parle de pluripotence, se demander : *est-ce que je la réduis à
une sélection par nœud de contrôle ?* Si oui, revenir aux **supports stables** et
à leur chevauchement — c'est là qu'est le concept, et probablement la contribution
la plus originale du projet.
