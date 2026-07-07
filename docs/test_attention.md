# Comment tester l'efficacité du mécanisme d'attention hard-codé

Le réseau ne « fait » pas de l'attention par une couche softmax entraînée : elle
est **hard-codée dans la dynamique** (poids + récurrence → adressage par contenu =
argmax d'overlap, régime B). Tester son efficacité demande donc (1) le bon
**cadre de tâche**, (2) les bonnes **métriques**, (3) les bonnes **baselines**.
Implémenté dans `experiments/attention_efficacite.py`,
figure `figures/attention_efficacite.png`.

## 1. Le cadre : une tâche CLÉ → VALEUR (pas de l'auto-association)

L'auto-association (rappel d'un motif bruité vers lui-même) qu'on avait mesurée au
jalon 2 ne teste pas *l'attention* : l'attention **récupère une VALEUR associée à
la meilleure CLÉ**. On encode donc l'état comme `[clé | valeur]`, on stocke des
paires (clé, valeur) comme attracteurs, on présente une **requête** = clé bruitée
+ valeur à zéro, on laisse la dynamique relaxer, et on **lit la partie valeur**.
C'est exactement l'opération d'attention d'un transformer (query attend keys,
retourne value), mais réalisée par la récurrence.

## 2. La métrique centrale : l'écart à l'attention IDÉALE

Mesurer l'accuracy de récupération (valeur lue == bonne valeur) **ne suffit pas** :
un score bas peut venir soit d'une requête ambiguë (tâche dure), soit d'un
mécanisme faible. Il faut **séparer les deux** en comparant à l'**attention
idéale** — l'argmax d'overlap sur les clés, `argmax_m ⟨requête, clé_m⟩`, qui est
le plafond du *hard* :

- **efficacité du mécanisme = écart (idéal − réseau)**. Si l'idéal réussit et pas
  le réseau, c'est le mécanisme (états parasites), pas la tâche.
- baselines complètes : **idéal argmax** (plafond), **hasard 1/P** (plancher), et
  — pour le lien *soft* — l'attention softmax `Σ softmax(β⟨q,k_m⟩) v_m` dont
  l'argmax est la limite β→∞ (le réseau *est* censé approcher ce hard).

## 3. La batterie (les axes à balayer)

- **Capacité** : accuracy vs **nombre de paires** stockées P (crosstalk).
- **Robustesse** : accuracy vs **bruit de requête** (le « champ récepteur »).
- **Sélectivité** : accuracy vs **similarité des clés** (interférence quand deux
  clés sont proches — à ajouter, cf. le résultat pluripotence sur cibles proches).
- **Latence** : **nb de pas** de convergence (efficacité en temps).
- **Câblage** : hebbien vs **appris à marge** vs idéal — l'attention n'est pas une
  propriété fixe, elle dépend de *comment* on pose W.

## 4. Ce qu'on trouve (mesuré, n_clé=n_val=16)

- **Le mécanisme hebbien est un mauvais attenteur** : dès P=4 paires (bruit 3/16),
  récupération ≈ 0.24 alors que l'idéal argmax est à ≈ 0.97 — l'écart est presque
  tout le signal. Le goulot n'est PAS la difficulté de la requête (l'idéal la
  résout), c'est la **dynamique hebbienne** (attracteurs parasites).
- **L'apprentissage à marge en fait un bon attenteur** : à P=6, 0.24 → **0.88**
  (plafond idéal 0.97). Poser W par marge (bassins profonds) rapproche fortement la
  récurrence de l'argmax idéal — l'efficacité de l'attention est un **choix de
  câblage**, pas une fatalité.
- **Robustesse** : les deux réseaux dégradent avec le bruit de requête ; l'idéal
  tient jusqu'à ~4/16 bits puis chute (au-delà, la requête ne désigne plus la clé —
  limite intrinsèque, pas du mécanisme).
- **Latence** : convergence en ~2–3.5 pas asynchrones, l'appris légèrement plus
  rapide. L'attention hard est *peu coûteuse en temps*.

## 5. À retenir (méthodo)

Pour juger une attention hard-codée : **tâche clé→valeur**, **accuracy vs l'idéal
argmax** (l'écart = efficacité du mécanisme), balayage **capacité / bruit /
sélectivité / latence**, et comparaison des **câblages** (hebbien vs appris) plus
le pont **soft** (softmax β→∞). Conclusion de fond : l'attention native du projet
est réelle mais **seulement aussi bonne que les bassins** — donc le levier
d'efficacité est l'apprentissage à marge / ASP (bassins profonds et
non-interférents), et la sélectivité renvoie directement au chevauchement des
supports stables (`conception_pluripotence.md`).
