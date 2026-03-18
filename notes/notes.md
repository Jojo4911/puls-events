# Notes

## Tests de pertinence du rag_system (16/03/2026)

commande lancée : `uv run python -m src.rag_system`

### 1er essai prompt initial et k = 5

#### Prompt initial

RAG_PROMPT_TEMPLATE = """\
Tu es un assistant spécialisé dans les événements culturels de la Drôme (en France).
Tu aides les utilisateurs à trouver des événements (concerts, expositions, spectacles, festivals, ateliers, etc.) en te basant UNIQUEMENT sur les documents fournis ci-dessous.

Règles :
- Réponds en français.
- Base ta réponse exclusivement sur les documents fournis.
- Si la réponse n'est pas dans les documents, dis-le clairement.
- N'invente jamais d'événements qui n'apparaît pas dans les documents.
- Cite le titre, le lieu et les dates des événements que tu mentionnes.

--- Documents ---
{context}
--- Fin des documents ---

Question : {question}
"""

#### Conclusion

Deux pistes d’amélioration :
1. Itération du prompt (en phase d'évaluation). Injection de la date du jour dans le prompt du LLM.
2. On pourrait demander au LLM de suggérer des événements proches quand rien ne correspond exactement aux critères.

### 2ème essai prompt amélioré, ajout de la date du jour et k = 10

#### Prompt amélioré

RAG_PROMPT_TEMPLATE = """\
Tu es un assistant spécialisé dans les événements culturels de la Drôme (en France).
Tu aides les utilisateurs à trouver des événements (concerts, expositions, spectacles, festivals, ateliers, etc.) en te basant UNIQUEMENT sur les documents fournis ci-dessous.

Règles :
- Réponds en français.
- Base ta réponse exclusivement sur les documents fournis.
- Si la réponse n'est pas dans les documents, dis-le clairement, mais propose des alternatives proches (lieu ou thème).
- N'invente jamais d'événements qui n'apparaît pas dans les documents.
- Cite le titre, le lieu et les dates des événements que tu mentionnes.
- Prends en compte la date d'aujourd'hui : {today_date}. Si l'utilisateur y fait référence (par exemple "Cette semaine", ou "ce week-end"), privilégie des évènements actuels ou futurs à cette date.
- Signale si un évènement est passé par rapport à {today_date}.

--- Documents ---
{context}
--- Fin des documents ---

Question : {question}
"""

#### Conclusion

Les résultats sont nettement meilleurs qu'avant les modifications :
Question 1 — "Concert jazz Valence"
Avant, le LLM disait juste "non" et s'arrêtait là. Maintenant il propose une alternative (le concert à La Bégude-de-Mazenc avec du jazz) et signale que c'est passé. C'est exactement le comportement qu'on voulait.
Question 2 — "Spectacles enfants Montélimar"
C'est la réponse la plus impressionnante. Le LLM fait un vrai travail de synthèse : il dit qu'il n'y a rien à Montélimar précisément, propose des alternatives sur place (le château), puis identifie "Poucette" comme spectacle enfants dans d'autres villes — en ne citant que les dates futures. C'est exactement le comportement d'un assistant utile.
Question 3 — "Ce week-end"
Le LLM comprend qu'on est le 16 mars 2026, identifie correctement le week-end prochain, dit honnêtement qu'il n'y a rien, et propose des événements futurs classés par mois. La date injectée fonctionne parfaitement.
Le passage de k=5 à k=10 combiné au nouveau prompt produit des réponses beaucoup plus riches. Les trois améliorations (date, alternatives, k augmenté) se renforcent mutuellement.

## Écriture des 20 paires question/ground - test_dataset.json (17/03/2026)

J’ai dû orienter mes questions pour que la date ne soit pas bloquante car une majorité des événements est passée.

Le vectorstore actuel créé avec FAISS ne supporte pas le filtrage des dates (contrairement à Chroma ou Qdrant)

Ground_truth très détaillés, à voir si ce n'est pas bloquant au moment de l'implémentation RAGAS.

## Évaluation manuelle (18/03/2026)

Après avoir créé une évaluation manuelle, j'ai les résultats suivants :

### Résultats

=== Métriques ===
Catégorie croisé : Il y a un score de 4.0 sur 4 questions.
Soit un score global de 100.0 %.
Catégorie lieu : Il y a un score de 3.0 sur 3 questions.
Soit un score global de 100.0 %.
Catégorie date : Il y a un score de 1.5 sur 4 questions.
Soit un score global de 37.5 %.
Catégorie type : Il y a un score de 3.5 sur 4 questions.
Soit un score global de 87.5 %.
Catégorie hors_périmètre : Il y a un score de 5.0 sur 5 questions.
Soit un score global de 100.0 %.

### Analyse

Déjà on se rend compte que notre système RAG est très bon de manière globale.

Quand on se penche sur les catégories où il est moins bon, on a :
* **La catégorie date** : Effectivement, puisqu'il s'agit d'un outil FAISS pour réaliser les embeddings, nous avons pas de tri sur les dates en tant que tel. C'est une recherche uniquement sur le sens. C'est d'ailleurs pourquoi à la question sur la fête de la musique, le modèle ne s'est pas basé sur la date réelle de la fête de la musique en France, mais sur le mot fête de la musique qu'il a pu retrouver correctement. Quand j'ai posé des questions sur une date en particulier, le modèle a échoué à chaque fois, complètement ou partiellement.
* **La catégorie type** : De manière générale, la catégorie type a été plutôt performante. La seule fois où j'ai évalué une réponse comme partielle, c'était qu'il me manquait un des trois événements que j'avais dans ma réponse vraie (ground truth). Cependant, cet événement pouvait être sujet à interprétation. En effet, la question relevait d'événements reliés aux voitures, et l'événement attendu était la visite d'une casse de voiture qui peut, selon les passionnés, n'être pas forcément intéressant.

## Évaluation automatique avec Ragas (18/03/2026)

Après avoir créé une évaluation automatique avec Ragas, j'ai les résultats suivants :

### Résultats

#### Résultats premier run

--- Scores Moyens (sur tout le dataset) ---
faithfulness         0.664980
answer_relevancy     0.768140
context_precision         NaN
context_recall       0.625263

#### Résultats second run

--- Scores Moyens (sur tout le dataset) ---
faithfulness 0.624249
answer_relevancy 0.807217
context_precision 1.000000
context_recall 0.622571

### Analyse
Pour plusieurs questions, j'ai des métriques à la valeur NaN. La raison principale de tout ça, c'est que je dois avoir des timeouts dus à mon tiers payant de Google. Et je n'ai pas pu avoir toutes les requêtes de Ragas.
Les moyennes sont calculées seulement sur des passes qui ont réussi, mais elles sont incomplètes.
Une amélioration pourrait être de gérer ce problème de requêtes par minute.
Le fichier n’a pas pu être enregistré car : AttributeError: 'NoneType' object has no attribute 'from_iterable'