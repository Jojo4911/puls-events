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

## Docker (20/03/2026)

Création d'un Dockerfile, d'un .dockerignore, et ensuite adaptation des fichiers d'environnement pour que la lecture des clés soit correcte.

### Résultats

J'ai testé deux choses :
* `/health` depuis la ligne de commande avec une bonne réponse.
* `/ask` depuis le navigateur avec le swagger, j'ai posé une question classique et j'ai enfin eu une réponse attendue avec un code 200.

### Points d'attention

Les points d'attention font écho à des difficultés rencontrées et résolues :
1. Docker Desktop non lancé → le daemon doit tourner
2. Espaces autour du = dans le .env → Docker est strict
3. Guillemets autour des valeurs → python-dotenv les retire, Docker non
4. Commentaires en fin de ligne → python-dotenv les supporte, Docker non

### Logs de rebuild depuis le swagger (Docker)

2026-03-20 13:40:12,501 - INFO - Début de la collecte — département: Drôme, historique: 365 jours
2026-03-20 13:40:13,760 - INFO - Récupérés: 100 / 1068 événements
2026-03-20 13:40:14,148 - INFO - Récupérés: 200 / 1068 événements
2026-03-20 13:40:14,564 - INFO - Récupérés: 300 / 1068 événements
2026-03-20 13:40:14,948 - INFO - Récupérés: 400 / 1068 événements
2026-03-20 13:40:15,338 - INFO - Récupérés: 500 / 1068 événements
2026-03-20 13:40:15,756 - INFO - Récupérés: 600 / 1068 événements
2026-03-20 13:40:16,686 - INFO - Récupérés: 700 / 1068 événements
2026-03-20 13:40:17,063 - INFO - Récupérés: 800 / 1068 événements
2026-03-20 13:40:17,483 - INFO - Récupérés: 900 / 1068 événements
2026-03-20 13:40:17,886 - INFO - Récupérés: 1000 / 1068 événements
2026-03-20 13:40:18,330 - INFO - Récupérés: 1068 / 1068 événements
2026-03-20 13:40:18,330 - INFO - Collecte terminée: 1068 évènements récupérés
2026-03-20 13:40:18,331 - INFO - Nettoyage de 1068 événements...
2026-03-20 13:40:18,426 - INFO - Nettoyage terminé: 1066 événements retenus
2026-03-20 13:40:18,496 - INFO - Données sauvegardées: /app/data/events_drome.json et /app/data/events_drome.csv
2026-03-20 13:40:18,497 - INFO - Chargement des données depuis /app/data/events_drome.csv
2026-03-20 13:40:18,559 - INFO - Conversion : 1066 événements -> 1066 documents LangChain
2026-03-20 13:40:18,653 - INFO - Chunking terminé : 1066 documents -> 1397 chunks (chunk_size=1000, overlap=150).
2026-03-20 13:40:18,654 - INFO -  - 909 documents non découpés (< 1000 car.)
2026-03-20 13:40:18,654 - INFO -  - 157 documents découpés en plusieurs chunks.
2026-03-20 13:40:18,654 - INFO - Embedding provider : Google (gemini-embedding-2-preview, task_type=RETRIEVAL_DOCUMENT)
2026-03-20 13:40:18,670 - INFO - Construction de l'index FAISS : 1397 chunks en 14 batchs (taille=100, délai=65s, durée estimée=15.2 min)...
2026-03-20 13:40:20,203 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:40:21,698 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:40:21,775 - INFO - Batch 1/14 traité (100 chunks, total=100/1397).
2026-03-20 13:40:21,954 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:40:21,955 - WARNING - Rate limit atteint (batch 2, tentative 1/8).Attente de 10s...
2026-03-20 13:40:32,124 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:40:32,125 - WARNING - Rate limit atteint (batch 2, tentative 2/8).Attente de 20s...
2026-03-20 13:40:52,456 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:40:52,457 - WARNING - Rate limit atteint (batch 2, tentative 3/8).Attente de 40s...
2026-03-20 13:41:34,169 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:41:35,719 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:41:35,752 - INFO - Batch 2/14 traité (100 chunks, total=200/1397).
2026-03-20 13:41:35,952 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:41:35,952 - WARNING - Rate limit atteint (batch 3, tentative 1/8).Attente de 10s...
2026-03-20 13:41:46,149 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:41:46,150 - WARNING - Rate limit atteint (batch 3, tentative 2/8).Attente de 20s...
2026-03-20 13:42:06,333 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:42:06,333 - WARNING - Rate limit atteint (batch 3, tentative 3/8).Attente de 40s...
2026-03-20 13:42:48,016 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:42:49,503 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:42:49,544 - INFO - Batch 3/14 traité (100 chunks, total=300/1397).
2026-03-20 13:42:49,720 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:42:49,720 - WARNING - Rate limit atteint (batch 4, tentative 1/8).Attente de 10s...
2026-03-20 13:42:59,919 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:42:59,919 - WARNING - Rate limit atteint (batch 4, tentative 2/8).Attente de 20s...
2026-03-20 13:43:21,646 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:43:23,167 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:43:23,236 - INFO - Batch 4/14 traité (100 chunks, total=400/1397).
2026-03-20 13:43:23,441 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:43:23,441 - WARNING - Rate limit atteint (batch 5, tentative 1/8).Attente de 10s...
2026-03-20 13:43:33,607 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:43:33,608 - WARNING - Rate limit atteint (batch 5, tentative 2/8).Attente de 20s...
2026-03-20 13:43:53,777 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:43:53,779 - WARNING - Rate limit atteint (batch 5, tentative 3/8).Attente de 40s...
2026-03-20 13:44:35,431 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:44:36,915 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:44:37,912 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-20 13:44:37,924 - INFO - Batch 5/14 traité (100 chunks, total=500/1397).
2026-03-20 13:44:38,096 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:44:38,097 - WARNING - Rate limit atteint (batch 6, tentative 1/8).Attente de 10s...
INFO:     Shutting down
INFO:     Waiting for connections to close. (CTRL+C to force quit)
2026-03-20 13:44:48,317 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 429 Too Many Requests"
2026-03-20 13:44:48,321 - WARNING - Rate limit atteint (batch 6, tentative 2/8).Attente de 20s...

### Conclusions
deux points critiques validés dans le conteneur :

Les chemins fonctionnent : /app/data/ est créé et les fichiers y sont écrits correctement
Les permissions sont OK : ton utilisateur user non-root peut écrire partout où il faut

Le pipeline Docker fonctionne de bout en bout, c'est prouvé.