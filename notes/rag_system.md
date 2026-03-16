# Tests de pertinence du rag_system

commande lancée : `uv run python -m src.rag_system`

## 1er essai prompt initial et k = 5

### Prompt initial

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

### Conclusion

Deux pistes d’amélioration :
1. Itération du prompt (en phase d'évaluation). Injection de la date du jour dans le prompt du LLM.
2. On pourrait demander au LLM de suggérer des événements proches quand rien ne correspond exactement aux critères.

### Résultats
(puls-events) PS C:\Users\jonat\Desktop\Cours en ligne\Formation AI Engineer\Projets\07\puls-events> uv run python -m src.rag_system
2026-03-16 11:48:44,971 - INFO - Initialisation du RAGSystem...
2026-03-16 11:48:45,961 - INFO - Embedding provider : Google (gemini-embedding-2-preview, task_type=RETRIEVAL_QUERY)
2026-03-16 11:48:46,548 - INFO - Chargement de l'index FAISS depuis C:\Users\jonat\Desktop\Cours en ligne\Formation AI Engineer\Projets\07\puls-events\faiss_index
2026-03-16 11:48:46,553 - INFO - Loading faiss with AVX2 support.
2026-03-16 11:48:46,627 - INFO - Successfully loaded faiss with AVX2 support.
2026-03-16 11:48:46,687 - INFO - Index FAISS chargé : 1403 vecteurs
2026-03-16 11:48:46,688 - INFO - Retriever configuré (k=5).
2026-03-16 11:48:46,688 - INFO - LLM provider : Google (gemini-3.1-flash-lite-preview), temperature=0.2
2026-03-16 11:48:47,233 - INFO - RAGSystem prêt.

============================================================
QUESTION : Y a-t-il un concert de jazz à Valence ?
============================================================
2026-03-16 11:48:47,233 - INFO - Question : 'Y a-t-il un concert de jazz à Valence ?'
2026-03-16 11:48:47,803 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-16 11:48:47,810 - INFO - Documents récupérés : 5
2026-03-16 11:48:47,817 - INFO - AFC is enabled with max remote calls: 10.
2026-03-16 11:48:48,589 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent "HTTP/1.1 200 OK"
2026-03-16 11:48:48,595 - INFO - Réponse générée (86 caractères, 5 sources).

RÉPONSE :
 D'après les documents fournis, il n'y a aucune mention d'un concert de jazz à Valence.

SOURCES :
  - Forró à Valence | Bourg-lès-Valence | 18 septembre 2024 - 25 juin 2025, les mercredis
  - Soirée cabaret | Valence | Samedi 17 mai, 19h00
  - Visite guidée de l'Hôtel de ville de Valence | Valence | Dimanche 21 septembre, 10h00, 11h00
  - Forró à Bourg-les-Valence | Bourg-lès-Valence | 10 septembre 2025 - 24 juin 2026, les mercredis
  - Forró à Bourg-les-Valence | Bourg-lès-Valence | 10 septembre 2025 - 24 juin 2026, les mercredis

============================================================
QUESTION : Quels spectacles pour enfants à Montélimar ?
============================================================
2026-03-16 11:48:48,598 - INFO - Question : 'Quels spectacles pour enfants à Montélimar ?'
2026-03-16 11:48:49,040 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-16 11:48:49,047 - INFO - Documents récupérés : 5
2026-03-16 11:48:49,050 - INFO - AFC is enabled with max remote calls: 10.
2026-03-16 11:48:50,033 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent "HTTP/1.1 200 OK"
2026-03-16 11:48:50,041 - INFO - Réponse générée (92 caractères, 5 sources).

RÉPONSE :
 D'après les documents fournis, il n'y a aucun spectacle pour enfants mentionné à Montélimar.

SOURCES :
  - Poucette - Théâtre de la Courte Echelle - Romans-sur-Isère (26) | Romans-sur-Isère | 17 - 19 mai 2026
  - Poucette - Festival La Fête des Enfants - Sainte Croix (26) | Sainte-Croix | Samedi 21 juin, 16h00
  - Poucette - Espace Culturel Liberté - St Marcel lès Valence (26) | Saint-Marcel-lès-Valence | Vendredi 7 novembre, 09h30, 14h00, 19h00
  - Poucette - Médiathèque - Donzère (26) | Donzère | Samedi 4 octobre, 16h00
  - Poucette - Train Théâtre / Hors les Murs - Loriol-sur-Drôme (26) | Loriol-sur-Drôme | Jeudi 30 avril 2026, 15h00

============================================================
QUESTION : Que faire ce week-end dans la Drôme ?
============================================================
2026-03-16 11:48:50,047 - INFO - Question : 'Que faire ce week-end dans la Drôme ?'
2026-03-16 11:48:50,466 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-16 11:48:50,469 - INFO - Documents récupérés : 5
2026-03-16 11:48:50,470 - INFO - AFC is enabled with max remote calls: 10.
2026-03-16 11:48:52,023 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent "HTTP/1.1 200 OK"
2026-03-16 11:48:52,027 - INFO - Réponse générée (981 caractères, 5 sources).

RÉPONSE :
 Pour répondre précisément à votre question, j'ai besoin de connaître la date exacte du week-end auquel vous faites référence.

Cependant, voici les événements disponibles dans les documents fournis, classés par date :

*   **Du 6 au 9 juin :**
    *   **Visite enrichie d'une exploitation agricole** à la Distillerie des 4 Vallées (Chamaloc).
    *   **Visite gratuite de la distillerie des 4 Vallées** à la Distillerie des 4 Vallées (Chamaloc).

*   **Le vendredi 20 juin :**
    *   **Cédille Tour 2025 : les tiers-lieux de Nyons** à Asoft (Nyons), à 09h00.

*   **Les 20 et 21 septembre :**
    *   **Visite guidée du village perché de Mirmande** à l'Église Sainte Foy (Mirmande).

*   **Le dimanche 21 septembre :**
    *   **Visite de la préfecture du conseil départemental de la Drôme** à la Préfecture de la Drôme (Valence), à 10h30, 11h30, 14h00 et 15h00.

Si vous cherchez des événements pour une autre période, ces informations ne figurent pas dans les documents fournis.

SOURCES :
  - Visite enrichie d'une exploitation agricole | Chamaloc | 6 - 9 juin
  - Visite gratuite de la distillerie des 4 Vallées | Chamaloc | 6 - 9 juin
  - Cédille Tour 2025 : les tiers-lieux de Nyons | Nyons | Vendredi 20 juin, 09h00
  - Visite guidée du village perché de Mirmande | Mirmande | 20 et 21 septembre
  - Visite de la préfecture du conseil départemental de la Drôme | Valence | Dimanche 21 septembre, 10h30, 11h30, 14h00, 15h00

## 2ème essai prompt amélioré, ajout de la date du jour et k = 10

### Prompt amélioré

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

### Conclusion

Les résultats sont nettement meilleurs qu'avant les modifications :
Question 1 — "Concert jazz Valence"
Avant, le LLM disait juste "non" et s'arrêtait là. Maintenant il propose une alternative (le concert à La Bégude-de-Mazenc avec du jazz) et signale que c'est passé. C'est exactement le comportement qu'on voulait.
Question 2 — "Spectacles enfants Montélimar"
C'est la réponse la plus impressionnante. Le LLM fait un vrai travail de synthèse : il dit qu'il n'y a rien à Montélimar précisément, propose des alternatives sur place (le château), puis identifie "Poucette" comme spectacle enfants dans d'autres villes — en ne citant que les dates futures. C'est exactement le comportement d'un assistant utile.
Question 3 — "Ce week-end"
Le LLM comprend qu'on est le 16 mars 2026, identifie correctement le week-end prochain, dit honnêtement qu'il n'y a rien, et propose des événements futurs classés par mois. La date injectée fonctionne parfaitement.
Le passage de k=5 à k=10 combiné au nouveau prompt produit des réponses beaucoup plus riches. Les trois améliorations (date, alternatives, k augmenté) se renforcent mutuellement.

### Résultats

(puls-events) PS C:\Users\jonat\Desktop\Cours en ligne\Formation AI Engineer\Projets\07\puls-events> uv run python -m src.rag_system
2026-03-16 21:01:10,553 - INFO - Initialisation du RAGSystem...
2026-03-16 21:01:12,729 - INFO - Embedding provider : Google (gemini-embedding-2-preview, task_type=RETRIEVAL_QUERY)
2026-03-16 21:01:13,332 - INFO - Chargement de l'index FAISS depuis C:\Users\jonat\Desktop\Cours en ligne\Formation AI Engineer\Projets\07\puls-events\faiss_index
2026-03-16 21:01:13,338 - INFO - Loading faiss with AVX2 support.
2026-03-16 21:01:14,006 - INFO - Successfully loaded faiss with AVX2 support.
2026-03-16 21:01:14,066 - INFO - Index FAISS chargé : 1403 vecteurs
2026-03-16 21:01:14,067 - INFO - Retriever configuré (k=10).
2026-03-16 21:01:14,068 - INFO - LLM provider : Google (gemini-3.1-flash-lite-preview), temperature=0.2
2026-03-16 21:01:14,677 - INFO - RAGSystem prêt.

============================================================
QUESTION : Y a-t-il un concert de jazz à Valence ?
============================================================
2026-03-16 21:01:14,678 - INFO - Question : 'Y a-t-il un concert de jazz à Valence ?'
2026-03-16 21:01:15,285 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-16 21:01:15,312 - INFO - Documents récupérés : 10
2026-03-16 21:01:15,322 - INFO - AFC is enabled with max remote calls: 10.
2026-03-16 21:01:16,616 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent "HTTP/1.1 200 OK"
2026-03-16 21:01:16,631 - INFO - Réponse générée (557 caractères, 10 sources).

RÉPONSE :
 D'après les documents fournis, il n'y a pas de concert de jazz prévu à Valence.

Cependant, un concert de jazz est mentionné dans le cadre de l'événement **"Découverte musicale et historique de Châteauneuf-de-Mazenc"**, qui se déroule au **Foyer International d'Études Françaises à La Bégude-de-Mazenc** le **dimanche 21 septembre**. Cet événement propose une programmation musicale variée incluant du jazz (groupe "Rétroactifs") après les concerts de l'après-midi.

Notez que cet événement est passé par rapport à la date d'aujourd'hui, lundi 16 mars 2026.

SOURCES :
  - Forró à Valence | Bourg-lès-Valence | 18 septembre 2024 - 25 juin 2025, les mercredis
  - Soirée cabaret | Valence | Samedi 17 mai, 19h00
  - Visite guidée de l'Hôtel de ville de Valence | Valence | Dimanche 21 septembre, 10h00, 11h00
  - Forró à Bourg-les-Valence | Bourg-lès-Valence | 10 septembre 2025 - 24 juin 2026, les mercredis
  - Forró à Bourg-les-Valence | Bourg-lès-Valence | 10 septembre 2025 - 24 juin 2026, les mercredis
  - Visite guidée l'architecture du XIXe siècle à Valence | Valence | Samedi 20 septembre, 14h00, 16h30
  - L'architecture Art déco à Valence | Valence | 16 et 17 octobre
  - Découverte musicale et historique de Châteauneuf-de-Mazenc | La Bégude-de-Mazenc | Dimanche 21 septembre, 15h00
  - Concert Ëda Diaz | Die | Lundi 21 juillet, 18h30
  - Visite guidée de la cathédrale Saint-Apollinaire | Valence | Dimanche 21 septembre, 14h30

============================================================
QUESTION : Quels spectacles pour enfants à Montélimar ?
============================================================
2026-03-16 21:01:16,643 - INFO - Question : 'Quels spectacles pour enfants à Montélimar ?'
2026-03-16 21:01:16,923 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-16 21:01:16,931 - INFO - Documents récupérés : 10
2026-03-16 21:01:16,937 - INFO - AFC is enabled with max remote calls: 10.
2026-03-16 21:01:19,371 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent "HTTP/1.1 200 OK"
2026-03-16 21:01:19,372 - INFO - Réponse générée (1237 caractères, 10 sources).

RÉPONSE :
 D'après les documents fournis, il n'y a pas de spectacles pour enfants spécifiquement programmés à Montélimar.

Cependant, voici des alternatives culturelles et familiales situées à Montélimar, bien que les dates indiquées soient passées par rapport à aujourd'hui (lundi 16 mars 2026) :

*   **Découverte d'un lieu d'exception : le palais médiéval de Montélimar** au Château de Montélimar (20 et 21 septembre 2025) : Un parcours familial interactif avec des maquettes, des ambiances sonores et des jeux.
*   **Zoom sur une architecture d'exception au château de Montélimar** au Château de Montélimar (20 et 21 septembre 2025) : Une visite guidée de la façade du château accompagnée d'un parcours familial interactif.
*   **Exposition Amazônia de Dalva Duarte** au Musée d'art contemporain de Montélimar (20 et 21 septembre 2025).

Si vous recherchez spécifiquement le spectacle **"Poucette"** (qui est un voyage poétique adapté au jeune public), celui-ci est programmé prochainement dans d'autres communes de la Drôme :
*   **Loriol-sur-Drôme (Salle des fêtes) :** jeudi 30 avril 2026.
*   **Romans-sur-Isère (Théâtre de la Courte Echelle) :** du 17 au 19 mai 2026.
*   **Sainte-Croix (Monastère de Sainte Croix) :** samedi 21 juin 2026.

SOURCES :
  - Poucette - Théâtre de la Courte Echelle - Romans-sur-Isère (26) | Romans-sur-Isère | 17 - 19 mai 2026
  - Poucette - Festival La Fête des Enfants - Sainte Croix (26) | Sainte-Croix | Samedi 21 juin, 16h00
  - Poucette - Espace Culturel Liberté - St Marcel lès Valence (26) | Saint-Marcel-lès-Valence | Vendredi 7 novembre, 09h30, 14h00, 19h00
  - Poucette - Médiathèque - Donzère (26) | Donzère | Samedi 4 octobre, 16h00
  - Poucette - Train Théâtre / Hors les Murs - Loriol-sur-Drôme (26) | Loriol-sur-Drôme | Jeudi 30 avril 2026, 15h00
  - Découverte d'un lieu d'exception : le palais médiéval de Montélimar | Montélimar | 20 et 21 septembre 2025
  - Exposition Amazônia de Dalva Duarte | Montélimar | 20 et 21 septembre
  - Visite enrichie d'une exploitation agricole | Chamaloc | 6 - 9 juin
  - Visite gratuite de la distillerie des 4 Vallées | Chamaloc | 6 - 9 juin
  - Zoom sur une architecture d'exception au château de Montélimar | Montélimar | 20 et 21 septembre 2025

============================================================
QUESTION : Que faire ce week-end dans la Drôme ?
============================================================
2026-03-16 21:01:19,373 - INFO - Question : 'Que faire ce week-end dans la Drôme ?'
2026-03-16 21:01:19,805 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-16 21:01:19,809 - INFO - Documents récupérés : 10
2026-03-16 21:01:19,809 - INFO - AFC is enabled with max remote calls: 10.
2026-03-16 21:01:29,001 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent "HTTP/1.1 200 OK"
2026-03-16 21:01:29,004 - INFO - Réponse générée (1735 caractères, 10 sources).

RÉPONSE :
 Nous sommes le lundi 16 mars 2026. D'après les documents fournis, il n'y a aucun événement programmé pour le week-end prochain (samedi 21 et dimanche 22 mars 2026).

Tous les événements listés dans les documents se déroulent plus tard dans l'année 2026 (ou sont passés par rapport à aujourd'hui, comme ceux de 2025).

Voici les événements à venir dans la Drôme pour lesquels vous pourriez planifier une sortie ultérieurement :

  - Découverte d'un lieu d'exception : le palais médiéval de Montélimar | Montélimar | 20 et 21 septembre 2025
  - Exposition Amazônia de Dalva Duarte | Montélimar | 20 et 21 septembre
  - Visite enrichie d'une exploitation agricole | Chamaloc | 6 - 9 juin
  - Visite gratuite de la distillerie des 4 Vallées | Chamaloc | 6 - 9 juin
  - Zoom sur une architecture d'exception au château de Montélimar | Montélimar | 20 et 21 septembre 2025

============================================================
QUESTION : Que faire ce week-end dans la Drôme ?
============================================================
2026-03-16 21:01:19,373 - INFO - Question : 'Que faire ce week-end dans la Drôme ?'
2026-03-16 21:01:19,805 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-16 21:01:19,809 - INFO - Documents récupérés : 10
2026-03-16 21:01:19,809 - INFO - AFC is enabled with max remote calls: 10.
2026-03-16 21:01:29,001 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent "HTTP/1.1 200 OK"
2026-03-16 21:01:29,004 - INFO - Réponse générée (1735 caractères, 10 sources).

RÉPONSE :
 Nous sommes le lundi 16 mars 2026. D'après les documents fournis, il n'y a aucun événement programmé pour le week-end prochain (samedi 21 et dimanche 22 mars 2026).

Tous les événements listés dans les documents se déroulent plus tard dans l'année 2026 (ou sont passés par rapport à aujourd'hui, comme ceux de 2025).

Voici les événements à venir dans la Drôme pour lesquels vous pourriez planifier une sortie ultérieurement :

2026-03-16 21:01:19,373 - INFO - Question : 'Que faire ce week-end dans la Drôme ?'
2026-03-16 21:01:19,805 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents "HTTP/1.1 200 OK"
2026-03-16 21:01:19,809 - INFO - Documents récupérés : 10
2026-03-16 21:01:19,809 - INFO - AFC is enabled with max remote calls: 10.
2026-03-16 21:01:29,001 - INFO - HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent "HTTP/1.1 200 OK"
2026-03-16 21:01:29,004 - INFO - Réponse générée (1735 caractères, 10 sources).

RÉPONSE :
 Nous sommes le lundi 16 mars 2026. D'après les documents fournis, il n'y a aucun événement programmé pour le week-end prochain (samedi 21 et dimanche 22 mars 2026).

Tous les événements listés dans les documents se déroulent plus tard dans l'année 2026 (ou sont passés par rapport à aujourd'hui, comme ceux de 2025).

Voici les événements à venir dans la Drôme pour lesquels vous pourriez planifier une sortie ultérieurement :

RÉPONSE :
 Nous sommes le lundi 16 mars 2026. D'après les documents fournis, il n'y a aucun événement programmé pour le week-end prochain (samedi 21 et dimanche 22 mars 2026).

Tous les événements listés dans les documents se déroulent plus tard dans l'année 2026 (ou sont passés par rapport à aujourd'hui, comme ceux de 2025).

Voici les événements à venir dans la Drôme pour lesquels vous pourriez planifier une sortie ultérieurement :


Tous les événements listés dans les documents se déroulent plus tard dans l'année 2026 (ou sont passés par rapport à aujourd'hui, comme ceux de 2025).

Voici les événements à venir dans la Drôme pour lesquels vous pourriez planifier une sortie ultérieurement :


Voici les événements à venir dans la Drôme pour lesquels vous pourriez planifier une sortie ultérieurement :

*   **Juin 2026 :**

*   **Juin 2026 :**
*   **Juin 2026 :**
    *   **Visite enrichie d'une exploitation agricole** et **Visite gratuite de la distillerie des 4 Vallées** à la Distillerie des 4 Vallées (Chamaloc), du 6 au 9 juin.
    *   **Cédille Tour 2025 : les tiers-lieux de Nyons** à Asoft (Nyons), le vendredi 20 juin à 09h00.
    *   **Cédille Tour 2025 : les tiers-lieux de Nyons** à Asoft (Nyons), le vendredi 20 juin à 09h00.

*   **Septembre 2026 :**
    *   **Atelier sténopé** aux Archives départementales de la Drôme (Valence), le samedi 20 septembre à 14h30.
*   **Septembre 2026 :**
    *   **Atelier sténopé** aux Archives départementales de la Drôme (Valence), le samedi 20 septembre à 14h30.
    *   **Atelier sténopé** aux Archives départementales de la Drôme (Valence), le samedi 20 septembre à 14h30.
    *   **Les 50 ans du bâtiment des Archives** aux Archives départementales de la Drôme (Valence), les 20 et 21 septembre.
    *   **Les 50 ans du bâtiment des Archives** aux Archives départementales de la Drôme (Valence), les 20 et 21 septembre.
    *   **Visite guidée du village perché de Mirmande** à l'Église Sainte Foy (Mirmande), les 20 et 21 septembre.
    *   **Visite guidée du Prieuré Notre Dame d'Andéa à Grâne** au Prieuré (Grâne), les 20 et 21 septembre.
    *   **Visite guidée du village perché de Mirmande** à l'Église Sainte Foy (Mirmande), les 20 et 21 septembre.
    *   **Visite guidée du Prieuré Notre Dame d'Andéa à Grâne** au Prieuré (Grâne), les 20 et 21 septembre.
    *   **Visite de la préfecture du conseil départemental de la Drôme** à la Préfecture de la Drôme (Valence), le dimanche 21 septembre.
    *   **Visite guidée : l'Hôtel de Ville de Romans** à l'Hôtel de Ville de Romans-sur-Isère, le dimanche 21 septembre à 14h30.

*   **Octobre 2026 :**
    *   **Café-outils : Quelles alternatives aux outils numériques de la "Big Tech" ?** au Moulin Digital (Alixan), le vendredi 3 octobre à 09h00.

*Note : Certains documents mentionnent des dates en 2025, ces événements sont donc passés.*

SOURCES :
  - Visite enrichie d'une exploitation agricole | Chamaloc | 6 - 9 juin
  - Visite gratuite de la distillerie des 4 Vallées | Chamaloc | 6 - 9 juin
  - Cédille Tour 2025 : les tiers-lieux de Nyons | Nyons | Vendredi 20 juin, 09h00
  - Visite guidée du village perché de Mirmande | Mirmande | 20 et 21 septembre
  - Visite de la préfecture du conseil départemental de la Drôme | Valence | Dimanche 21 septembre, 10h30, 11h30, 14h00, 15h00
  - Atelier sténopé | Valence | Samedi 20 septembre, 14h30
  - Visite guidée : l'Hôtel de Ville de Romans | Romans-sur-Isère | Dimanche 21 septembre, 14h30
  - Café-outils : Quelles alternatives aux outils numériques de la "Big Tech" ? | Alixan | Vendredi 3 octobre, 09h00
  - Visite guidée du Prieuré Notre Dame d'Andéa à Grâne | Grane | 20 et 21 septembre
  - Les 50 ans du bâtiment des Archives | Valence | 20 et 21 septembre