# Notes

## Tests de pertinence du rag_system

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