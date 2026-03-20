# Puls-Events — Assistant de recommandation d'événements culturels

## Objectifs

Puls-Events est un assistant intelligent de recommandation d'événements culturels, développé dans le cadre d'un projet d'ingénierie IA.

Le système repose sur une architecture **RAG (Retrieval-Augmented Generation)** qui combine :
- Une **base de données vectorielle FAISS** indexant les événements culturels issus d'Open Agenda
- Un **LLM (Mistral / Gemini)** pour générer des réponses personnalisées et contextualisées
- Une **API REST (FastAPI)** exposant le système pour une utilisation par les équipes métier

L'objectif est de permettre à un utilisateur de poser une question en langage naturel (ex : *"Quels concerts ont lieu dans la Drôme ce week-end ?"*) et d'obtenir des recommandations pertinentes basées sur les données réelles d'événements.

## Structure du projet

```
puls-events/
├── src/                    # Code source principal
│   ├── __init__.py         # Package Python
│   ├── ingestion.py        # Collecte et prétraitement des données Open Agenda
│   ├── chunking.py         # Découpage des documents pour la vectorisation
│   ├── vectorstore.py      # Embeddings, index FAISS (build/save/load/search)
│   ├── build_index.py      # Script de construction/reprise de l'index FAISS
│   ├── llm.py              # Configuration LLM (get_llm(), extract_text())
│   └── rag_system.py       # Chaîne RAG (RAGSystem, format_docs(), prompt)
├── api/                    # API REST FastAPI
│   ├── __init__.py         # Package Python
│   ├── main.py             # Application FastAPI, endpoints, lifespan
│   └── schemas.py          # Modèles Pydantic (AskRequest, AskResponse, SourceItem)
├── scripts/                # Scripts utilitaires et d'évaluation
│   ├── test_search.py      # Tests manuels de recherche sémantique
│   ├── evaluate_manual.py  # Évaluation manuelle interactive du RAG
│   └── evaluate_rag.py     # Évaluation automatisée avec Ragas
├── tests/                  # Tests unitaires et d'intégration
│   ├── test_imports.py     # Validation des imports clés
│   ├── test_ingestion.py   # Tests du pipeline de collecte (33 tests)
│   ├── test_chunking.py    # Tests du chunking
│   ├── test_vectorstore.py # Tests de l'index FAISS et des embeddings (14 tests)
│   ├── test_llm.py         # Tests de la configuration LLM
│   ├── test_rag_system.py  # Tests de la chaîne RAG et du formatage
│   └── test_api.py         # Tests de l'API FastAPI (mockés)
├── data/                   # Données collectées, traitées et résultats
│   ├── events_drome.csv    # Événements bruts nettoyés (1068 événements)
│   ├── chunks_drome.json   # Chunks pour vectorisation (1403 chunks)
│   ├── test_dataset.json   # Jeu de test annoté (20 questions, 5 catégories)
│   ├── evaluation_results.json  # Résultats de l'évaluation manuelle
│   └── ragas_results.json  # Scores Ragas automatisés
├── faiss_index/            # Index FAISS sauvegardé
├── Dockerfile              # Image Docker pour l'API
├── .dockerignore           # Fichiers exclus du build Docker
├── docs/                   # Documentation technique et rapports
├── .env.example            # Template des variables d'environnement
├── .gitignore              # Fichiers et dossiers exclus du versionnement
├── pyproject.toml          # Configuration du projet et dépendances (UV)
├── uv.lock                 # Verrouillage des versions exactes des dépendances
├── requirements.txt        # Dépendances (généré depuis uv.lock pour compatibilité pip)
├── notes.md                # Justifications techniques, résultats, observations
└── README.md
```

## Installation

### Prérequis

- Python ≥ 3.12
- [UV](https://docs.astral.sh/uv/) (recommandé) ou pip

### Installation avec UV

```bash
git clone https://github.com/Jojo4911/puls-events.git
cd puls-events
uv sync
```

### Installation avec pip

```bash
git clone https://github.com/Jojo4911/puls-events.git
cd puls-events
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Configuration

Copiez le fichier d'environnement et renseignez vos clés API :

```bash
cp .env.example .env
```

Variables à configurer dans `.env` :

| Variable              | Description                                          |
|-----------------------|------------------------------------------------------|
| `MISTRAL_API_KEY`     | Clé API Mistral                                      |
| `GOOGLE_API_KEY`      | Clé API Google Gemini                                |
| `OPENAGENDA_API_KEY`  | Clé API Open Agenda                                  |
| `EMBEDDING_PROVIDER`  | Provider d'embeddings : `mistral` ou `google`        |
| `LLM_PROVIDER`        | Provider LLM : `mistral` ou `google`                 |

### Vérification de l'environnement

```bash
uv run pytest tests/ -v
```

Tous les tests doivent passer pour confirmer que l'environnement est correctement configuré.

## Pipeline de données

### 1. Collecte et prétraitement

Le script `src/ingestion.py` gère la collecte et le nettoyage des données en un seul pipeline :

```bash
uv run python src/ingestion.py
```

Ce script :
1. **Récupère** les événements culturels du département de la Drôme via l'API Open Agenda (OpenDataSoft), avec pagination automatique
2. **Filtre** les événements des 12 derniers mois
3. **Nettoie** les données : suppression des balises HTML, déduplication, suppression des entrées vides
4. **Structure** chaque événement en un texte formaté pour la vectorisation (titre, date, lieu, description, mots-clés)
5. **Sauvegarde** le dataset nettoyé en JSON et CSV dans `data/`

### 2. Chunking

Le script `src/chunking.py` découpe les documents pour la vectorisation :

```bash
uv run python src/chunking.py
```

Ce script :
1. **Convertit** chaque événement en objet Document LangChain avec métadonnées (titre, date, lieu, URL)
2. **Découpe** les textes longs via `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=150)
3. **Propage** les métadonnées de l'événement source à chaque chunk
4. **Sauvegarde** les chunks en JSON dans `data/`

Stratégie de chunking : l'analyse de la distribution des longueurs a montré que 85% des textes font moins de 1000 caractères. Le `chunk_size` de 1000 préserve l'intégrité de la majorité des événements tout en découpant les 15% restants en 2-3 chunks avec chevauchement.

Les données pré-collectées et pré-chunkées sont disponibles dans le dossier `data/` pour une utilisation immédiate sans appel API.

### 3. Vectorisation et index FAISS

Le module `src/vectorstore.py` gère les embeddings et l'index FAISS. Le script `src/build_index.py` orchestre la construction complète :

```bash
# Construction complète de l'index
uv run python -m src.build_index

# Construction partielle (ex : 900 premiers chunks, utile pour gérer les quotas API de google)
uv run python -m src.build_index 900

# Reprise d'un index partiel (ajoute automatiquement les chunks manquants)
uv run python -m src.build_index --resume
```

Ce pipeline :
1. **Charge** les chunks depuis le CSV nettoyé
2. **Vectorise** chaque chunk via l'API d'embeddings (Mistral ou Google selon `EMBEDDING_PROVIDER`)
3. **Indexe** les vecteurs dans FAISS avec conservation des métadonnées
4. **Sauvegarde** l'index sur disque dans `faiss_index/`

L'architecture supporte le switch entre providers d'embeddings via la variable `EMBEDDING_PROVIDER`. Pour le provider Google, les embeddings sont optimisés par type de tâche : `RETRIEVAL_DOCUMENT` à l'indexation et `RETRIEVAL_QUERY` à la recherche, ce qui améliore la pertinence des résultats.

**Note sur le choix du modèle d'embedding :** le modèle cible est Mistral (`mistral-embed`), conformément aux exigences du projet. En raison d'un problème d'accès à l'API Mistral (ticket de support ouvert, non résolu au moment du développement), le modèle Google `gemini-embedding-2-preview` (768 dimensions) est utilisé en fallback. L'architecture permet de switcher entre les deux providers sans modification de code.

### 4. Système RAG

Le module `src/rag_system.py` orchestre la chaîne RAG complète via la classe `RAGSystem` :

1. **Charge** l'index FAISS et initialise le retriever (k=10 documents)
2. **Récupère** les documents pertinents via recherche sémantique
3. **Formate** les résultats avec métadonnées (titre, lieu, dates en français via `format_datetime_fr()`)
4. **Génère** une réponse contextualisée via le LLM (configuré dans `src/llm.py`)

Le prompt RAG injecte la date du jour (`{today_date}`) pour permettre au LLM de distinguer les événements passés des événements à venir, et inclut des règles pour suggérer des alternatives quand aucun résultat exact n'est trouvé.

La configuration LLM supporte le switch entre providers via `LLM_PROVIDER` dans `.env`. Le module `src/llm.py` expose `get_llm()` et `get_embeddings()` avec gestion transparente des différences de format de réponse entre providers.

## API REST

L'API expose le système RAG via FastAPI :

```bash
uv run uvicorn api.main:app --reload
```

Documentation Swagger interactive disponible à `http://127.0.0.1:8000/docs`.

### Endpoints

| Méthode | Route      | Description                                                                       |
|---------|------------|-----------------------------------------------------------------------------------|
| GET     | `/`        | Redirection vers la documentation Swagger                                         |
| GET     | `/health`  | État de santé de l'API                                                            |
| POST    | `/ask`     | Poser une question au système RAG                                                 |
| POST    | `/rebuild` | Reconstruire la base vectorielle (`?limit=N` pour limiter le nombre d'événements) |

### Exemple d'utilisation

```bash
# Via Swagger UI (recommandé)
# Ouvrir http://127.0.0.1:8000/docs dans le navigateur

# Via Python
import requests
response = requests.post("http://127.0.0.1:8000/ask", json={"question": "Quels événements à Valence ?"})
print(response.json())
```

Le RAGSystem est initialisé une seule fois au démarrage du serveur via le mécanisme `lifespan` de FastAPI, évitant de recharger l'index FAISS à chaque requête.

## Évaluation

### Jeu de test annoté

20 paires question/réponse de référence réparties en 5 catégories : lieu (3), type (4), date (4), croisé (4), hors_périmètre (5). Les ground truths sont vérifiées contre les données source (`events_drome.csv`).

### Évaluation manuelle

```bash
uv run python scripts/evaluate_manual.py
```

Classification interactive des réponses (correcte / partiellement correcte / incorrecte) avec calcul des scores par catégorie.

| Catégorie       | Score  |
|-----------------|--------|
| lieu            | 100%   |
| croisé          | 100%   |
| hors_périmètre  | 100%   |
| type            | 87.5%  |
| date            | 37.5%  |

**Limitation identifiée :** la recherche purement sémantique FAISS ne filtre pas par date. Les questions ciblant une date précise échouent car le retriever retourne des documents sémantiquement proches mais aux mauvaises dates. C'est une limitation structurelle documentée, avec des pistes d'amélioration identifiées (recherche hybride, filtrage post-retrieval, migration vers un vector store avec filtrage de métadonnées).

### Évaluation automatisée (Ragas)

```bash
uv run python scripts/evaluate_rag.py
```

| Métrique           | Score  |
|--------------------|--------|
| faithfulness       | 0.624  |
| answer_relevancy   | 0.807  |
| context_precision  | 1.000  |
| context_recall     | 0.623  |

*Note : résultats partiels en raison des limites de taux du tier gratuit Gemini (15 RPM). `context_precision` non calculé (TimeoutError).*

### Tests

```bash
uv run pytest tests/ -v
```

## Technologies

| Composant               | Technologie                                |
|-------------------------|--------------------------------------------|
| Orchestration RAG       | LangChain                                  |
| Base vectorielle        | FAISS (faiss-cpu)                          |
| LLM                     | Mistral AI / Google Gemini (switchable)    |
| Embeddings              | Mistral Embed / Google Gemini (switchable) |
| API                     | FastAPI + Uvicorn                          |
| Évaluation              | Ragas                                      |
| Conteneurisation        | Docker                                     |
| Gestion des dépendances | UV                                         |
| Source de données       | Open Agenda (OpenDataSoft)                 |

## Licence

Projet réalisé dans le cadre de la formation OpenClassrooms — Ingénieur IA.