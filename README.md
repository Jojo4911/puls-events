# Puls-Events — Assistant de recommandation d'événements culturels

## Objectifs

Puls-Events est un assistant intelligent de recommandation d'événements culturels, développé dans le cadre d'un projet d'ingénierie IA.

Le système repose sur une architecture **RAG (Retrieval-Augmented Generation)** qui combine :
- Une **base de données vectorielle FAISS** indexant les événements culturels issus d'Open Agenda
- Un **LLM (Mistral / Gemini)** pour générer des réponses personnalisées et contextualisées
- Une **API REST (FastAPI)** exposant le système pour une utilisation par les équipes métier

L'objectif est de permettre à un utilisateur de poser une question en langage naturel (ex : *"Quels concerts ont lieu à Lyon ce week-end ?"*) et d'obtenir des recommandations pertinentes basées sur les données réelles d'événements.

## Structure du projet

```
puls-events/
├── src/                # Code source principal (ingestion, chunking, embeddings, RAG)
├── api/                # Endpoints FastAPI
├── tests/              # Tests unitaires et scripts de validation
├── data/               # Données brutes et traitées (non versionnées)
├── docs/               # Documentation technique et rapports
├── .env.example        # Template des variables d'environnement
├── .gitignore          # Fichiers et dossiers exclus du versionnement
├── pyproject.toml      # Configuration du projet et dépendances (UV)
├── uv.lock             # Verrouillage des versions exactes des dépendances
├── requirements.txt    # Dépendances (généré depuis uv.lock pour compatibilité pip)
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

| Variable              | Description                        |
|-----------------------|------------------------------------|
| `MISTRAL_API_KEY`     | Clé API Mistral                    |
| `GOOGLE_API_KEY`      | Clé API Google Gemini (fallback)   |
| `OPENAGENDA_API_KEY`  | Clé API Open Agenda                |

### Vérification de l'environnement

```bash
uv run python tests/test_imports.py
# ou
python tests/test_imports.py
```

Tous les tests doivent passer pour confirmer que l'environnement est correctement configuré.

## Technologies

| Composant               | Technologie                  |
|-------------------------|------------------------------|
| Orchestration RAG       | LangChain                    |
| Base vectorielle        | FAISS (faiss-cpu)            |
| LLM                     | Mistral AI / Google Gemini   |
| Embeddings              | Mistral Embed / HuggingFace  |
| API                     | FastAPI + Uvicorn            |
| Évaluation              | Ragas                        |
| Conteneurisation        | Docker                       |
| Gestion des dépendances | UV                           |

## Licence

Projet réalisé dans le cadre de la formation OpenClassrooms — Ingénieur IA.