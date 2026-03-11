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
├── src/                # Code source principal (ingestion, chunking, embeddings, RAG)
│   └── ingestion.py    # Collecte et prétraitement des données Open Agenda
├── api/                # Endpoints FastAPI
├── tests/              # Tests unitaires et scripts de validation
├── data/               # Données collectées et traitées
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
# source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate    # Windows
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

## Pipeline de données

### Collecte et prétraitement

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

Les données pré-collectées sont disponibles dans le dossier `data/` pour une utilisation immédiate sans appel API.

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
| Source de données       | Open Agenda (OpenDataSoft)   |

## Licence

Projet réalisé dans le cadre de la formation OpenClassrooms — Ingénieur IA.