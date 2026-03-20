"""Création de l'API"""

# --- Imports ---
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse

from api.schemas import AskRequest, AskResponse
from src.rag_system import RAGSystem
from src.ingestion import fetch_events, clean_events, save_events
from src.chunking import load_and_chunk
from src.vectorstore import build_index, save_index

# --- Chargement du RAG System au début ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Instanciation du RAGSystem
    app.state.rag_system = RAGSystem()
    yield

# Création de l'instance FastAPI
app = FastAPI(
    title="Puls Event RAG API",
    description="RAG for events in Drôme",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Définition des Endpoints ---
@app.get("/", summary="Premier endpoint de base avec redirection sur /docs")
def read_root():
    return RedirectResponse(url="/docs")

@app.get("/health", summary="Renvoie l'état de santé de l'API")
def health_check():
    return {"status": "ok", "RAG system": "loaded"}

@app.post("/ask", summary="Permet de poser une question au RAG")
def ask_question(body: AskRequest, request: Request):
    rag = request.app.state.rag_system
    try:
        result = rag.ask(body.question)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur du système RAG : {e}")

@app.post("/rebuild", summary="Reconstruit toute la base de données vectorielle")
def rebuild_vector_database(request: Request):
    log_dict = {"message": "Échec de la reconstruction de la base de données vectorielle"}
    try:
        # === I. Ingestion ===
        # 1. Récupère tous les évènements depuis l'API.
        current_step = "fetch_events() de ingestion.py"
        raw_events = fetch_events()
        log_dict["fetch"] = f"Ingestion : {len(raw_events)} événements récupérés"
        # 2. Nettoie et structure les évènements bruts en DataFrame.
        current_step = "clean_events() de ingestion.py"
        df = clean_events(raw_events)
        log_dict["clean"] = f"Ingestion : {len(df)} événements nettoyés retenus"
        # 3. Sauvegarde le DataFrame en CSV dans le dossier data/.
        current_step = "save_events() de ingestion.py"
        events_csv = save_events(df)[1]
        log_dict["save events"] = f"Ingestion : Événements enregistrés dans {events_csv}"
        # === II. Chunking ===
        # 4. chargement CSV -> Documents -> Chunks.
        current_step = "load_and_chunk() de chunking.py"
        chunks = load_and_chunk(events_csv)
        log_dict["chunk"] = f"Chunking : {len(chunks)} événements chunkés"
        # === III. Indexation FAISS ===
        # 5. Construit un index FAISS à partir des chunks.
        current_step = "build_index de vectorstore.py"
        index = build_index(chunks)
        log_dict["build index"] = "Indexation : Construction de l'index réussi"
        # 6. Sauvegarde l'index FAISS sur le disque.
        current_step = "save_index() de vectorstore.py"
        faiss_path = save_index(index)
        log_dict["save index"] = "Indexation : Index enregistré"
        # 7. redémarrage du RAGSystem
        current_step = "app.state.rag_system = RAGSystem()"
        request.app.state.rag_system = RAGSystem()
        log_dict["message"] = "Reconstruction de la base de données réussie !"
        return log_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"La reconstruction s'est arrêtée à la fonction : {current_step} : {e}")