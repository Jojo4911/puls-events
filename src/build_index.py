"""
Script de construction de l'index FAISS.

Pipeline complet :
    1. Chargement des événements nettoyés (CSV)
    2. Chunking (via src.chungking)
    3. Vectorisation et indexation FAISS (via src.vectorstore)
    4. Sauvegarde de l'index sur le disque

Deux modes d'utilisation :
    - Construction : python -m src.build_index [max_chunks]
    - Reprise :      python -m src.build_index --resume

Ce script est conçu pour être :
- Exécuté manuellement : `uv run python -m src.build_index
- Appelé par l'endpoint /rebuild de l'API FastAPI
"""

# --- Imports ---
import logging
import time

from src.chunking import load_and_chunk
from src.vectorstore import (
    build_index,
    save_index,
    load_index,
    resume_index,
    FAIS_INDEX_DIR,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def rebuild_index(max_chunks: int | None = None) -> dict:
    """
    Reconstruit l'index FAISS à partir des données sources.

    Args:
        max_chunks: Limite le nombre de chunks à traiter (pour gérer les quotas API)

    Returns:
        Dictionnaire avec les statistiques ed construction
        (utile pour l'endpoint /rebuild de l'API).
    """
    start = time.time()

    # 1. Chargement + Chunking
    logger.info("=== Étape 1/3 : Chargement et chunking ===")
    chunks = load_and_chunk()

    # 2. Construction de l'index FAISS
    logger.info("=== Étape 2/3 : Construction de l'index FAISS ===")
    vectorstore = build_index(chunks, max_chunks=max_chunks)

    # 3. Sauvegarde
    logger.info("=== Étape 3/3 : Sauvegarde de l'index ===")
    index_path = save_index(vectorstore)

    elapsed = time.time() - start

    stats = {
        "chunks": len(chunks),
        "vectors": vectorstore.index.ntotal,
        "dimension": vectorstore.index.d,
        "index_path": str(index_path),
        "duration_seconds": round(elapsed, 1),
        "complete": vectorstore.index.ntotal == len(chunks),
    }

    logger.info("=== Construction terminée ===")
    logger.info("Statistiques : %s", stats)

    return stats


def resume_partial_index() -> dict:
    """
    Charge l'index partiel existant et y ajoute les chunks manquants.

    Détermine automatiquement quels chunks restent à indexer
    en comparant le nombre de vecteurs dans l'index avec le nombre
    total de chunks.

    Returns:
        Dictionnaire avec les statistiques de reprise.
    """
    start = time.time()

    # 1. Chargement + Chunking
    logger.info("=== Étape 1/3 : Chargement et chunking ===")
    chunks = load_and_chunk()

    # 2. Chargement de l'index partiel
    logger.info("=== Étape 2/3 : Chargement de l'index partiel ===")
    existing = load_index()
    already_indexed = existing.index.ntotal
    logger.info("Index existant : %d vecteurs sur %d chunks.", already_indexed, len(chunks))

    if already_indexed >= len(chunks):
        logger.info("L'index est déjà complet, rien à faire.")
        return {
            "chunks_total": len(chunks),
            "chunks_indexed": already_indexed,
            "chunks_added": 0,
            "complete": True,
            "duration_seconds": 0,
        }
    
    remaining = chunks[already_indexed:]
    logger.info("Chunks restants à indexer : %d", len(remaining))

    # 3. Ajout des chunks manquants
    logger.info("=== Étape 3/3 : Ajout des chunks manquants ===")
    completed = resume_index(remaining)
    save_index(completed)

    elapsed = time.time() - start

    stats = {
        "chunks_total": len(chunks),
        "chunks_indexed": completed.index.ntotal,
        "chunks_added": len(remaining),
        "dimension": completed.index.d,
        "index_path": str(FAIS_INDEX_DIR),
        "complete": completed.index.ntotal == len(chunks),
        "duration_seconds": round(elapsed, 1),
    }

    logger.info("=== Reprise terminée ===")
    logger.info("Statistiques : %s", stats)

    return stats


if __name__ == "__main__":
    import sys

    if "--resume" in sys.argv:
        resume_partial_index()
    else:
        max_chunks = None
        for arg in sys.argv[1:]:
            if arg.isdigit():
                max_chunks = int(arg)
        rebuild_index(max_chunks=max_chunks)