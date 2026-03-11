"""
Chunking des événements culturels pour la vectorisation.

Ce module convertit les événements nettoyés (DataFrame) en objets Document
LangChain, avec découpage des textes longs et conservation des métadonnées.

Stratégie choisie : RecursiveCharacterTextSplitter avec chunk_size=1000
et chunk_overlap=150. Justification basée sur l'analyse de la distribution :
- 84% des textes font < 1000 caractères → restent en un seul chunk
- 15% font entre 1000 et 5000 caractères → découpés en 2-5 chunks
- Le chevauchement de 150 caractères préserve le contexte entre les chunks
"""

# Imports
import json
import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document # Anciennement `from langchain.schema import Document`
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Conversion DataFrame -> Documents LangChain ---


def row_to_document(row: pd.Series) -> Document:
    """
    Convertit une ligne du DataFrame en objet Document LangChain.

    Le texte principal est `text_for_embedding`.
    Les métadonnées conservent les informations structurées de l'événement
    pour enrichir les réponses du RAG.
    """
    metadata = {
        "uid": row.get("uid", ""),
        "title": row.get("title", ""),
        "date_display": row.get("date_display", ""),
        "first_date": row.get("first_date", ""),
        "last_date": row.get("last_date", ""),
        "location_name": row.get("location_name", ""),
        "location_city": row.get("location_city", ""),
        "location_address": row.get("location_address", ""),
        "keywords": row.get("keywords", ""),
        "url": row.get("url", ""),
    }

    return Document(
        page_content=row["text_for_embedding"],
        metadata=metadata,
    )

def dataframe_to_documents(df: pd.DataFrame) -> list[Document]:
    """Convertit le DataFrame complet en liste de Documents LangChain."""
    df = df.fillna("") # Pour éviter les "NaN" dans le Json
    documents = []
    for _, row in df.iterrows():
        documents.append(row_to_document(row))
    
    logger.info("Conversion : %d événements -> %d documents LangChain", len(df), len(documents))
    return documents


# --- Chunking ---


def chunk_documents(
        documents: list[Document],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Découpe les documents en chunks avec le RecursiveCharacterTextSplitter.

    Les métadonnées de chaque document sont propagées à tous ses chunks.

    Args:
        documents: Liste de Documents LangChain.
        chunk_size: Taille maximale de chaque chunk (en caractères).
        chunk_overlap: Chevauchement entre les chunks.

    Returns:
        Liste de Documents découpés, avec métadonnées conservées.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # Statistiques
    unchanged = sum(1 for doc in documents if len(doc.page_content) <= chunk_size)
    split_count = len(documents) - unchanged

    logger.info(
        "Chunking terminé : %d documents -> %d chunks "
        "(chunk_size=%d, overlap=%d).",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    logger.info(
        " - %d documents non découpés (< %d car.)",
        unchanged, chunk_size,
    )
    logger.info(
        " - %d documents découpés en plusieurs chunks.",
        split_count,
    )

    return chunks


# --- Sauvegarde ---


def save_chunks(chunks: list[Document], filename: str = "chunks_drome") -> Path:
    """
    Sauvegarde les chunks en JSON pour inspection et réutilisation.

    Returns:
        Chemin du fichier sauvegardé.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / f"{filename}.json"

    chunks_data = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    logger.info("Chunks sauvegardés : %s (%d chunks)", json_path, len(chunks))
    return json_path


# --- Pipeline ---


def load_and_chunk(
        csv_path: Path | str | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Pipeline complet : chargement CSV -> Documents -> Chunks.

    Args:
        csv_path: Chemin vers le CSV nettoyé. Par défaut : data/events_drome.csv.
        chunk_size: Taille des chunks.
        chunk_overlap: Chevauchement.

    Returns:
        Liste de Documents LangChain chunkés avec métadonnées.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "events_drome.csv"
    
    logger.info("Chargement des données depuis %s", csv_path)
    df = pd.read_csv(csv_path)

    documents = dataframe_to_documents(df)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)

    return chunks


# --- Point d’entrée ---


def main():
    """Exécute le pipeline de chunking et sauvegarde les résultats."""
    chunks = load_and_chunk()
    save_chunks(chunks)

    # Résumé
    lengths = [len(c.page_content) for c in chunks]
    logger.info("--- Résumé du chunking ---")
    logger.info("Nombre total de chunks : %d", len(chunks))
    logger.info("Longueur min : %d | max : %d | moyenne : %.0f",
                min(lengths), max(lengths), sum(lengths) / len(lengths))
    
    # Exemple de chunk
    logger.info("--- Exemple de chunk ---")
    sample = chunks[0]
    logger.info("Contenu : %s...", sample.page_content[:200])
    logger.info("Métadonnées : %s", sample.metadata)


if __name__ == "__main__":
    main()