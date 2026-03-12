"""
Construction et gestion de l'index FAISS.

Ce module gère :
- La sélection du modèle d'embedding (Mistral ou Google, via variable d'env)
- Le support du task_type Google (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY)
- La construction de l'index FAISS à partir des chunks LangChain
- La sauvegarde et le chargement de l'index
- Les requêtes de similarité
"""

# --- Imports ---
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuration ---

FAIS_INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google") # "mistral" ou "google"


def get_embeddings(task_type: str | None = None):
    """
    Retourne le modèle d'embedding selon EMBEDDING_PROVIDER.

    - "mistal" -> MistralEmbeddings (modèle : mistral-embed)
    - "google" -> GoogleGenerativeAIEmbeddings (modèle : gemini-embedding-2-preview)

    Args:
        task_type: Type de tâche pour optimiser les embeddings Google.
            - "RETRIEVAL_DOCUMENT" pour l'indexation des chunks
            - "RETRIEVAL_QUERY" pour les requêtes utilisateur
            - None -> pas d'optimisation (ou Mistral, qui ne supporte pas ce paramètre)

    Raises:
        ValueError : Si le provider n'est pas reconnu.
        EnvironnementError : Si la clé API correspondante est absente.
    """
    provider = EMBEDDING_PROVIDER.lower()

    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY non définie dans .env")
        
        from langchain_mistralai import MistralAIEmbeddings # Lazy import : évite d’importer des packages potentiellement inutiles
        
        if task_type:
            logger.info(
                "Note : task_type='%s' ignoré (non supporté par Mistral)", task_type
            )
        
        logger.info("Embedding provider : Mistral (mistral-embed)")
        return MistralAIEmbeddings(
            model="mistral-embed",
            api_key=api_key,
        )
    
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY non définie dans .env")
        
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        logger.info(
            "Embedding provider : Google (gemini-embedding-2-preview, task_type=%s)",
            task_type or "non spécifié",
        )
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            api_key=api_key,
            task_type=task_type,
            output_dimensionality=768, # Taille de la dimension de sortie flexible avec Google
        )
    
    else:
        raise ValueError(
            f"EMBEDDING_PROVIDER inconnu : '{provider}'."
            "Valeurs acceptées : 'mistral', 'google'."
        )
    

# --- Construction de l’index ---


def build_index(
    chunks: list[Document],
    batch_size: int = 100,
    delay: float = 65.0,
    max_chunks: int | None = None,
) -> FAISS:
    """
    Construit un index FAISS à partir d'une liste de Document LangChain.

    Utilise task_type="RETRIEVAL_DOCUMENT" pour les embeddings Google,
    ce qui optimise les vecteurs pour être retrouvés par une recherche.

    Calibré pour respecter les limites de l'API Google (free tier) :
    - RPM = 100 textes/min → batch de 100 toutes les 65s
    - RPD = 1000 textes/jour → paramètre max_chunks pour limiter

    Args:
        chunks: Liste de Documents LangChain (issus du chunking).
        batch_size: Nombre de chunks par lot (défaut : 100)
        delay: Délai en secondes entre chaque lot (défaut : 65.0)
        max_chunks: Nombre max de chunks à traiter (None = tous).

    Returns:
        Index FAISS pour la recherche.
    """
    import time

    embeddings = get_embeddings(task_type="RETRIEVAL_DOCUMENT")

    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        logger.info("Limitation à %d chunks (max_chunks=%d).", len(chunks), max_chunks)

    total = len(chunks)
    total_batches = (total + batch_size - 1) // batch_size
    estimated_minutes = (total_batches * delay) / 60
    logger.info(
        "Construction de l'index FAISS : %d chunks en %d batchs "
        "(taille=%d, délai=%.0fs, durée estimée=%.1f min)...",
        total, total_batches, batch_size, delay, estimated_minutes,
    )

    vectorstore = None

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        # Retry avec backoff en cas de rate limit
        for attempt in range(8):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                logger.info(
                    "Batch %d/%d traité (%d chunks, total=%d/%d).",
                    batch_num, total_batches, len(batch),
                    min(i + batch_size, total), total,
                )
                break
            except Exception as e:
                if "429" in str(e) or "RESSOURCE_EXHAUSTED" in str(e):
                    wait = 10 * (2 ** attempt)
                    logger.warning(
                        "Rate limit atteint (batch %d, tentative %d/8)."
                        "Attente de %.0fs...",
                        batch_num, attempt + 1, wait
                    )
                    time.sleep(wait)
                else:
                    raise # Erreur non liée au rate limit → on la propage
        else:
            # Sauvegarde d'urgence avant de planter
            if vectorstore is not None:
                emergency_path = save_index(vectorstore, FAIS_INDEX_DIR / "partial")
                logger.error(
                    "Sauvegarde d'urgence effectuée dans %s (%d vecteurs).",
                    emergency_path, vectorstore.index.ntotal,
                )
            raise RuntimeError(
                f"Échec après 8 tentatives sur le batch {batch_num}."
                f"Index partiel sauvegardé ({vectorstore.index.ntotal} vecteurs)."
            )
    
    logger.info("Index FAISS construit : %d vecteurs.", vectorstore.index.ntotal)
    return vectorstore


def resume_index(
        remaining_chunks: list[Document],
        path: Path | str | None = None,
        batch_size: int = 100,
        delay: float = 65.0,
) -> FAISS:
    """
    Charge un index partiel et y ajoute de nouveaux chunks.

    Utilisé pour compléter un index après un reset de quota RPD (Requests Per Day).

    Args:
        remaining_chunks: Chunks à ajouter à l'index existant.
        path: Répertoire de l'index partiel. Par défaut : faiss_index/
        batch_size: Nombre de chunks par lot (défaut : 100)
        delay: Délai en secondes entre chaque lot (défaut : 65.0)

    Returns:
        Index FAISS pour la recherche complété.
    """
    import time

    vectorstore = load_index(path)
    embeddings = get_embeddings(task_type="RETRIEVAL_DOCUMENT")
    # Remplace l'embedding RETRIEVAL_QUERY du load_index par RETRIEVAL_DOCUMENT
    vectorstore.embedding_function = embeddings

    total = len(remaining_chunks)
    total_batches = (total + batch_size -1) // batch_size
    logger.info(
        "Reprise : ajout de %d chunks à l'index existant (%d vecteurs)...",
        total, vectorstore.index.ntotal,
    )

    for i in range(0, total, batch_size):
        batch = remaining_chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        for attempt in range(8):
            try:
                vectorstore.add_documents(batch)
                logger.info(
                    "Batch %d/%d traité (%d chunks).",
                    batch_num, total_batches, len(batch),
                )
                break
            except Exception as e:
                if "429" in str(e) or "RESSOURCE_EXHAUSTED" in str(e):
                    wait = 10 * (2 ** attempt)
                    logger.warning(
                        "Rate limit atteint (batch %d, tentative %d/8)."
                        "Attente de %.0fs...",
                        batch_num, attempt + 1, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        else:
            if vectorstore is not None:
                save_index(vectorstore)
            raise RuntimeError(f"Echec après 8 tentatives sur le batch {batch_num}.")
        
        if batch_num < total_batches:
            time.sleep(delay)

    logger.info(
        "Index FAISS complété : %d vecteurs au total.",
        vectorstore.index.ntotal,
    )
    return vectorstore


def save_index(vectorstore: FAISS, path: Path | str | None = None) -> Path:
    """
    Sauvegarde l'index FAISS sur le disque.

    Args:
        vectorstore: Index FAISS à sauvegarder.
        path: Répertoire de destination. Par défaut : faiss_index/
    
    Returns:
        Chemin du répertoire de sauvegarde.
    """
    if path is None:
        path = FAIS_INDEX_DIR
    
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(path))
    logger.info("Index FAISS sauvegardé dans %s", path)

    return path


def load_index(path: Path | str | None = None) -> FAISS:
    """
    Charge un index FAISS depuis le disque.

    Utilise task_type="RETRIEVAL_QUERY" pour les embeddings Google,
    car l'index chargé sera utilisé pour la recherche de requête.

    Args:
        path: Répertoire contenant l'index. Par défaut : faiss_index/

    Returns:
        Index FAISS prêt pour la recherche.
    """
    if path is None:
        path = FAIS_INDEX_DIR
    
    path = Path(path)

    embeddings = get_embeddings(task_type="RETRIEVAL_QUERY")

    logger.info("Chargement de l'index FAISS depuis %s", path)
    vectorstore = FAISS.load_local(
        str(path),embeddings=embeddings, allow_dangerous_deserialization=True
    )
    # On autorise allow_dangerous_deserialization car c’est nous qui avons créé l’index (source fiable)
    logger.info("Index FAISS chargé : %d vecteurs", vectorstore.index.ntotal)

    return vectorstore


# --- Recherche ---


def search(vectorstore: FAISS, query: str, k: int = 5) -> list[Document]:
    """
    Recherche les k documents les plus similaires à la requête.

    Args:
        vectorstore: Index FAISS.
        query: Requête en langage naturel.
        k: Nombre de résultats à retourner.

    Return:
        Liste de Documents LangChain les plus pertinents.
    """
    results = vectorstore.similarity_search(query, k=k)
    logger.info("Recherche '%s' -> '%s' résultats.", query[:50], len(results))
    return results