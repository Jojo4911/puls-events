"""
Configuration et accès au LLM (Large Language Model).

Ce module gère :
- La sélection du LLM (Mistral ou Google, via variable d'env LLM_PROVIDER)
- La configuration commune (température basse pour le RAG)
- Un point d'entrée unique get_llm() utilisé par la chaîne RAG
"""

# --- Imports ---
import os
import logging

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuration --

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google") # "Mistral" ou "Google"


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """
    Retourne le LLM configuré selon LLM_PROVIDER.

    - "mistral" → ChatMistralAI (modèle : mistral-larg-latest)
    - "google" → ChatGoogleGenerativeAI (modèle : gemini-3.1-flash-lite-preview)

    Args:
        temperature: Contrôle la créativité des réponses.
            Valeur basse (0.1-0.3) recommandée pour le RAG
            afin de limiter les hallucinations.
    
    Returns:
        Instance LangChaine BaseChatModel prête à l'emploi.
    
    Raises:
        ValueError: Si le provider n'est pas reconnu.
        EnvironmentError: Si la clé API correspondante est absente.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY non définie dans .env")
        
        from langchain_mistralai import ChatMistralAI # Lazy import
        logger.info(
            "LLM provider : Mistral (mistral-large-latest), temperature=%.1f",
            temperature,
        )
        return ChatMistralAI(
            model="mistral-large-latest",
            api_key=api_key,
            temperature=temperature,
        )
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY non définie dans .env")
        
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info(
            "LLM provider : Google (gemini-3.1-flash-lite-preview), temperature=%.1f",
            temperature,
        )
        return ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            google_api_key=api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"LLM_PROVIDER inconnu : '{provider}'."
            "Valeurs acceptées : 'mistral', 'google'."
        )


def extract_text(response) -> str:
    """
    Extrait le texte d'une réponse LLM, quel que soit le format.

    Gère les deux cas :
    - str (Mistral) : retourne tel quel
    - list[dict] (Gemini 3.x) : extrait les blocs de type 'text'
    """
    content = response.content

    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        return "\n".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    
    return str(content)


# --- Test rapide ---


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    llm = get_llm()
    print(f"LLM chargé : {llm.__class__.__name__}\n")

    response = llm.invoke(
        "Donne-moi un exemple d'événement culturel dans la Drôme, en une phrase."
    )
    print(f"Réponse : {extract_text(response)}")