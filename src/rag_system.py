"""
Système RAG (Retrieval-Augmented Generation) pour Puls-Events.

Ce module orchestre la chaîne RAG complète :
    1. Récupération des documents pertinents (retriever FAISS)
    2. Construction du prompt avec contexte
    3. Génération de la réponse avec le LLM
    4. Extraction de la réponse et des sources

Utilise LCEL (LangChain Expression Language) pour assembler la chaîne.
"""

# --- Imports ---
import logging
from datetime import date

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.vectorstore import load_index
from src.llm import get_llm, extract_text

logger = logging.getLogger(__name__)

# --- Prompt RAG ---

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


def format_docs(docs: list[Document]) -> str:
    """
    Formate les documents récupérés pour les injecter dans le prompt.

    Chaque document inclut ses métadonnées clés (titre, lieu, dates)
    pour aider le LLM à formuler des réponses précises.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = (
            f"[Document {i}]\n"
            f"Titre : {meta.get('title', 'N/A')}\n"
            f"Lieu : {meta.get('location_name', 'N/A')} — {meta.get('location_city', 'N/A')}\n"
            f"Dates : {meta.get('date_display', 'N/A')}\n"
            f"Contenu : {doc.page_content}"
        )
        formatted.append(header)

    return "\n\n".join(formatted)


def format_date_fr(d: date) -> str:
    """
    Formate une date donnée par l'objet date en une date en texte naturel en français.

    "2026-03-16" → "lundi 16 mars 2026"

    Args:
        d: Date retournée par le module datetime.

    Returns:
        str: Date plus naturelle en français.
    """
    days_fr = {
        0: "lundi",
        1: "mardi",
        2: "mercredi",
        3: "jeudi",
        4: "vendredi",
        5: "samedi",
        6: "dimanche",
        }

    months_fr = {
        1: "janvier",
        2: "février",
        3: "mars",
        4: "avril",
        5: "mai",
        6: "juin",
        7: "juillet",
        8: "août",
        9: "septembre",
        10: "octobre",
        11: "novembre",
        12: "décembre",
        }

    return f"{days_fr[d.weekday()]} {d.day} {months_fr[d.month]} {d.year}"



# --- Classe RAGSystem ---


class RAGSystem:
    """
    Système RAG complet pour la recommandation d'événements culturels.

    Charge l'index FAISS et le LLM une seule fois à l'initialisation,
    puis expose une méthode ask() pour intéroger le système.

    Attributes:
        vectorstore: Index FAISS chargé.
        llm: modèle de langage configuré.
        retriever : Retriever LangChain (wrapper autour de FAISS).
        prompt: Template du prompt RAG.
    """

    def __init__(self, k: int = 10):
        """
        Initialise le système RAG.

        Args:
            k: Nombre de documents à récupérer par requête.
        """
        logger.info("Initialisation du RAGSystem...")

        # 1. Chargement de l'index FAISS
        self.vectorstore = load_index()
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        logger.info("Retriever configuré (k=%d).", k)

        # 2. Chargement du LLM
        self.llm = get_llm()

        # 3. Configuration du prompt
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        logger.info("RAGSystem prêt.")

    def ask(self, question: str) -> dict:
        """
        Pose une question au système RAG.

        Pipeline :
            1. Récupère les k documents les plus pertinents
            2. Formate le contexte avec les métadonnées
            3. Construit le prompt et invoque le LLM
            4. Retourne la réponse et les sources
        
        Args:
            question: Question en langage naturel.
        
        Returns:
            Dictionnaire avec :
                - "answer": Réponse générée par le LLM (str)
                - "source": Liste des documents utilisés comme contexte
        """
        # 1. Retrieval — récupère les documents pertinents
        logger.info("Question : '%s'", question[:80])
        docs = self.retriever.invoke(question)
        logger.info("Documents récupérés : %d", len(docs))

        # 2. Augmentation — construction du prompt avec le contexte
        context = format_docs(docs)

        # 3. Augmentation — récupération de la date du jour en chaîne de caractères
        today_date = format_date_fr(date.today())

        # 4. Génération — invocation de la chaîne de prompt → LLM
        chain = self.prompt | self.llm
        response = chain.invoke({
            "today_date": today_date,
            "context": context,
            "question": question,
        })

        answer = extract_text(response)

        # 5. Extraction des sources (métadonnées des documents)
        sources = [
            {
                "title": doc.metadata.get("title", "N/A"),
                "location_name": doc.metadata.get("location_name", "N/A"),
                "location_city": doc.metadata.get("location_city", "N/A"),
                "date_display": doc.metadata.get("date_display", "N/A"),
                "url": doc.metadata.get("url", ""),
            }
            for doc in docs
        ]

        logger.info("Réponse générée (%d caractères, %d sources).",
                    len(answer), len(sources))
        
        return {
            "answer": answer,
            "sources": sources,
        }
    

# --- Test rapide ---


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    rag = RAGSystem()

    test_questions = [
        "Y a-t-il un concert de jazz à Valence ?",
        "Quels spectacles pour enfants à Montélimar ?",
        "Que faire ce week-end dans la Drôme ?",
    ]

    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"QUESTION : {question}")
        print(f"{'=' * 60}")

        result = rag.ask(question)

        print(f"\nRÉPONSE :\n {result['answer']}")
        print(f"\nSOURCES :")
        for s in result["sources"]:
            print(f"  - {s['title']} | {s['location_city']} | {s['date_display']}")