"""Tests unitaires pour src/rag_system.py."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.rag_system import format_docs, RAGSystem


# --- Fixtures ---


def make_docs(n: int = 3) -> list[Document]:
    """Crée une liste de Documents de test avec métadonnées."""
    docs = []
    for i in range(n):
        docs.append(
            Document(
                page_content=f"Description de l'événement {i + 1}.",
                metadata={
                    "title": f"Événement {i + 1}",
                    "location_name": f"Salle {i + 1}",
                    "location_city": "Valence",
                    "date_display": f"Le {i + 10} mars 2026",
                    "url": f"https://example.com/event-{i + 1}",
                    "uid": f"uid-{i + 1}",
                    "first_date": f"2026-03-{i + 10}",
                    "last_date": f"2026-03-{i + 10}",
                    "location_address": f"{i + 1} rue Test",
                    "keywords": "test",
                },
            )
        )
    return docs


# --- Tests format_docs ---


class TestFormatDocs:
    """Tests pour la fonction format_docs()."""

    def test_format_single_doc(self):
        """format_docs() formate correctement un seul document."""
        docs = make_docs(1)
        result = format_docs(docs)

        assert "[Document 1]" in result
        assert "Titre : Événement 1" in result
        assert "Lieu : Salle 1 — Valence" in result
        assert "Dates : Le 10 mars 2026" in result
        assert "Description de l'événement 1." in result

    def test_format_multiple_docs(self):
        """format_docs() numérote les documents."""
        docs = make_docs(3)
        result = format_docs(docs)

        assert "[Document 1]" in result
        assert "[Document 2]" in result
        assert "[Document 3]" in result

    def test_format_missing_metadata(self):
        """format_docs() gère les métadonnées manquantes avec 'N/A'."""
        doc = Document(page_content="Texte sans méta.", metadata={})
        result = format_docs([doc])

        assert "Titre : N/A" in result
        assert "Lieu : N/A — N/A" in result

    def test_format_empty_list(self):
        """format_docs() retourne une chaîne vide pour une liste vide."""
        assert format_docs([]) == ""


# --- Tests RAGSystem ---


class TestRAGSystem:
    """Tests pour la classe RAGSystem."""

    @patch("src.rag_system.get_llm")
    @patch("src.rag_system.load_index")
    def test_init(self, mock_load_index, mock_get_llm):
        """RAGSystem s'initialise avec l'index et le LLM."""
        mock_vectorstore = MagicMock()
        mock_load_index.return_value = mock_vectorstore
        mock_get_llm.return_value = MagicMock()

        rag = RAGSystem(k=3)

        mock_load_index.assert_called_once()
        mock_get_llm.assert_called_once()
        mock_vectorstore.as_retriever.assert_called_once_with(
            search_kwargs={"k": 3}
        )

    @patch("src.rag_system.get_llm")
    @patch("src.rag_system.load_index")
    def test_ask_returns_answer_and_sources(self, mock_load_index, mock_get_llm):
        """ask() retourne un dict avec 'answer' et 'sources'."""
        # Setup mocks
        docs = make_docs(2)
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_load_index.return_value = mock_vectorstore

        # Mock LLM : simule une réponse Gemini
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Voici les événements trouvés à Valence."
        mock_llm.__or__ = MagicMock()  # Pour le pipe operator
        mock_get_llm.return_value = mock_llm

        rag = RAGSystem(k=2)

        # On patche la chaîne pour court-circuiter le pipe LCEL
        with patch.object(rag, "prompt") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            result = rag.ask("Concerts à Valence ?")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "Voici les événements trouvés à Valence."
        assert len(result["sources"]) == 2

    @patch("src.rag_system.get_llm")
    @patch("src.rag_system.load_index")
    def test_ask_sources_contain_metadata(self, mock_load_index, mock_get_llm):
        """ask() retourne les métadonnées des sources."""
        docs = make_docs(1)
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_load_index.return_value = mock_vectorstore

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Réponse test."
        mock_get_llm.return_value = mock_llm

        rag = RAGSystem(k=1)

        with patch.object(rag, "prompt") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            result = rag.ask("Test ?")

        source = result["sources"][0]
        assert source["title"] == "Événement 1"
        assert source["location_city"] == "Valence"
        assert source["date_display"] == "Le 10 mars 2026"
        assert source["url"] == "https://example.com/event-1"

    @patch("src.rag_system.get_llm")
    @patch("src.rag_system.load_index")
    def test_ask_calls_retriever_with_question(self, mock_load_index, mock_get_llm):
        """ask() passe bien la question au retriever."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = make_docs(1)
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_load_index.return_value = mock_vectorstore

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Réponse."
        mock_get_llm.return_value = mock_llm

        rag = RAGSystem()

        with patch.object(rag, "prompt") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            rag.ask("Festival été Drôme")

        mock_retriever.invoke.assert_called_once_with("Festival été Drôme")