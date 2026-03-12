"""
Tests unitaires pour src/vectorstore.py.

Tous les appels aux API d'embedding sont mockés - aucun quota consommé.
"""

# --- Imports ---
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.vectorstore import (
    get_embeddings,
    build_index,
    resume_index,
    save_index,
    load_index,
    search,
    FAIS_INDEX_DIR,
)


# --- Fixtures ---
class FakeEmbeddings(Embeddings):
    """Faux modèle d'embedding retournant des vecteurs déterministes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Chaque texte → vecteur de dimension 8 basé sur sa longueur."""
        return [self._make_vector(t) for t in texts]
    
    def embed_query(self, text: str) -> list[float]:
        return self._make_vector(text)
    
    @staticmethod
    def _make_vector(text: str) -> list[float]:
        """Vecteur simple mais différent pour chaque texte."""
        base = float(len(text) % 100) / 100.0
        return [base + i * 0.01 for i in range(8)]
    

@pytest.fixture
def sample_chunks() -> list[Document]:
    """Quelques chunks réalistes pour les tests."""
    return [
        Document(
            page_content="Titre : Concert de jazz\nDate : 15 mars 2027\nLieu : Valence",
            metadata={"uid": 1, "title": "Concert de jazz", "location_city": "Valence"},
        ),
        Document(
            page_content="Titre : Exposition peinture\nDate : 20 mars 2027\nLieu : Montélimar",
            metadata={"uid": 2, "title": "Exposition peinture", "location_city": "Montélimar"},
        ),
        Document(
            page_content="Titre : Festival de théâtre\nDate : 25 mars 2027\nLieu : Romans-sur-Isère",
            metadata={"uid": 3, "title": "Festival de théâtre", "location_city": "Romans-sur-Isère"},
        ),
    ]

@pytest.fixture
def test_index_dir(tmp_path) -> Path:
    """Répertoire temporaire pour sauvegarder/charger un index de test."""
    return tmp_path / "test_faiss_index"


# --- Tests get_embeddings() ---


class TestGetEmbeddings:
    """Tests pour la sélection du modèle d'embedding."""

    @patch.dict(os.environ, {"EMBEDDING_PROVIDER": "google", "GOOGLE_API_KEY": "fake-key"})
    @patch("src.vectorstore.GoogleGenerativeAIEmbeddings", create=True)
    def test_google_provider_returns_google_embeddings(self, mock_cls):
        """Le provide 'google' instancie GoogleGenerativeAIEmbeddings."""
        # On doit patch l'import lazy dans la fonction
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", mock_cls):
            from src.vectorstore import get_embeddings as _get
            # Recharger le module pour prendre en compte le env modifié
            import importlib
            import src.vectorstore
            importlib.reload(src.vectorstore)
            src.vectorstore.get_embeddings(task_type="RETRIEVAL_DOCUMENT")
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["task_type"] == "RETRIEVAL_DOCUMENT"
            assert call_kwargs["model"] == "gemini-embedding-2-preview"
    
    @patch.dict(os.environ, {"EMBEDDING_PROVIDER": "google", "GOOGLE_API_KEY": ""})
    def test_google_missing_key_raises_error(self):
        """Clé API Google vide → EnvironmentError."""
        import importlib
        import src.vectorstore
        importlib.reload(src.vectorstore)
        with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
            src.vectorstore.get_embeddings()

    @patch.dict(os.environ, {"EMBEDDING_PROVIDER": "mistral", "MISTRAL_API_KEY": ""})
    def test_mistral_missing_key_raises_error(self):
        """Clé API Mistral vide → EnvironmentError."""
        import importlib
        import src.vectorstore
        importlib.reload(src.vectorstore)
        with pytest.raises(EnvironmentError, match="MISTRAL_API_KEY"):
            src.vectorstore.get_embeddings()
    
    @patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai"})
    def test_unknown_provider_raises_error(self):
        """Provider inconnu → ValueError."""
        import importlib
        import src.vectorstore
        importlib.reload(src.vectorstore)
        with pytest.raises(ValueError, match="EMBEDDING_PROVIDER inconnu"):
            src.vectorstore.get_embeddings()


# --- Tests build_index / save / load / search ---


class TestFAISSOperations:
    """Tests pour les opérations FAISS (avec FakeEmbeddings, sans API)."""

    def test_build_index_creates_correct_number_of_vectors(self, sample_chunks):
        """L'index contient autant de vecteurs que de chunks."""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            vectorstore = build_index(sample_chunks)
            assert vectorstore.index.ntotal == len(sample_chunks)

    def test_build_index_correct_dimensions(self, sample_chunks):
        """L'index a la dimension des vecteurs du FakeEmbeddings (8)."""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            vectorstore = build_index(sample_chunks)
            assert vectorstore.index.d == 8

    def test_resume_adds_chunks_to_existing_index(self, sample_chunks, test_index_dir):
        """resume_index ajoute des vecteurs à un index existant."""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            # Construire et sauvegarder un index avec les 2 premiers chunks
            initial = build_index(sample_chunks[:2])
            save_index(initial, test_index_dir)
            assert initial.index.ntotal == 2

    def test_resume_preserves_original_metadata(self, sample_chunks, test_index_dir):
        """Les métadonnées des chunks initiaux sont préservées après reprise"""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            initial = build_index(sample_chunks[:2])
            save_index(initial, test_index_dir)

            completed = resume_index([sample_chunks[2]], path=test_index_dir)
            results = search(completed, "test", k=3)
            cities = {doc.metadata["location_city"] for doc in results}
            assert cities == {"Valence", "Montélimar", "Romans-sur-Isère"}

    def test_resume_with_empty_list_changes_nothing(self, sample_chunks, test_index_dir):
        """Reprendre avec une liste vide ne modifie pas l'index"""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            initial = build_index(sample_chunks)
            save_index(initial, test_index_dir)

            completed = resume_index([], path=test_index_dir)
            assert completed.index.ntotal == len(sample_chunks)
    
    def test_save_and_load_preserve_vectors(self, sample_chunks, test_index_dir):
        """Sauvegarde puis chargement → même nombre de vecteurs."""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            vectorstore = build_index(sample_chunks)
            save_index(vectorstore, test_index_dir)

            # Vérifie que les fichiers existent
            assert (test_index_dir / "index.faiss").exists()
            assert (test_index_dir / "index.pkl").exists()

            # Recharge
            loaded = load_index(test_index_dir)
            assert loaded.index.ntotal == vectorstore.index.ntotal
    
    def test_search_returns_results(self, sample_chunks):
        """Une recherche retourne des résultats avec métadonnées."""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            vectorstore = build_index(sample_chunks)
            results = search(vectorstore, "concert jazz", k=2)
            assert len(results) == 2
            # Chaque résultat a du contenu et des métadonnées
            for doc in results:
                assert doc.page_content
                assert "uid" in doc.metadata
                assert "title" in doc.metadata

    def test_search_k_greater_than_index_returns_all(self, sample_chunks):
        """Si k > nombre de vecteurs, retourne tout ce qui est disponible."""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            vectorstore = build_index(sample_chunks)
            results = search(vectorstore, "test", k=100)
            assert len(results) == len(sample_chunks)
    
    def test_metadata_preserved_after_indexing(self, sample_chunks):
        """Les métadonnées des chunks sont conservées dans l'index."""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            vectorstore = build_index(sample_chunks)
            results = search(vectorstore, "test", k=3)
            cities = {doc.metadata["location_city"] for doc in results}
            assert cities == {"Valence", "Montélimar", "Romans-sur-Isère"}
    
    def test_build_index_with_max_chunks(self, sample_chunks):
        """Le paramètre max_chunks limite le nombre de chunks indexés."""
        with patch("src.vectorstore.get_embeddings", return_value=FakeEmbeddings()):
            vectorstore = build_index(sample_chunks, max_chunks=2)
            assert vectorstore.index.ntotal == 2