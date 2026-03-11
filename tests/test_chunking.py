"""
Tests unitaires pour le module de chunking (src/chunking.py).

Couvre : conversion DataFrame -> Documents, chunking avec métadonnées,
paramétrage du splitter, et validation des données sauvegardées.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document

from src.chunking import (
    chunk_documents,
    dataframe_to_documents,
    load_and_chunk,
    row_to_document,
)

# --- Chemin vers les données ---

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_PATH = DATA_DIR / "chunks_drome.json"


# --- Fixtures ---


@pytest.fixture
def sample_row():
    """Une ligne type du DataFrame nettoyé."""
    return pd.Series({
        "uid": "12345",
        "title": "Festival de Jazz",
        "description": "Un super festival de jazz",
        "long_description": "Un super festival de jazz en plein air avec des artistes internationaux.",
        "keywords": "Jazz, Musique, Festival",
        "date_display": "Samedi 15 juin, 20h00",
        "first_date": "2025-06-15T18:00:00+00:00",
        "last_date": "2025-06-15T18:00:00+00:00",
        "location_name": "Parc de la Mairie",
        "location_city": "Valence",
        "location_address": "26000 Valence",
        "url": "https://openagenda.com/event/12345",
        "text_for_embedding": "Titre : Festival de Jazz\nDate : Samedi 15 juin, 20h00\nLieu : Parc de la Mairie, 26000 Valence\nDescription : Un super festival de jazz en plein air.",
    })


@pytest.fixture
def sample_dataframe(sample_row):
    """Un DataFrame avec quelques lignes de test."""
    rows = []
    for i in range(5):
        row = sample_row.copy()
        row["uid"] = str(1000 + i)
        row["title"] = f"Événement {i}"
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def short_document():
    """Un document court qui ne sera pas découpé."""
    return Document(
        page_content="Titre : Concert\nDate : Vendredi 20 juin\nDescription : Un petit concert.",
        metadata={"uid": "1", "title": "Concert", "location_city": "Valence"},
    )


@pytest.fixture
def long_document():
    """Un document long qui sera découpé en plusieurs chunks."""
    long_text = "Titre : Grand Festival\nDescription : " + "Ceci est une phrase assez longue pour le test. " * 50
    return Document(
        page_content=long_text,
        metadata={"uid": "2", "title": "Grand Festival", "location_city": "Montélimar"},
    )


# =============================================================================
# Tests — Conversion row -> Document
# =============================================================================


class TestRowToDocument:
    """Tests pour la fonction row_to_document."""

    def test_returns_document(self, sample_row):
        doc = row_to_document(sample_row)
        assert isinstance(doc, Document)

    def test_page_content_is_text_for_embedding(self, sample_row):
        doc = row_to_document(sample_row)
        assert doc.page_content == sample_row["text_for_embedding"]

    def test_metadata_contains_uid(self, sample_row):
        doc = row_to_document(sample_row)
        assert doc.metadata["uid"] == "12345"

    def test_metadata_contains_all_keys(self, sample_row):
        doc = row_to_document(sample_row)
        expected_keys = [
            "uid", "title", "date_display", "first_date", "last_date",
            "location_name", "location_city", "location_address",
            "keywords", "url",
        ]
        for key in expected_keys:
            assert key in doc.metadata, f"Clé manquante dans metadata : {key}"
    
    def test_metadata_contains_url(self, sample_row):
        doc = row_to_document(sample_row)
        assert doc.metadata["url"] == "https://openagenda.com/event/12345"

    def test_handles_nan_values(self):
        row = pd.Series({
            "uid": "99",
            "title": "Test",
            "keywords": float("nan"),
            "text_for_embedding": "Titre : Test",
            "date_display": "",
            "first_date": "",
            "last_date": "",
            "location_name": "",
            "location_city": "",
            "location_address": "",
            "url": "",
        })
        doc = row_to_document(row)
        assert isinstance(doc, Document)


# =============================================================================
# Tests — Conversion DataFrame -> Documents
# =============================================================================


class TestDataframeToDocuments:
    """Tests pour la fonction dataframe_to_documents."""

    def test_returns_correct_count(self, sample_dataframe):
        docs = dataframe_to_documents(sample_dataframe)
        assert len(docs) == len(sample_dataframe)

    def test_all_are_documents(self, sample_dataframe):
        docs = dataframe_to_documents(sample_dataframe)
        assert all(isinstance(d, Document) for d in docs)

    def test_unique_uids(self, sample_dataframe):
        docs = dataframe_to_documents(sample_dataframe)
        uids = [d.metadata["uid"] for d in docs]
        assert len(set(uids)) == len(uids)

    def test_fillna_cleans_nan(self):
        df = pd.DataFrame([{
            "uid": "1",
            "title": "Test",
            "keywords": float("nan"),
            "description": "",
            "long_description": "",
            "date_display": "",
            "first_date": "",
            "last_date": "",
            "location_name": "",
            "location_city": "",
            "location_address": "",
            "url": "",
            "text_for_embedding": "Titre : Test\nDescription : Un test basique.",
        }])
        docs = dataframe_to_documents(df)
        assert docs[0].metadata["keywords"] == ""


# =============================================================================
# Tests — Chunking
# =============================================================================


class TestChunkDocuments:
    """Tests pour la fonction chunk_documents."""

    def test_short_document_not_split(self, short_document):
        chunks = chunk_documents([short_document], chunk_size=1000, chunk_overlap=150)
        assert len(chunks) == 1
        assert chunks[0].page_content == short_document.page_content
    
    def test_long_document_is_split(self, long_document):
        chunks = chunk_documents([long_document], chunk_size=1000, chunk_overlap=150)
        assert len(chunks) > 1
    
    def test_metadata_propagated_to_all_chunks(self, long_document):
        chunks = chunk_documents([long_document], chunk_size=1000, chunk_overlap=150)
        for chunk in chunks:
            assert chunk.metadata["uid"] == "2"
            assert chunk.metadata["title"] == "Grand Festival"
            assert chunk.metadata["location_city"] == "Montélimar"
    
    def test_chunk_size_respected(self, long_document):
        chunk_size=1000
        chunks = chunk_documents([long_document], chunk_size=chunk_size, chunk_overlap=150)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 50 # Petite marge pour les séparateurs
    
    def test_mixed_documents(self, short_document, long_document):
        chunks = chunk_documents([short_document, long_document], chunk_size=1000, chunk_overlap=150)
        assert len(chunks) >= 2  # au moins 1 pour le court + plusieurs pour le long
    
    def test_custom_parameters(self, long_document):
        chunks_small = chunk_documents([long_document], chunk_size=500, chunk_overlap=50)
        chunks_long = chunk_documents([long_document], chunk_size=2000, chunk_overlap=200)
        assert len(chunks_small) > len(chunks_long)

    def test_empty_list(self):
        chunks = chunk_documents([])
        assert chunks == []


# =============================================================================
# Tests d'intégration — données sauvegardées
# =============================================================================


class TestLoadAndChunk:
    """Test d'intégration pour le pipeline complet."""

    @pytest.fixture(autouse=True)
    def check_csv_exists(self):
        csv_path = DATA_DIR / "events_drome.csv"
        if not csv_path.exists():
            pytest.skip("CSV non généré — lancer d'abord : uv run python src/ingestion.py")

    def test_returns_list_of_documents(self):
        chunks = load_and_chunk()
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunks_have_metadata(self):
        chunks = load_and_chunk()
        assert "uid" in chunks[0].metadata
        assert "title" in chunks[0].metadata


class TestSavedChunks:
    """Validation du fichier chunks_drome.json généré par le pipeline."""

    @pytest.fixture(autouse=True)
    def check_chunks_exist(self):
        if not CHUNK_PATH.exists():
            pytest.skip("Chunks non générés — lancer d'abord : uv run python src/chunking.py")
    
    def test_json_is_valid(self):
        with open(CHUNK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_chunks_have_required_fields(self):
        with open(CHUNK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for chunk in data[:10]: # Vérification des 10 premiers
            assert "page_content" in chunk
            assert "metadata" in chunk
            assert len(chunk["page_content"]) > 0
    
    def test_metadata_has_uid(self):
        with open(CHUNK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for chunk in data[:10]:
            assert "uid" in chunk["metadata"]

    def test_more_chunks_than_events(self):
        with open(CHUNK_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        csv_path = DATA_DIR / "events_drome.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            assert len(chunks) >= len(df)

    def test_no_nan_in_metadata(self):
        with open(CHUNK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for chunk in data:
            for key, value in chunk["metadata"].items():
                assert value is not None, f"Valeur None trouvée dans metadata['{key}']"
                if isinstance(value, float):
                    assert not pd.isna(value), f"Nan trouvé dans metadata['{key}']"