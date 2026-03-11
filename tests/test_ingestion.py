"""
Tests unitaires pour le pipeline d'ingestion (src/ingestion.py).

Couvre : nettoyage HTML, extraction de champs, construction du texte
pour embedding, nettoyage global du DataFrame, et validation des données
sauvegardées.
"""

# Imports
import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion import (
    build_text_for_embedding,
    build_where_clause,
    clean_events,
    clean_html,
    format_keywords,
    select_fields,
)

# --- Chemin vers les données sauvegardées ---

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JSON_PATH = DATA_DIR / "events_drome.json"
CSV_PATH = DATA_DIR / "events_drome.csv"


# =============================================================================
# Tests unitaires — fonctions de nettoyage
# =============================================================================


class TestCleanHtml:
    """Tests pour la fonction clean_html"""

    def test_removes_html_tags(self):
        assert clean_html("<p>Bonjour<p>") == "Bonjour"
    
    def test_removes_br_tags(self):
        assert clean_html("Ligne 1<br>Ligne 2") == "Ligne 1 Ligne 2"

    def test_decodes_html_entities(self):
        assert clean_html("Caf&eacute; &amp; th&eacute;") == "Café & thé"

    def test_normalize_whitespace(self):
        assert clean_html("<p>   Bonjour  tout le  monde     <p>") == "Bonjour tout le monde"

    def test_returns_empty_string_for_none(self):
        assert clean_html(None) == ""
    
    def test_returns_empty_string_for_empty(self):
        assert clean_html("") == ""
    
    def test_complex_html(self):
        html = "<p>Venez <strong>nombreux</strong> à la <em>fête</em>!</p>\n<p>Entrée libre.</p>"
        result = clean_html(html)
        assert "Venez" in result
        assert "nombreux" in result
        assert "Entrée libre" in result
        assert "<" not in result

class TestFormatKeywords:
    """Tests pour la fonction format_keywords"""

    def test_formats_list(self):
        assert format_keywords(["Jazz", "Concert"]) == "Jazz, Concert"

    def test_returns_empty_for_none(self):
        assert format_keywords(None) == ""

    def test_returns_empty_for_empty_list(self):
        assert format_keywords([]) == ""

    def test_single_keyword(self):
        assert format_keywords(["Jazz"]) == "Jazz"


# =============================================================================
# Tests unitaires — extraction et structuration
# =============================================================================


# Fixture : Enregistrement brut typique de l’API
@pytest.fixture
def raw_record():
    return {
        "uid": "12345",
        "title_fr": "Festival de Jazz",
        "description_fr": "Un super festival",
        "longdescription_fr": "<p>Un super festival de <strong>jazz</strong> en plein air.</p>",
        "keywords_fr": ["Jazz", "Musique", "Festival"],
        "daterange_fr": "Samedi 15 juin, 20h00",
        "firstdate_begin": "2025-06-15T18:00:00+00:00",
        "lastdate_begin": "2025-06-15T18:00:00+00:00",
        "location_name": "Parc de la Mairie",
        "location_city": "Valence",
        "location_address": "26000 Valence",
        "location_coordinates": {"lat": 44.9334, "lon": 4.8924},
        "canonicalurl": "https://openagenda.com/event/12345",
    }


class TestSelectedFields:
    """Tests pour la fonction select_fields."""

    def test_extracts_all_fields(self, raw_record):
        result = select_fields(raw_record)
        assert result["uid"] == "12345"
        assert result["title"] == "Festival de Jazz"
        assert result["location_city"] == "Valence"
        assert result["latitude"] == 44.9334
        assert result["longitude"] == 4.8924
        assert result["url"] == "https://openagenda.com/event/12345"

    def test_cleans_html_in_long_description(self, raw_record):
        result = select_fields(raw_record)
        assert result["long_description"] == "Un super festival de jazz en plein air."

    def test_handles_missing_coordinates(self):
        record = {"uid": "99", "location_coordinates": None}
        result = select_fields(record)
        assert result["latitude"] == None
        assert result["longitude"] == None
    
    def test_handles_empty_record(self):
        record = {}
        result = select_fields(record)
        assert result["uid"] == ""
        assert result["title"] == ""
        assert result["latitude"] == None


class TestBuildTextForEmbedding:
    """Tests pour la fonction build_texte_for_embedding."""

    def test_contains_all_parts(self):
        row = {
            "title": "Concert Rock",
            "date_display": "Vendredi 20 juin, 21h00",
            "location_name": "Salle des fêtes",
            "location_address": "26000 Valence",
            "long_description": "Un concert exceptionnel.",
            "description": "Concert de rock.",
            "keywords": "Rock, Musique",
        }
        text = build_text_for_embedding(row)
        assert "Titre : Concert Rock" in text
        assert "Date : Vendredi 20 juin, 21h00" in text
        assert "Lieu : Salle des fêtes, 26000 Valence" in text
        assert "Description : Un concert exceptionnel." in text
        assert "Mots-clés : Rock, Musique" in text

    def test_prefers_long_description(self):
        row = {
            "title": "Test",
            "long_description": "",
            "description": "Description courte.",
        }
        text = build_text_for_embedding(row)
        assert "Description courte" in text

    def test_falls_back_to_short_description(self):
        row = {
            "title": "Test",
            "long_description": "",
            "description": "Description courte.",
        }
        text = build_text_for_embedding(row)
        assert "Description : Description courte." in text

    def test_handles_minimal_data(self):
        row = {"title": "Événement minimal"}
        text = build_text_for_embedding(row)
        assert "Titre : Événement minimal" in text

    
# =============================================================================
# Tests unitaires — pipeline de nettoyage global
# =============================================================================


class TestCleanEvents:
    """Tests pour la fonction clean_events"""

    def test_removes_duplicates(self):
        events = [
            {"uid": "1", "title_fr": "Event A", "description_fr": "Desc A"},
            {"uid": "1", "title_fr": "Event A", "description_fr": "Desc A"},
            {"uid": "2", "title_fr": "Event B", "description_fr": "Desc B"},
        ]
        df = clean_events(events)
        assert len(df) == 2
        assert df["uid"].nunique() == 2

    def test_removes_empty_events(self):
        events = [
            {"uid": "1", "title_fr": "", "description_fr": ""},
            {"uid": "2", "title_fr": "Event B", "description_fr": "Desc B"}
        ]
        df = clean_events(events)
        assert len(df) == 1
        assert df.iloc[0]["uid"] == "2"
    
    def test_creates_text_for_embedding_column(self):
        events = [
            {"uid": "1", "title_fr": "Festival", "description_fr": "Un beau festival d'été en plein air"}
        ]
        df = clean_events(events)
        assert "text_for_embedding" in df.columns
        assert len(df.iloc[0]["text_for_embedding"]) >= 30

    def test_filters_short_texts(self):
        events = [
            {"uid": "1", "title_fr": "OK", "description_fr": ""},  # trop court après build
            {"uid": "2", "title_fr": "Grand festival de musique", "description_fr": "Description suffisamment longue pour passer"},
        ]
        df = clean_events(events)
        assert len(df) == 1
        assert all(df["text_for_embedding"].str.len() >= 30)


# =============================================================================
# Tests unitaires — clause WHERE pour l'API
# =============================================================================


class TestBuildWhereClause:
    """Tests pour la fonction build_where_clause"""

    def test_contains_department(self):
        clause = build_where_clause("Drôme", 365)
        assert 'location_department="Drôme"' in clause
    
    def test_contains_date_filter(self):
        clause = build_where_clause("Drôme", 365)
        assert "lastdate_begin>=" in clause

    def test_custom_department(self):
        clause = build_where_clause("Isère", 365)
        assert 'location_department="Isère"' in clause


# =============================================================================
# Tests d'intégration — validation des données sauvegardées
# =============================================================================


class TestSavedData:
    """Validation des fichiers de données générés par le pipeline."""

    @pytest.fixture(autouse=True)
    def check_data_exists(self):
        """Skip ces test si les données n'ont pas encore été générées."""
        if not JSON_PATH.exists():
            pytest.skip("Données non générées — lancer d'abord: uv run python src/ingestion.py")

    def test_json_file_is_valid(self):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_csv_file_is_valid(self):
        df = pd.read_csv(CSV_PATH)
        assert len(df) > 0

    def test_no_duplicate_uids(self):
        df = pd.read_csv(CSV_PATH)
        assert df["uid"].is_unique
    
    def test_required_columns_present(self):
        df = pd.read_csv(CSV_PATH)
        expected_columns = [
            "uid", "title", "description", "long_description",
            "keywords", "date_display", "first_date", "last_date",
            "location_name", "location_city", "location_address",
            "text_for_embedding", "url",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Colonne manquante : {col}"

    def test_text_for_embedding_not_empty(self):
        df = pd.read_csv(CSV_PATH)
        assert df["text_for_embedding"].notna().all()
        assert (df["text_for_embedding"].str.len() >= 30).all()

    def test_minimum_event_count(self):
        df = pd.read_csv(CSV_PATH)
        assert len(df) >= 100, f"Seulement {len(df)} événements - attendu au moins 100"

    def test_json_csv_consistency(self):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        df = pd.read_csv(CSV_PATH)
        assert len(json_data) == len(df), "JSON et CSV ont un nombre différent d'enregistrements"