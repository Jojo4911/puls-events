"""Tests unitaires et d'intégration pour l'API."""

# --- Imports ---
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app


# -- Fixture pour le test client ---
@pytest.fixture()
def client():
    with patch("api.main.RAGSystem") as mock_rag:
        mock_rag.return_value.ask.return_value = {
            "answer": "Il y a 3 événements",
            "sources": [
                {
                    "title": "Titre 1",
                    "location_name": "1 rue de la paix",
                    "location_city": "Valence",
                    "date_display": "1er janvier",
                    "url": "https://valence.com/event1"
                },
                {
                    "title": "Titre 2",
                    "location_name": "2 rue de la paix",
                    "location_city": "Valence",
                    "date_display": "2 janvier",
                    "url": "https://valence.com/event2"
                },
                {
                    "title": "Titre 3",
                    "location_name": "3 rue de la paix",
                    "location_city": "Valence",
                    "date_display": "3 janvier",
                    "url": "https://valence.com/event3"
                }
            ],
            "contexts": ["event1", "event2", "event3"],
        }
        with TestClient(app) as c:
            yield c


# ---Health Endpoint---
def test_get_health(client):
    response_health = client.get("/health")
    assert response_health.status_code == 200
    assert response_health.json()["status"] == "ok"


# ---Root Endpoint (with redirection)---
def test_read_root(client):
    response_read_root = client.get("/", follow_redirects=False)
    assert response_read_root.status_code == 307
    assert response_read_root.headers["location"] == "/docs"

# ---Ask Endpoint---
# Testing endpoint with a valid question
def test_ask_valid_mocked(client):
    response_ask_valid_mocked = client.post("/ask", json={"question": "Quels événements à Valence ?"})
    assert response_ask_valid_mocked.status_code == 200
    assert response_ask_valid_mocked.json().keys() == {'answer', 'sources', 'contexts'}
    assert response_ask_valid_mocked.json()['answer'] == "Il y a 3 événements"

# Testing endpoint with an empty question
def test_ask_empty_question(client):
    response_ask_empty_question = client.post("/ask", json={"question":""})
    assert response_ask_empty_question.status_code == 422
    assert "string_too_short" in str(response_ask_empty_question.json())

# Testing endpoint with an error
def test_ask_rag_error(client):
    app.state.rag_system.ask.side_effect = Exception("Erreur test")
    response_ask_rag_error = client.post("/ask", json={"question": "Quels événements à Valence ?"})
    assert response_ask_rag_error.status_code == 500
    assert "Erreur test" in str(response_ask_rag_error.json())
    app.state.rag_system.ask.side_effect = None