"""Tests unitaires pour src/llm.py."""

import pytest
from unittest.mock import patch, MagicMock


# --- Tests get_llm ---


class TestGetLlm:
    """Tests pour la fonction get_llm()."""

    @patch.dict("os.environ", {"LLM_PROVIDER": "google", "GOOGLE_API_KEY": "fake-key"})
    @patch("src.llm.ChatGoogleGenerativeAI", create=True)
    def test_google_provider(self, mock_chat):
        """get_llm() avec provider google retourne un ChatGoogleGenerativeAI."""
        from src.llm import get_llm

        # Le lazy import dans get_llm() va chercher ChatGoogleGenerativeAI
        # dans le module — on doit patcher au bon endroit
        with patch("src.llm.ChatGoogleGenerativeAI", mock_chat):
            pass

        # Alternative plus simple : on re-importe en patchant le provider
        import src.llm as llm_module

        with patch.object(llm_module, "LLM_PROVIDER", "google"):
            with patch(
                "langchain_google_genai.ChatGoogleGenerativeAI"
            ) as mock_google:
                mock_google.return_value = MagicMock()
                result = llm_module.get_llm(temperature=0.3)
                mock_google.assert_called_once_with(
                    model="gemini-3.1-flash-lite-preview",
                    google_api_key="fake-key",
                    temperature=0.3,
                )

    @patch.dict("os.environ", {"LLM_PROVIDER": "mistral", "MISTRAL_API_KEY": "fake-key"})
    def test_mistral_provider(self):
        """get_llm() avec provider mistral retourne un ChatMistralAI."""
        import src.llm as llm_module

        with patch.object(llm_module, "LLM_PROVIDER", "mistral"):
            with patch("langchain_mistralai.ChatMistralAI") as mock_mistral:
                mock_mistral.return_value = MagicMock()
                result = llm_module.get_llm()
                mock_mistral.assert_called_once_with(
                    model="mistral-large-latest",
                    api_key="fake-key",
                    temperature=0.2,
                )

    @patch.dict("os.environ", {"LLM_PROVIDER": "google", "GOOGLE_API_KEY": ""}, clear=False)
    def test_missing_google_api_key(self):
        """get_llm() lève EnvironmentError si GOOGLE_API_KEY est vide."""
        import src.llm as llm_module

        with patch.object(llm_module, "LLM_PROVIDER", "google"):
            with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
                llm_module.get_llm()

    @patch.dict("os.environ", {"LLM_PROVIDER": "mistral", "MISTRAL_API_KEY": ""}, clear=False)
    def test_missing_mistral_api_key(self):
        """get_llm() lève EnvironmentError si MISTRAL_API_KEY est vide."""
        import src.llm as llm_module

        with patch.object(llm_module, "LLM_PROVIDER", "mistral"):
            with pytest.raises(EnvironmentError, match="MISTRAL_API_KEY"):
                llm_module.get_llm()

    def test_unknown_provider(self):
        """get_llm() lève ValueError si le provider est inconnu."""
        import src.llm as llm_module

        with patch.object(llm_module, "LLM_PROVIDER", "openai"):
            with pytest.raises(ValueError, match="LLM_PROVIDER inconnu"):
                llm_module.get_llm()

    def test_default_temperature(self):
        """La température par défaut est 0.2."""
        import src.llm as llm_module

        with patch.object(llm_module, "LLM_PROVIDER", "google"):
            with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}):
                with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_google:
                    mock_google.return_value = MagicMock()
                    llm_module.get_llm()
                    _, kwargs = mock_google.call_args
                    assert kwargs["temperature"] == 0.2


# --- Tests extract_text ---


class TestExtractText:
    """Tests pour la fonction extract_text()."""

    def test_string_content(self):
        """extract_text() retourne le texte tel quel si c'est un str (Mistral)."""
        from src.llm import extract_text

        response = MagicMock()
        response.content = "Voici un événement à Valence."
        assert extract_text(response) == "Voici un événement à Valence."

    def test_list_content_gemini(self):
        """extract_text() extrait le texte des blocs Gemini 3.x."""
        from src.llm import extract_text

        response = MagicMock()
        response.content = [
            {
                "type": "text",
                "text": "Le festival Jazz à Crest a lieu chaque été.",
                "extras": {"signature": "abc123"},
            }
        ]
        assert extract_text(response) == "Le festival Jazz à Crest a lieu chaque été."

    def test_list_multiple_blocks(self):
        """extract_text() concatène plusieurs blocs texte."""
        from src.llm import extract_text

        response = MagicMock()
        response.content = [
            {"type": "text", "text": "Première partie."},
            {"type": "text", "text": "Deuxième partie."},
        ]
        assert extract_text(response) == "Première partie.\nDeuxième partie."

    def test_list_filters_non_text(self):
        """extract_text() ignore les blocs qui ne sont pas de type 'text'."""
        from src.llm import extract_text

        response = MagicMock()
        response.content = [
            {"type": "text", "text": "Texte utile."},
            {"type": "other", "data": "ignoré"},
        ]
        assert extract_text(response) == "Texte utile."

    def test_fallback_to_str(self):
        """extract_text() convertit en str si le format est inattendu."""
        from src.llm import extract_text

        response = MagicMock()
        response.content = 42
        assert extract_text(response) == "42"