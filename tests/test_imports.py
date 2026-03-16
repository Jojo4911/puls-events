"""Test des imports clés - Étape 1 : Validationd de l'environnement."""

def test_faiss():
    import faiss
    # Vérifie qu’on peut créer un index basique
    index = faiss.IndexFlatL2(128)
    assert index.d == 128
    print("faiss-cpu OK")

def test_langchain_faiss():
    from langchain_community.vectorstores import FAISS # Nouvelle version (ancienne : from langchain.vectorstores import FAISS)
    print("langchain FAISS vectorstores OK")

def test_huggingface_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings # Nouvelle version (ancienne : from langchain.embeddings import HugginFaceEmbeddings)
    print("HuggingFaceEmbeddings OK")

def test_mistral():
    from mistralai import Mistral # Nouvelle version (ancienne : from mistral import MistralClient)
    print("mistralai OK")

def test_langchain_mistral():
    from langchain_mistralai.chat_models import ChatMistralAI
    from langchain_mistralai.embeddings import MistralAIEmbeddings
    print("langchain-mistralai OK")

def test_langchain_google():
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("langchain-google-geain OK (fallback Gemini)")

def test_ragas():
    from ragas.metrics.collections import faithfulness, answer_relevancy
    print("ragas OK")

def test_fastapi():
    from fastapi import FastAPI
    app = FastAPI()
    assert app is not None
    print("fastapi OK")

if __name__ == "__main__":
    tests = [
        test_faiss,
        test_langchain_faiss,
        test_huggingface_embeddings,
        test_mistral,
        test_langchain_mistral,
        test_langchain_google,
        test_ragas,
        test_fastapi
    ]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"{test.__name__} FAILED: {e}")
            failed += 1

if __name__ == "__main__":
    print(f"\nResultat: {passed} passed, {failed} failed")