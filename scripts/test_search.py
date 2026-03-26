"""Test manuel de recherche sémantique sur l'index FAISS."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vectorstore import load_index, search

QUERIES = [
    "concert jazz Valence",
    "exposition gratuite",
    "spectacle enfants Montélimar",
    "festival été Drôme",
    "atelier peinture",
    "randonnée nature",
]


def main():
    vectorstore = load_index()
    print(f"Index chargé : {vectorstore.index.ntotal} vecteurs\n")

    for query in QUERIES:
        print(f"{'=' * 60}")
        print(f"REQUÊTE : {query}")
        print(f"{'=' * 60}")

        results = search(vectorstore, query, k=3)

        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            print(f"\n --- Résultats {i}---")
            print(f"  Titre : {meta.get('title', 'N/A')}")
            print(f"  Lieu  : {meta.get('location_name', 'N/A')} — {meta.get('location_city', 'N/A')}")
            print(f"  Dates : {meta.get('first_date', '?')} → {meta.get('last_date', '?')}")
            print(f" Affichage dates : {meta.get('date_display', 'N/A')}")
            print(f"  Texte : {doc.page_content[:150]}...")

        print()


if __name__ == "__main__":
    main()