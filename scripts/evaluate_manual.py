"""
Ce module va permettre d'évaluer manuellement chaque question du fichier test_dataset.json.

    1. Chargement du jeu de test depuis data/test_dataset.json
    2. Boucle d'évaluation sur chaque question
    3. Affichage côte à côte des question | ground_truth | answer pour faire une classification manuelle
    4. Calcul des métriques
    5. Sauvegarde dans data/evaluation_results.json
"""

# --- Imports ---
import json
from src.rag_system import RAGSystem
from pathlib import Path

# --- Configuration ---
TEST_DATASET = Path(__file__).resolve().parent.parent / "data" / "test_dataset.json"
TARGET_DATASET = Path(__file__).resolve().parent.parent / "data" / "evaluation_results.json"
rag = RAGSystem()

# === 1. Chargement du jeu de test ===
with open(TEST_DATASET, 'r', encoding="utf8") as f:
    data = json.load(f)

# === 2. Boucle d'évaluation sur chaque question ===
categories = {"croisé": {"score": 0.0, "count": 0}, "lieu": {"score": 0.0, "count": 0}, "date": {"score": 0.0, "count": 0}, "type": {"score": 0.0, "count": 0}, "hors_périmètre": {"score": 0.0, "count": 0}}
note_modified = {1: 1.0, 2: 0.5, 3: 0.0} # transformation des notes pour avoir une moyenne supérieure pour une bonne appréciation
for i, item in enumerate(data, 1):
    item_result = rag.ask(item['question'])
    item['answer'] = item_result['answer']
    item['contexts'] = item_result['contexts']
    # === 3. Classification manuelle ===
    print(f"\n\n=== Question {i} === : {item['question']}")
    print(f"\n=== Réponse du RAG === : {item['answer']}")
    print(f"\n=== Réponse espérée (ground truth) === : {item['ground_truth']}")
    note = 0
    while (note < 1 or note > 3):
        try:
            note = int(input(f"\nVeuillez noter la réponse du RAG (1 pour correcte, 2 pour partielle, 3 pour incorrecte) : "))
        except ValueError:
            print("\nVeuillez saisir un chiffre entre 1 et 3.")
    item['verdict'] = note
    # === 4. Calcul des métriques ===
    categories[item['category']]["score"] += note_modified[note]
    categories[item['category']]["count"] += 1

print("\n=== Métriques ===\n")
for cat, values in categories.items():
    print(f"Catégorie {cat} : Il y a un score de {values['score']} sur {values['count']} questions.")
    if values["count"] == 0:
        pourcentage = 0
    else:
        pourcentage = (values["score"] / values["count"]) * 100

    print(f"Soit un score global de {pourcentage} %.\n")

# === 5. Sauvegarde ===

with open(TARGET_DATASET, 'w', encoding="utf8") as f:
    json.dump(obj=data, fp=f, ensure_ascii=False, indent=2)