"""
Ce module va permettre d'évaluer automatiquement avec Ragas chaque question du fichier test_dataset.json.

1. Chargement des résultats
2. Formatage pour Ragas
3. Configuration du LLM et des embeddings pour Ragas
4. Lancement de l'évaluation
5. Affichage et sauvegarde des résultats
"""

# --- Imports ---
from pathlib import Path
import json
from datasets import Dataset
from src.llm import get_llm
from src.vectorstore import get_embeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
import pandas as pd

# --- Configuration ---
RESULTS_DATASET = Path(__file__).resolve().parent.parent / "data" / "evaluation_results.json"
RAGAS_RESULTS = Path(__file__).resolve().parent.parent / "data" / "ragas_results.json"

def load_and_format(results: Path) -> Dataset:
    """
    Préparation des données

    Args:
        results: Chemin du fichier JSON "evaluation_results.json".
    
    Returns:
        Dataset d'évaluation des données.
    """
    # === 1. Chargement des résultats ===
    with open(results, 'r', encoding="utf8") as f:
        data = json.load(f)

    # === 2. Formatage pour Ragas ===
    questions = [item['question']for item in data]
    answers = [item['answer']for item in data]
    contexts = [item['contexts']for item in data]
    ground_truths = [item['ground_truth']for item in data]

    evaluation_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
    }

    evaluation_dataset = Dataset.from_dict(evaluation_data)

    return evaluation_dataset

def run_evaluation(eval_dataset: Dataset) -> dict:
    """
    Lancement de l'évaluation Ragas et affichage des résultats

    Args:
        eval_dataset: Dataset d'évaluation des données, créé par la fonction load_and_format().

    Returns:
        average_scores: Dictionnaire des scores moyens de l'évaluation sur tout le dataset.
    """
    # === 3. Configuration du LLM et des embeddings pour Ragas ===
    llm = get_llm()
    embeddings = get_embeddings("SEMANTIC_SIMILARITY")

    # === 4. Lancement de l'évaluation ===
    metrics_to_evaluate = [
        faithfulness,       # Génération: fidèle au contexte ?
        answer_relevancy,   # Génération: réponse pertinente à la question ?
        context_precision,  # Récupération: contexte précis (peu de bruit) ?
        context_recall,     # Récupération: infos clés récupérées (nécessite ground_truth) ?
    ]

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics_to_evaluate,
        llm=llm,                 # LLM pour juger certaines métriques
        embeddings=embeddings,   # Embeddings pour juger d'autres métriques
    )

    # === 5. Affichage des résultats ===
    # Transformation des résultats en DataFrame Pandas
    results_df = results.to_pandas()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 150) # Ajustez si nécessaire
    print(results_df)

    # Calcul et affichage des scores moyens
    print("\n--- Scores Moyens (sur tout le dataset) ---")
    average_scores = results_df.mean(numeric_only=True).to_dict()
    print(average_scores)

    return average_scores


if __name__ == "__main__":

    try:
        scores = run_evaluation(load_and_format(RESULTS_DATASET))
        
        # Sauvegarde des résultats
        with open(RAGAS_RESULTS, 'w', encoding="utf8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

        print("Sauvegarde réussie des résultats !")

    except Exception as e:
        print(f"Erreur lors de l'évaluation Ragas : {e}")

        print("Sauvegarde impossible")