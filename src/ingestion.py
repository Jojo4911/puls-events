"""
Ingestion des événements culturels depuis l'API Open Agenda (OpenDataSoft).

Ce module récupère les événements du département de la Drôme sur les 12 derniers mois, nettoie les données et les structure pour la vectorisation.
"""

# --- Imports ---
import json
import logging
import re
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path

import pandas as pd
import requests

# --- Configuration ---
BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/evenements-publics-openagenda/records"
)
DEPARTEMENT = "Drôme"
DAYS_HISTORY = 365
PAGE_SIZE = 100 # Maximum autorisé par l’API OpenDataSoft

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FIELDS_TO_KEEP = [
    "uid",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "keywords_fr",
    "daterange_fr",
    "firstdate_begin",
    "lastdate_begin",
    "location_name",
    "location_city",
    "location_address",
    "location_coordinates",
    "canonicalurl",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Collecte ---


def build_where_clause(department: str, days_history: int) -> str:
    """Construit la clause WHERE pour l'API OpenDataSoft."""
    date_limit = (datetime.now() - timedelta(days=days_history)).strftime("%Y-%m-%d")
    return f'location_department="{department}" AND lastdate_begin>="{date_limit}"'


def fetch_events(department: str = DEPARTEMENT, days_history: int = DAYS_HISTORY, limit: int | None = None) -> list[dict]:
    """
    Récupère tous les évènements depuis l'API avec pagination.

    returns:
        Liste de dictionnaires représentant chaque évenement brut.
    """
    where_clause = build_where_clause(department, days_history)
    all_records = []
    offset = 0

    logger.info("Début de la collecte — département: %s, historique: %d jours", department, days_history)

    while True:
        params = {
            "where": where_clause,
            "limit": PAGE_SIZE,
            "offset": offset,
            "order_by": "lastdate_begin DESC",
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Erreur lors de l'appel API (offest=%d): %s", offset, e)
            break

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        all_records.extend(results)

        if (limit is not None) and (len(all_records) >= limit):
            break

        offset += PAGE_SIZE

        total = data.get("total_count", "?")
        logger.info("Récupérés: %d / %s événements", len(all_records), total)

        # Sécurité : arrêter si on a tout récupéré
        if isinstance(total, int) and len(all_records) >= total:
            break

    logger.info("Collecte terminée: %d évènements récupérés", len(all_records))
    all_records = all_records[:limit]
    return all_records


# --- Nettoyage ---


def clean_html(text: str | None) -> str:
    """Supprime les balises HTML et décode les entités."""
    if not text:
        return ""
    # Retire les balises HTML
    clean = re.sub(r"<[^>]+>", " ", text)
    # Décode les entités HTML (&amp; -> &, etc.)
    clean = unescape(clean)
    # Normalise les espaces multiples
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def format_keywords(keywords: list | None) -> str:
    """Convertit la liste de mots-clés en chaîne séparée avec des virgules."""
    if not keywords:
        return ""
    return ", ".join(keywords)


def select_fields(record: dict) -> dict:
    """Extrait et nettoie les champs pertinents d'un enregistrement brut."""
    return {
        "uid": record.get("uid", ""),
        "title": record.get("title_fr", ""),
        "description": clean_html(record.get("description_fr")),
        "long_description": clean_html(record.get("longdescription_fr")),
        "keywords": format_keywords(record.get("keywords_fr")),
        "date_display": record.get("daterange_fr", ""),
        "first_date": record.get("firstdate_begin", ""),
        "last_date": record.get("lastdate_begin", ""),
        "location_name": record.get("location_name", ""),
        "location_city": record.get("location_city", ""),
        "location_address": record.get("location_address", ""),
        "latitude": record.get("location_coordinates", {}).get("lat") if record.get("location_coordinates") else None,
        "longitude": record.get("location_coordinates", {}).get("lon") if record.get("location_coordinates") else None,
        "url": record.get("canonicalurl", ""),
    }


def build_text_for_embedding(row: dict) -> str:
    """
    Construit le texte structuré qui sera vectorisé.
    
    Format:
        Titre: ...
        Date: ...
        Lieu: ...
        Description: ...
        Mots-clés: ...
    """
    parts = []

    if row.get("title"):
        parts.append(f"Titre : {row['title']}")

    if row.get("date_display"):
        parts.append(f"Date : {row['date_display']}")

    location_parts = [p for p in [row.get("location_name"), row.get("location_address")] if p]
    if location_parts:
        parts.append(f"Lieu : {', '.join(location_parts)}")

    # Préférer la description longue si elle existe, sinon la courte
    description = row.get("long_description") or row.get("description") or ""
    if description:
        parts.append(f"Description : {description}")

    if row.get("keywords"):
        parts.append(f"Mots-clés : {row['keywords']}")

    return "\n".join(parts)

def clean_events(raw_events: list[dict]) -> pd.DataFrame:
    """Nettoie et structure les évènements bruts en DataFrame.
    
    Returns:
        DataFrame avec les champs nettoyés et le reste pour l'embedding.
    """
    logger.info("Nettoyage de %d événements...", len(raw_events))

    # Extraction et nettoyage des champs
    cleaned = [select_fields(record) for record in raw_events]
    df = pd.DataFrame(cleaned)

    # Suppression des doublons par UID
    initial_count = len(df)
    df = df.drop_duplicates(subset="uid", keep="first")
    duplicates = initial_count - len(df)
    if duplicates:
        logger.info("Supprimé %d doublons", duplicates)

    # Suppression des événements sans titre ni description
    df = df[
        (df["title"].str.len() > 0) | (df["description"].str.len() > 0)
    ].copy()

    # Construction du texte pour l’embedding
    df["text_for_embedding"] = df.apply(
        lambda row: build_text_for_embedding(row.to_dict()), axis=1
    )

    # Suppression des lignes avec un texte trop court (< 30 caractères)
    df = df[df["text_for_embedding"].str.len() >= 30].copy()

    df = df.reset_index(drop=True)
    logger.info("Nettoyage terminé: %d événements retenus", len(df))
    return df


# --- Analyse des longueurs de textes pour l’embedding ---


def analyze_text_lengths(df: pd.DataFrame) -> None:
    """Affiche les statistiques de longueur du texte pour embedding."""
    lengths = df["text_for_embedding"].str.len()

    logger.info("=== Distribution des longueurs du texte pour embedding ===")
    logger.info("Min: %d | Max: %d | Moyenne: %.0f | Médiane: %.0f",
                lengths.min(), lengths.max(), lengths.mean(), lengths.median())

    bins = [0, 200, 500, 1000, 2000, 5000, float("inf")]
    labels = ["< 200", "200-500", "500-1000", "1000-2000", "2000-5000", "> 5000"]
    counts = pd.cut(lengths, bins=bins, labels=labels).value_counts().sort_index()
    for label, count in counts.items():
        pct = count / len(df) * 100
        logger.info("  %10s : %4d (%5.1f%%)", label, count, pct)


# --- Sauvegarde ---


def save_events(df: pd.DataFrame, filename: str = "events_drome") -> tuple[Path, Path]:
    """
    Sauvegarde le DataFrame en JSON et CSV dans le dossier data/.

    returns:
        Tuple des chemins (json_path, csv_path)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / f"{filename}.json"
    csv_path = DATA_DIR / f"{filename}.csv"

    # JSON : lisible, avec les méthadonnées
    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    # CSV : pratique pour l’exploration avec Pandas
    df.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info("Données sauvegardées: %s et %s", json_path, csv_path)
    return json_path, csv_path


# --- Point d’entrée ---


def main():
    """Pipeline complet : Collecte -> Nettoyage -> Sauvegarde."""
    # 1. Collecte
    raw_events = fetch_events()

    if not raw_events:
        logger.error("Aucun événement récupéré. Vérifiez la connexion et les filtres.")
        return
    
    # 2. Nettoyage et structuration
    df = clean_events(raw_events)

    # 3. Analyse de la longueur des textes générés pour l’embedding
    analyze_text_lengths(df)
    
    # 4. Sauvegarde
    save_events(df)

    # 5. Résumé
    logger.info("--- Résumé ---")
    logger.info("Événements bruts récupérés : %d", len(raw_events))
    logger.info("Événements nettoyés retenus : %d", len(df))
    logger.info("Villes représentées : %d", df["location_city"].nunique())
    logger.info("Longueur moyenne du texte pour embedding : %.0f caractères", df["text_for_embedding"].str.len().mean())


if __name__ == "__main__":
    main()