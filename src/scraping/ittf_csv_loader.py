"""
Loader pour les CSV du repo romanzdk/ittf-data-scrape.

Contenu des fichiers :
  ittf.csv             → snapshots hebdomadaires du classement ITTF (2001→)
                         colonnes : index, Rank, Previous, ID, Assoc, Gender,
                                    Name, Points, Previous Points,
                                    WeekNum, MonthNum, YearNum
  ittf_player_info.csv → infos biographiques des joueurs
                         colonnes : Player ID, Name, Assoc, Gender, Birth year,
                                    Activity, Playing hand, Playing style, Grip
  ittf_rankings.csv    → idem ittf.csv sans colonne index (hommes)
  ittf_rankings_women.csv → idem, femmes

Ces fichiers se téléchargent une seule fois (~38 MB total) et sont importés en DB.
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from sqlalchemy.orm import Session

from ..database.db import get_session
from ..database.models import IttfRanking, Player

RAW_BASE = "https://raw.githubusercontent.com/romanzdk/ittf-data-scrape/master/data"

URLS = {
    "ittf_rankings_men": f"{RAW_BASE}/ittf_rankings.csv",
    "ittf_rankings_women": f"{RAW_BASE}/ittf_rankings_women.csv",
    "ittf_player_info": f"{RAW_BASE}/ittf_player_info.csv",
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_csv(name: str, force: bool = False) -> Path:
    """Télécharge un CSV si absent (ou si force=True)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / f"{name}.csv"

    if dest.exists() and not force:
        logger.info(f"{name}.csv déjà présent ({dest.stat().st_size // 1024} KB)")
        return dest

    url = URLS[name]
    logger.info(f"Téléchargement {name} depuis {url}...")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    total = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
            total += len(chunk)

    logger.info(f"{name}.csv téléchargé ({total // 1024} KB)")
    return dest


def _week_to_date(year: int, month: int, week: int) -> datetime:
    """Convertit (year, month, week) en date approximative."""
    try:
        # Approximation : première semaine du mois = jour 1
        day = max(1, (week % 4) * 7 + 1) if week > 0 else 1
        day = min(day, 28)
        return datetime(int(year), int(month), day)
    except (ValueError, TypeError):
        return datetime(int(year), 1, 1)


def load_rankings_into_db(gender: str = "M", force_download: bool = False) -> int:
    """
    Importe les snapshots de rankings ITTF en DB via insert en masse.

    Args:
        gender: "M" ou "F"
        force_download: re-télécharge même si le fichier existe

    Returns:
        Nombre de snapshots insérés
    """
    from sqlalchemy import text as sa_text
    from src.database.db import engine

    name = "ittf_rankings_men" if gender == "M" else "ittf_rankings_women"
    csv_path = download_csv(name, force=force_download)

    logger.info(f"Lecture {csv_path.name}...")
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Rank": "rank", "ID": "ittf_id", "Assoc": "country",
        "Gender": "gender_raw", "Name": "name", "Points": "points",
        "WeekNum": "week", "MonthNum": "month", "YearNum": "year",
    })

    df = df.dropna(subset=["rank", "ittf_id", "year"])
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    df = df.dropna(subset=["rank"])
    df["ittf_id"] = df["ittf_id"].astype(str).str.strip()

    logger.info(f"{len(df)} lignes à traiter...")

    # Étape 1 : récupère la map ittf_id → player_id depuis la DB
    with engine.connect() as conn:
        existing_players = pd.read_sql(
            sa_text("SELECT id, ittf_id FROM players WHERE ittf_id IS NOT NULL"),
            conn,
        )
    id_map = dict(zip(existing_players["ittf_id"].astype(str), existing_players["id"]))
    logger.info(f"  {len(id_map)} joueurs connus en DB")

    # Joueurs manquants → insertion en bloc
    missing_ids = set(df["ittf_id"].unique()) - set(id_map.keys())
    if missing_ids:
        df_missing = (
            df[df["ittf_id"].isin(missing_ids)][["ittf_id", "name", "country", "gender_raw"]]
            .drop_duplicates("ittf_id")
            .copy()
        )
        df_missing["gender"] = gender
        df_missing = df_missing.rename(columns={"ittf_id": "ittf_id", "country": "country"})
        with engine.connect() as conn:
            for _, row in df_missing.iterrows():
                conn.execute(sa_text(
                    "INSERT OR IGNORE INTO players (name, country, gender, ittf_id) "
                    "VALUES (:name, :country, :gender, :ittf_id)"
                ), {"name": str(row["name"]), "country": str(row.get("country", "")),
                    "gender": gender, "ittf_id": str(row["ittf_id"])})
            conn.commit()
        # Recharge la map
        with engine.connect() as conn:
            existing_players = pd.read_sql(
                sa_text("SELECT id, ittf_id FROM players WHERE ittf_id IS NOT NULL"), conn
            )
        id_map = dict(zip(existing_players["ittf_id"].astype(str), existing_players["id"]))
        logger.info(f"  {len(missing_ids)} nouveaux joueurs insérés")

    # Étape 2 : construit le DataFrame des snapshots
    df["player_id"] = df["ittf_id"].map(id_map)
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(int)
    df["snapshot_date"] = df.apply(
        lambda r: _week_to_date(r["year"], r["month"], r["week"]), axis=1
    )

    # Déduplique en mémoire (même player_id + snapshot_date)
    df = df.drop_duplicates(subset=["player_id", "snapshot_date"])

    # Étape 3 : INSERT OR IGNORE en blocs de 5000
    CHUNK = 5_000
    inserted = 0
    df["snapshot_date"] = df["snapshot_date"].astype(str)
    rows = df[["player_id", "rank", "points", "snapshot_date"]].to_dict("records")

    with engine.connect() as conn:
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i: i + CHUNK]
            conn.execute(
                sa_text(
                    "INSERT OR IGNORE INTO ittf_rankings "
                    "(player_id, rank, points, snapshot_date) "
                    "VALUES (:player_id, :rank, :points, :snapshot_date)"
                ),
                chunk,
            )
            conn.commit()
            inserted += len(chunk)
            logger.info(f"  {min(inserted, len(rows))}/{len(rows)} snapshots traités...")

    logger.info(f"Rankings {gender} : {inserted} nouveaux snapshots insérés")
    return inserted


def load_player_info_into_db(force_download: bool = False) -> int:
    """
    Enrichit la table players avec les infos biographiques ITTF.

    Colonnes utiles : Birth year, Playing hand, Playing style, Grip
    Ces infos deviennent des features (âge exact, main, style de jeu).
    """
    csv_path = download_csv("ittf_player_info", force=force_download)

    logger.info("Lecture ittf_player_info.csv...")
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Player ID": "ittf_id",
        "Name": "name",
        "Assoc": "country",
        "Gender": "gender_raw",
        "Birth year": "birth_year",
        "Activity": "activity",
        "Playing hand": "playing_hand",
        "Playing style": "playing_style",
        "Grip": "grip",
    })
    df = df.dropna(subset=["ittf_id", "name"])

    updated = 0
    with get_session() as session:
        for _, row in df.iterrows():
            try:
                player = (
                    session.query(Player)
                    .filter_by(ittf_id=str(row["ittf_id"]))
                    .first()
                )
                if not player:
                    # Crée le joueur même s'il n'a pas encore de matchs
                    gender = "M" if str(row.get("gender_raw", "")).lower() in ("m", "male") else "F"
                    player = Player(
                        name=str(row["name"]),
                        country=str(row.get("country", "")),
                        gender=gender,
                        ittf_id=str(row["ittf_id"]),
                    )
                    session.add(player)
                    session.flush()

                # Enrichit avec les infos bio
                if row.get("birth_year") and str(row["birth_year"]).strip().isdigit():
                    year = int(row["birth_year"])
                    if 1940 <= year <= 2010:
                        player.date_of_birth = datetime(year, 1, 1)

                updated += 1
                if updated % 5_000 == 0:
                    session.flush()
                    logger.info(f"  {updated} joueurs mis à jour...")

            except Exception as e:
                logger.debug(f"Erreur player info : {e}")

    logger.info(f"Player info : {updated} joueurs enrichis")
    return updated


def _get_or_create_player_by_ittf_id(
    session: Session, ittf_id: str, name: str, country: str, gender: str
) -> Player:
    player = session.query(Player).filter_by(ittf_id=ittf_id).first()
    if not player:
        player = session.query(Player).filter_by(name=name).first()
    if not player:
        player = Player(
            name=name,
            country=country,
            gender=gender,
            ittf_id=ittf_id,
        )
        session.add(player)
        session.flush()
    elif not player.ittf_id:
        player.ittf_id = ittf_id
    return player
