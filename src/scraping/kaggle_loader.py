"""
Loader pour le dataset Kaggle Setka Cup (medaxone).
https://www.kaggle.com/datasets/medaxone/one-month-table-tennis-dataset

Le fichier doit être téléchargé manuellement et placé dans :
  data/raw/kaggle_setka_2022.csv

Le dataset contient 1 mois de matchs Setka Cup (juin-juillet 2022).
Il sert de données de bootstrap et de test pour valider le pipeline.

Note : les colonnes exactes seront déterminées à l'ouverture du fichier.
       Le loader détecte automatiquement la structure et adapte le mapping.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from ..database.db import get_session
from ..database.models import Competition, Match, Player

KAGGLE_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "kaggle_setka_2022.csv"
COMPETITION_ID = "setka_cup"


def load_kaggle_setka(file_path: Path | None = None) -> int:
    """
    Importe le dataset Kaggle Setka Cup en DB.

    Returns:
        Nombre de matchs insérés, ou -1 si fichier absent.
    """
    path = file_path or KAGGLE_FILE
    if not path.exists():
        logger.warning(f"Fichier Kaggle absent : {path}")
        return -1

    logger.info(f"Lecture {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"Colonnes détectées : {list(df.columns)}")
    logger.info(f"Lignes : {len(df)}")

    # Mapping flexible — adapte selon les colonnes réelles du CSV
    df = _normalize_columns(df)
    if df is None:
        logger.error("Impossible de normaliser les colonnes du CSV Kaggle")
        return 0

    inserted = 0
    with get_session() as session:
        comp = _get_or_create_competition(session)

        for _, row in df.iterrows():
            try:
                p1 = _get_or_create_player(session, str(row["player1_name"]))
                p2 = _get_or_create_player(session, str(row["player2_name"]))

                # Déduplique
                existing = (
                    session.query(Match)
                    .filter_by(
                        player1_id=p1.id,
                        player2_id=p2.id,
                        played_at=row["played_at"],
                    )
                    .first()
                )
                if existing:
                    continue

                session.add(Match(
                    external_id=f"kaggle_{row.get('match_id', inserted)}",
                    competition_id=comp.id,
                    player1_id=p1.id,
                    player2_id=p2.id,
                    played_at=row["played_at"],
                    winner=int(row["winner"]),
                    score_p1=int(row.get("score_p1", 0)),
                    score_p2=int(row.get("score_p2", 0)),
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Erreur ligne Kaggle : {e}")

    logger.info(f"Kaggle Setka : {inserted} matchs insérés")
    return inserted


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Tente de mapper les colonnes du CSV vers la structure attendue.
    Affiche les colonnes disponibles pour aider à l'ajustement manuel.
    """
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # Mapping courants pour les datasets TT
    mappings = [
        # (pattern dans le nom de colonne, nom normalisé)
        (["player_1", "player1", "home", "p1_name", "player_a"], "player1_name"),
        (["player_2", "player2", "away", "p2_name", "player_b"], "player2_name"),
        (["winner", "win", "result"], "winner"),
        (["score_1", "score1", "sets_1", "sets1", "p1_score"], "score_p1"),
        (["score_2", "score2", "sets_2", "sets2", "p2_score"], "score_p2"),
        (["date", "time", "datetime", "match_date", "played"], "played_at"),
        (["match_id", "id", "game_id"], "match_id"),
    ]

    rename_map = {}
    for patterns, target in mappings:
        for col in cols:
            if any(p in col for p in patterns) and target not in rename_map.values():
                rename_map[col] = target
                break

    df = df.rename(columns=rename_map)

    required = ["player1_name", "player2_name", "winner"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        logger.error(
            f"Colonnes requises manquantes : {missing}\n"
            f"Colonnes disponibles : {list(df.columns)}\n"
            f"Éditer kaggle_loader.py pour ajuster le mapping."
        )
        return None

    # Normalise la colonne winner → 1 ou 2
    if "winner" in df.columns:
        df["winner"] = df["winner"].apply(_normalize_winner, p1_col=df.get("player1_name"))

    # Normalise played_at
    if "played_at" in df.columns:
        df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce")
        df = df.dropna(subset=["played_at"])
    else:
        df["played_at"] = datetime(2022, 6, 10)  # date par défaut si absente

    return df


def _normalize_winner(val, p1_col=None) -> int:
    """Convertit la valeur winner en 1 ou 2."""
    if isinstance(val, (int, float)):
        v = int(val)
        return v if v in (1, 2) else 1
    val_str = str(val).strip().lower()
    if val_str in ("1", "home", "p1", "player1", "player_1", "a"):
        return 1
    if val_str in ("2", "away", "p2", "player2", "player_2", "b"):
        return 2
    return 1


def _get_or_create_competition(session) -> Competition:
    comp = session.query(Competition).filter_by(comp_id=COMPETITION_ID).first()
    if not comp:
        comp = Competition(
            comp_id=COMPETITION_ID,
            name="Setka Cup",
            country="UA",
            comp_type="league",
            priority=1,
        )
        session.add(comp)
        session.flush()
    return comp


def _get_or_create_player(session, name: str) -> Player:
    player = session.query(Player).filter_by(name=name).first()
    if not player:
        player = Player(name=name)
        session.add(player)
        session.flush()
    return player
