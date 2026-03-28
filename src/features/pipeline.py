"""
Pipeline complet d'ingénierie de features.

Charge les matchs depuis la DB, applique tous les calculateurs,
et retourne un DataFrame prêt pour l'entraînement.

Colonnes finales du dataset :
  match_id, played_at, competition_id, winner (target),
  elo_diff, elo_win_prob_p1,
  h2h_matches, h2h_winrate_p1, h2h_recent_winrate_p1,
  form_p1, form_p2, form_diff,
  avg_sets_p1, avg_sets_p2,
  rest_hours_p1, rest_hours_p2, fatigue_p1, fatigue_p2,
  ittf_rank_p1, ittf_rank_p2, rank_diff,
  age_p1, age_p2, age_diff,
  odds_p1, odds_p2, implied_prob_p1
"""
from datetime import datetime

import pandas as pd
import yaml
from loguru import logger
from sqlalchemy import text

from ..database.db import engine
from .elo import EloCalculator
from .form import FormCalculator
from .h2h import H2HCalculator


def load_matches_from_db() -> pd.DataFrame:
    query = text("""
        SELECT
            m.id            AS match_id,
            m.played_at,
            m.player1_id,
            m.player2_id,
            m.winner,
            m.score_p1,
            m.score_p2,
            m.is_walkover,
            m.odds_p1,
            m.odds_p2,
            c.comp_id       AS competition_id,
            c.priority      AS comp_priority,
            p1.date_of_birth AS dob_p1,
            p2.date_of_birth AS dob_p2,
            COALESCE(ir1.rank, 9999) AS ittf_rank_p1,
            COALESCE(ir2.rank, 9999) AS ittf_rank_p2
        FROM matches m
        JOIN competitions c  ON m.competition_id = c.id
        JOIN players p1      ON m.player1_id = p1.id
        JOIN players p2      ON m.player2_id = p2.id
        LEFT JOIN (
            SELECT player_id, rank,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM ittf_rankings
        ) ir1 ON ir1.player_id = m.player1_id AND ir1.rn = 1
        LEFT JOIN (
            SELECT player_id, rank,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM ittf_rankings
        ) ir2 ON ir2.player_id = m.player2_id AND ir2.rn = 1
        WHERE m.is_walkover = 0
        ORDER BY m.played_at ASC
    """)
    df = pd.read_sql(query, engine)
    df["played_at"] = pd.to_datetime(df["played_at"])
    logger.info(f"Chargé {len(df)} matchs depuis la DB")
    return df


def _add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, dob_col in [("age_p1", "dob_p1"), ("age_p2", "dob_p2")]:
        df[dob_col] = pd.to_datetime(df[dob_col])
        age_days = (df["played_at"] - df[dob_col]).dt.days
        df[col] = (age_days / 365.25).clip(lower=0, upper=50).fillna(25.0)
    df["age_diff"] = df["age_p1"] - df["age_p2"]
    return df


def _add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["odds_p1"].notna() & df["odds_p2"].notna() & (df["odds_p1"] > 1) & (df["odds_p2"] > 1)
    df["implied_prob_p1"] = 0.5
    df.loc[mask, "implied_prob_p1"] = (
        (1 / df.loc[mask, "odds_p1"])
        / (1 / df.loc[mask, "odds_p1"] + 1 / df.loc[mask, "odds_p2"])
    )
    return df


def build_features(config_path: str = "config/settings.yaml") -> pd.DataFrame:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    feat_cfg = config.get("features", {})
    elo_cfg = feat_cfg.get("elo", {})
    form_cfg = feat_cfg.get("form", {})
    h2h_cfg = feat_cfg.get("h2h", {})

    df = load_matches_from_db()
    if df.empty:
        logger.warning("Aucun match en base — dataset vide")
        return df

    logger.info("Calcul des features Elo...")
    elo_calc = EloCalculator(
        initial_rating=elo_cfg.get("initial_rating", 1500),
        k_default=elo_cfg.get("k_factor_default", 32),
        k_new_player=elo_cfg.get("k_factor_new_player", 40),
        competition_weights=elo_cfg.get("competition_weights", {}),
    )
    df = elo_calc.compute(df)

    logger.info("Calcul des features H2H...")
    h2h_calc = H2HCalculator(
        min_matches=h2h_cfg.get("min_matches", 3),
        max_age_days=h2h_cfg.get("max_age_days", 730),
    )
    df = h2h_calc.compute(df)

    logger.info("Calcul des features de forme récente...")
    form_calc = FormCalculator(
        window=form_cfg.get("window", 5),
        decay=form_cfg.get("decay", 0.85),
        fatigue_threshold_hours=feat_cfg.get("rest", {}).get("fatigue_threshold_hours", 48),
    )
    df = form_calc.compute(df)

    df = _add_age_features(df)
    df = _add_odds_features(df)

    df["rank_diff"] = df["ittf_rank_p1"] - df["ittf_rank_p2"]

    # Target : 1 si P1 gagne, 0 sinon
    df["target"] = (df["winner"] == 1).astype(int)

    FEATURE_COLS = [
        "elo_diff", "elo_win_prob_p1",
        "h2h_matches", "h2h_winrate_p1", "h2h_recent_winrate_p1",
        "form_p1", "form_p2", "form_diff",
        "avg_sets_p1", "avg_sets_p2",
        "rest_hours_p1", "rest_hours_p2", "fatigue_p1", "fatigue_p2",
        "ittf_rank_p1", "ittf_rank_p2", "rank_diff",
        "age_p1", "age_p2", "age_diff",
        "implied_prob_p1",
    ]
    META_COLS = ["match_id", "played_at", "competition_id", "odds_p1", "odds_p2", "target"]

    available = [c for c in FEATURE_COLS + META_COLS if c in df.columns]
    result = df[available].copy()

    logger.info(
        f"Dataset prêt : {len(result)} lignes, "
        f"{len([c for c in FEATURE_COLS if c in result.columns])} features"
    )
    return result
