"""
Pipeline complet d'ingénierie de features.

Charge les matchs depuis la DB, applique tous les calculateurs,
et retourne un DataFrame prêt pour l'entraînement.

Colonnes finales du dataset :
  match_id, played_at, competition_id, winner (target),
  elo_diff, elo_win_prob_p1,
  elo_intl_diff, elo_intl_win_prob_p1,
  h2h_matches, h2h_winrate_p1, h2h_recent_winrate_p1,
  form_p1, form_p2, form_diff,
  avg_sets_p1, avg_sets_p2,
  avg_set_margin_p1, avg_set_margin_p2, set_margin_diff,
  close_sets_rate_p1, close_sets_rate_p2,
  rest_hours_p1, rest_hours_p2, fatigue_p1, fatigue_p2,
  ittf_rank_p1, ittf_rank_p2, rank_diff,
  wtt_rank_p1, wtt_rank_p2, wtt_rank_diff,
  rank_velocity_p1, rank_velocity_p2, rank_velocity_diff,
  rank_stability_p1, rank_stability_p2,
  age_p1, age_p2, age_diff,
  has_odds, implied_prob_p1
"""
import numpy as np
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
            m.sets_detail,
            m.is_walkover,
            m.odds_p1,
            m.odds_p2,
            c.comp_id       AS competition_id,
            c.priority      AS comp_priority,
            p1.date_of_birth AS dob_p1,
            p2.date_of_birth AS dob_p2,
            COALESCE(ir1.rank, 9999) AS ittf_rank_p1,
            COALESCE(ir2.rank, 9999) AS ittf_rank_p2,
            COALESCE(wr1.rank, 9999) AS wtt_rank_p1,
            COALESCE(wr2.rank, 9999) AS wtt_rank_p2
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
        LEFT JOIN (
            SELECT player_id, rank,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM wtt_rankings
        ) wr1 ON wr1.player_id = m.player1_id AND wr1.rn = 1
        LEFT JOIN (
            SELECT player_id, rank,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM wtt_rankings
        ) wr2 ON wr2.player_id = m.player2_id AND wr2.rn = 1
        WHERE m.is_walkover = 0
          AND c.priority < 99
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
    odds_p1 = pd.to_numeric(df["odds_p1"], errors="coerce")
    odds_p2 = pd.to_numeric(df["odds_p2"], errors="coerce")
    mask = odds_p1.notna() & odds_p2.notna() & (odds_p1 > 1) & (odds_p2 > 1)
    implied = pd.Series(0.5, index=df.index, dtype="float64")
    implied.loc[mask] = (1 / odds_p1[mask]) / (1 / odds_p1[mask] + 1 / odds_p2[mask])
    df["has_odds"] = mask.astype(int)
    df["implied_prob_p1"] = implied
    return df


def _add_ranking_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute rank_velocity et rank_stability à partir des snapshots ittf_rankings.

    rank_velocity > 0 → joueur en progression (rang qui s'améliore = numéro qui baisse)
    rank_stability   → écart-type du rang sur 180 jours (faible = joueur stable)
    """
    _zero_cols = [
        "rank_velocity_p1", "rank_velocity_p2", "rank_velocity_diff",
        "rank_stability_p1", "rank_stability_p2",
    ]
    try:
        with engine.connect() as conn:
            ranks = pd.read_sql(
                text("SELECT player_id, rank, snapshot_date FROM ittf_rankings ORDER BY player_id, snapshot_date"),
                conn,
            )
    except Exception as e:
        logger.warning(f"Impossible de charger ittf_rankings pour ranking trajectory : {e}")
        for col in _zero_cols:
            df[col] = 0.0
        return df

    if ranks.empty:
        for col in _zero_cols:
            df[col] = 0.0
        return df

    ranks["snapshot_date"] = pd.to_datetime(ranks["snapshot_date"])

    # Précalcul : dict player_id → (dates_ns array, ranks array) trié chronologiquement
    rank_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid, grp in ranks.groupby("player_id"):
        grp_s = grp.sort_values("snapshot_date")
        rank_arrays[int(pid)] = (
            grp_s["snapshot_date"].values.astype("int64"),
            grp_s["rank"].values.astype("float64"),
        )

    WINDOW_NS = np.int64(pd.Timedelta(days=180).value)

    def _features(player_id: int, match_ns: np.int64) -> tuple[float, float]:
        arr = rank_arrays.get(int(player_id))
        if arr is None:
            return 0.0, 0.0
        dates_ns, rank_vals = arr
        end = int(np.searchsorted(dates_ns, match_ns, side="left"))
        if end == 0:
            return 0.0, 0.0
        start = int(np.searchsorted(dates_ns, match_ns - WINDOW_NS, side="left"))
        window = rank_vals[start:end]
        if len(window) < 2:
            return 0.0, 0.0
        # Positif = rang s'améliore (premier rang élevé → dernier rang bas)
        velocity = float(np.clip(window[0] - window[-1], -200, 200))
        stability = float(np.std(window))
        return velocity, stability

    match_dates_ns = df["played_at"].values.astype("int64")
    p1_ids = df["player1_id"].values
    p2_ids = df["player2_id"].values
    n = len(df)

    vel_p1 = np.empty(n)
    vel_p2 = np.empty(n)
    stab_p1 = np.empty(n)
    stab_p2 = np.empty(n)

    for i in range(n):
        vel_p1[i], stab_p1[i] = _features(p1_ids[i], match_dates_ns[i])
        vel_p2[i], stab_p2[i] = _features(p2_ids[i], match_dates_ns[i])

    df["rank_velocity_p1"] = vel_p1
    df["rank_velocity_p2"] = vel_p2
    df["rank_velocity_diff"] = vel_p1 - vel_p2
    df["rank_stability_p1"] = stab_p1
    df["rank_stability_p2"] = stab_p2
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
        min_matches=h2h_cfg.get("min_matches", 1),
        max_age_days=h2h_cfg.get("max_age_days", 1825),
    )
    df = h2h_calc.compute(df)

    logger.info("Calcul des features de forme récente...")
    form_calc = FormCalculator(
        window=form_cfg.get("window", 10),
        decay=form_cfg.get("decay", 0.85),
        fatigue_threshold_hours=feat_cfg.get("rest", {}).get("fatigue_threshold_hours", 48),
    )
    df = form_calc.compute(df)

    df = _add_age_features(df)
    df = _add_odds_features(df)

    df["rank_diff"] = df["ittf_rank_p1"] - df["ittf_rank_p2"]
    df["wtt_rank_diff"] = df["wtt_rank_p1"] - df["wtt_rank_p2"]

    logger.info("Calcul de la trajectoire de classement ITTF...")
    df = _add_ranking_trajectory(df)

    # Target : 1 si P1 gagne, 0 sinon
    df["target"] = (df["winner"] == 1).astype(int)

    FEATURE_COLS = [
        # Elo global (toutes compétitions)
        "elo_diff", "elo_win_prob_p1",
        # Elo international (WTT Champions/Star Contender/Cup Finals, Worlds, JO)
        "elo_intl_diff", "elo_intl_win_prob_p1",
        # H2H (corrigé : min_matches=1, fenêtre 5 ans)
        "h2h_matches", "h2h_winrate_p1", "h2h_recent_winrate_p1",
        # Forme récente (fenêtre 10 matchs)
        "form_p1", "form_p2", "form_diff",
        # Sets (volume + dominance)
        "avg_sets_p1", "avg_sets_p2",
        "avg_set_margin_p1", "avg_set_margin_p2", "set_margin_diff",
        "close_sets_rate_p1", "close_sets_rate_p2",
        # Fatigue & repos
        "rest_hours_p1", "rest_hours_p2", "fatigue_p1", "fatigue_p2",
        # Rankings statiques
        "ittf_rank_p1", "ittf_rank_p2", "rank_diff",
        "wtt_rank_p1", "wtt_rank_p2", "wtt_rank_diff",
        # Trajectoire de classement (dynamique)
        "rank_velocity_p1", "rank_velocity_p2", "rank_velocity_diff",
        "rank_stability_p1", "rank_stability_p2",
        # Âge
        "age_p1", "age_p2", "age_diff",
        # Cotes bookmaker
        "has_odds", "implied_prob_p1",
    ]
    META_COLS = [
        "match_id", "played_at", "competition_id", "odds_p1", "odds_p2", "target",
        "player1_id", "player2_id", "elo_p1", "elo_p2", "elo_intl_p1", "elo_intl_p2",
    ]

    available = [c for c in FEATURE_COLS + META_COLS if c in df.columns]
    result = df[available].copy()

    logger.info(
        f"Dataset prêt : {len(result)} lignes, "
        f"{len([c for c in FEATURE_COLS if c in result.columns])} features"
    )
    return result
