"""
Requêtes DB pour le dashboard Streamlit.
Toutes les fonctions retournent des DataFrames pandas.
Retournent des DataFrames vides si la DB est vide ou absente.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from sqlalchemy import text
    from src.database.db import engine
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False


def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    if not DB_AVAILABLE:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception:
        return pd.DataFrame()


# ── DATA ─────────────────────────────────────────────────────────────────────

def get_summary_stats() -> dict:
    df = _query("""
        SELECT
            (SELECT COUNT(*) FROM matches)       AS total_matches,
            (SELECT COUNT(*) FROM players)       AS total_players,
            (SELECT COUNT(*) FROM competitions)  AS total_competitions,
            (SELECT COUNT(*) FROM matches WHERE odds_p1 IS NOT NULL) AS matches_with_odds,
            (SELECT MIN(played_at) FROM matches) AS earliest_match,
            (SELECT MAX(played_at) FROM matches) AS latest_match
    """)
    if df.empty:
        return {
            "total_matches": 0, "total_players": 0,
            "total_competitions": 0, "matches_with_odds": 0,
            "earliest_match": None, "latest_match": None,
        }
    row = df.iloc[0]
    return {
        "total_matches": int(row.get("total_matches") or 0),
        "total_players": int(row.get("total_players") or 0),
        "total_competitions": int(row.get("total_competitions") or 0),
        "matches_with_odds": int(row.get("matches_with_odds") or 0),
        "earliest_match": row.get("earliest_match"),
        "latest_match": row.get("latest_match"),
    }


def get_matches_per_competition() -> pd.DataFrame:
    return _query("""
        SELECT c.name AS competition, c.comp_type AS type,
               COUNT(*) AS matches,
               MIN(m.played_at) AS first_match,
               MAX(m.played_at) AS last_match,
               SUM(CASE WHEN m.odds_p1 IS NOT NULL THEN 1 ELSE 0 END) AS with_odds
        FROM matches m
        JOIN competitions c ON m.competition_id = c.id
        GROUP BY c.id, c.name, c.comp_type
        ORDER BY matches DESC
    """)


def get_matches_over_time() -> pd.DataFrame:
    return _query("""
        SELECT strftime('%Y-%m', played_at) AS month,
               c.name AS competition,
               COUNT(*) AS matches
        FROM matches m
        JOIN competitions c ON m.competition_id = c.id
        GROUP BY month, c.name
        ORDER BY month
    """)


def get_top_players(
    limit: int = 20,
    min_matches: int = 10,
    gender: str | None = None,
    countries: list[str] | None = None,
) -> pd.DataFrame:
    where_clauses = []
    params: dict = {"limit": limit, "min_matches": min_matches}

    if gender in ("M", "F"):
        where_clauses.append("p.gender = :gender")
        params["gender"] = gender

    if countries:
        placeholders = ", ".join(f":c{i}" for i in range(len(countries)))
        where_clauses.append(f"p.country IN ({placeholders})")
        for i, c in enumerate(countries):
            params[f"c{i}"] = c

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    return _query(f"""
        SELECT p.name, p.country, p.gender, p.date_of_birth,
               COUNT(DISTINCT m.id) AS matches_played,
               ROUND(AVG(CASE WHEN m.winner = 1 AND m.player1_id = p.id
                              OR m.winner = 2 AND m.player2_id = p.id
                         THEN 1.0 ELSE 0.0 END) * 100, 1) AS win_rate_pct
        FROM players p
        JOIN matches m ON p.id = m.player1_id OR p.id = m.player2_id
        {where_sql}
        GROUP BY p.id, p.name, p.country, p.gender, p.date_of_birth
        HAVING COUNT(DISTINCT m.id) >= :min_matches
        ORDER BY matches_played DESC
        LIMIT :limit
    """, params)


def get_player_countries() -> list[str]:
    df = _query("SELECT DISTINCT country FROM players WHERE country IS NOT NULL ORDER BY country")
    if df.empty:
        return []
    return df["country"].tolist()


def get_ittf_ranking_coverage() -> pd.DataFrame:
    return _query("""
        SELECT strftime('%Y', snapshot_date) AS year,
               COUNT(DISTINCT player_id) AS players_ranked
        FROM ittf_rankings
        GROUP BY year
        ORDER BY year
    """)


# ── BACKTEST ──────────────────────────────────────────────────────────────────

def get_betting_history(paper_only: bool | None = None) -> pd.DataFrame:
    where = ""
    params: dict = {}
    if paper_only is True:
        where = "WHERE b.is_paper = 1"
    elif paper_only is False:
        where = "WHERE b.is_paper = 0"

    return _query(f"""
        SELECT b.placed_at, b.stake, b.odds, b.predicted_prob,
               b.model_edge, b.result, b.profit_loss, b.is_paper,
               c.name AS competition
        FROM betting_records b
        JOIN matches m ON b.match_id = m.id
        JOIN competitions c ON m.competition_id = c.id
        {where}
        ORDER BY b.placed_at
    """, params)


def get_betting_stats_by_competition() -> pd.DataFrame:
    return _query("""
        SELECT c.name AS competition,
               COUNT(*) AS n_bets,
               ROUND(AVG(b.model_edge) * 100, 2) AS avg_edge_pct,
               ROUND(SUM(b.profit_loss) / SUM(b.stake) * 100, 2) AS roi_pct,
               ROUND(AVG(CASE WHEN b.result = 'win' THEN 1.0 ELSE 0.0 END) * 100, 1) AS win_rate_pct
        FROM betting_records b
        JOIN matches m ON b.match_id = m.id
        JOIN competitions c ON m.competition_id = c.id
        GROUP BY c.name
        ORDER BY n_bets DESC
    """)


# ── MONITORING ────────────────────────────────────────────────────────────────

def get_recent_bets(days: int = 30, paper: bool = True) -> pd.DataFrame:
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    return _query("""
        SELECT b.placed_at, b.odds, b.predicted_prob, b.model_edge,
               b.stake, b.result, b.profit_loss,
               p1.name AS player1, p2.name AS player2,
               c.name AS competition
        FROM betting_records b
        JOIN matches m ON b.match_id = m.id
        JOIN players p1 ON m.player1_id = p1.id
        JOIN players p2 ON m.player2_id = p2.id
        JOIN competitions c ON m.competition_id = c.id
        WHERE b.placed_at >= :cutoff
          AND b.is_paper = :paper
        ORDER BY b.placed_at DESC
        LIMIT 100
    """, {"cutoff": cutoff, "paper": 1 if paper else 0})


def get_rolling_roi(window: int = 50) -> pd.DataFrame:
    """ROI glissant sur les N derniers paris."""
    df = _query("""
        SELECT placed_at, profit_loss, stake, is_paper
        FROM betting_records
        WHERE result != 'pending'
        ORDER BY placed_at
    """)
    if df.empty:
        return df
    df["cumulative_pl"] = df["profit_loss"].cumsum()
    df["cumulative_stake"] = df["stake"].cumsum()
    df["cumulative_roi"] = df["cumulative_pl"] / df["cumulative_stake"] * 100
    df["rolling_roi"] = (
        df["profit_loss"].rolling(window).sum()
        / df["stake"].rolling(window).sum()
        * 100
    )
    return df
