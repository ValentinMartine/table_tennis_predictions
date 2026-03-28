"""
Requêtes DB pour le dashboard Streamlit.
Toutes les fonctions retournent des DataFrames pandas.
Retournent des DataFrames vides si la DB est vide ou absente.
"""
from __future__ import annotations

import json
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


def get_matches_per_competition(include_archived: bool = True) -> pd.DataFrame:
    where = "" if include_archived else "WHERE c.priority < 99"
    return _query(f"""
        SELECT c.name AS competition, c.comp_type AS type, c.priority,
               COUNT(*) AS matches,
               MIN(m.played_at) AS first_match,
               MAX(m.played_at) AS last_match,
               SUM(CASE WHEN m.odds_p1 IS NOT NULL THEN 1 ELSE 0 END) AS with_odds
        FROM matches m
        JOIN competitions c ON m.competition_id = c.id
        {where}
        GROUP BY c.id, c.name, c.comp_type, c.priority
        ORDER BY c.priority, matches DESC
    """)


def get_matches_over_time(include_archived: bool = False) -> pd.DataFrame:
    where = "" if include_archived else "WHERE c.priority < 99"
    return _query(f"""
        SELECT strftime('%Y-%m', played_at) AS month,
               c.name AS competition,
               c.priority,
               COUNT(*) AS matches
        FROM matches m
        JOIN competitions c ON m.competition_id = c.id
        {where}
        GROUP BY month, c.name, c.priority
        ORDER BY month
    """)


def get_competition_status() -> pd.DataFrame:
    """Statut de chaque compétition : dernière date, jours depuis le dernier match."""
    return _query("""
        SELECT c.name AS competition,
               c.comp_id,
               c.priority,
               c.comp_type AS type,
               COUNT(*) AS total_matches,
               MIN(m.played_at) AS first_match,
               MAX(m.played_at) AS last_match
        FROM competitions c
        LEFT JOIN matches m ON m.competition_id = c.id
        GROUP BY c.id, c.name, c.comp_id, c.priority, c.comp_type
        ORDER BY c.priority, c.name
    """)


def get_top_players(
    limit: int = 20,
    min_matches: int = 10,
    gender: str | None = None,
    countries: list[str] | None = None,
    priority_max: int = 98,
) -> pd.DataFrame:
    where_clauses = ["c.priority <= :priority_max"]
    params: dict = {"limit": limit, "min_matches": min_matches, "priority_max": priority_max}

    if gender in ("M", "F"):
        where_clauses.append("p.gender = :gender")
        params["gender"] = gender

    if countries:
        placeholders = ", ".join(f":c{i}" for i in range(len(countries)))
        where_clauses.append(f"p.country IN ({placeholders})")
        for i, c in enumerate(countries):
            params[f"c{i}"] = c

    where_sql = "WHERE " + " AND ".join(where_clauses)

    return _query(f"""
        SELECT p.name, p.country, p.gender, p.date_of_birth,
               COUNT(DISTINCT m.id) AS matches_played,
               ROUND(AVG(CASE WHEN m.winner = 1 AND m.player1_id = p.id
                              OR m.winner = 2 AND m.player2_id = p.id
                         THEN 1.0 ELSE 0.0 END) * 100, 1) AS win_rate_pct
        FROM players p
        JOIN matches m ON p.id = m.player1_id OR p.id = m.player2_id
        JOIN competitions c ON m.competition_id = c.id
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


def get_player_names(search: str = "", limit: int = 500) -> list[str]:
    """Retourne les noms de joueurs correspondant à la recherche (pour autocomplete)."""
    df = _query("""
        SELECT p.name, COUNT(m.id) AS n
        FROM players p
        JOIN matches m ON p.id = m.player1_id OR p.id = m.player2_id
        WHERE LOWER(p.name) LIKE :search
        GROUP BY p.id, p.name
        HAVING n >= 3
        ORDER BY n DESC
        LIMIT :limit
    """, {"search": f"%{search.lower()}%", "limit": limit})
    if df.empty:
        return []
    return df["name"].tolist()


def get_ittf_ranking_coverage() -> pd.DataFrame:
    return _query("""
        SELECT strftime('%Y', snapshot_date) AS year,
               COUNT(DISTINCT player_id) AS players_ranked
        FROM ittf_rankings
        GROUP BY year
        ORDER BY year
    """)


# ── H2H ───────────────────────────────────────────────────────────────────────

def get_h2h(player1_name: str, player2_name: str) -> pd.DataFrame:
    """Tous les matchs entre deux joueurs (dans les deux sens)."""
    return _query("""
        SELECT m.played_at, c.name AS competition,
               p1.name AS player1, p2.name AS player2,
               m.score_p1, m.score_p2, m.winner,
               m.round_name
        FROM matches m
        JOIN players p1 ON m.player1_id = p1.id
        JOIN players p2 ON m.player2_id = p2.id
        JOIN competitions c ON m.competition_id = c.id
        WHERE (p1.name = :n1 AND p2.name = :n2)
           OR (p1.name = :n2 AND p2.name = :n1)
        ORDER BY m.played_at DESC
    """, {"n1": player1_name, "n2": player2_name})


def get_h2h_summary(player1_name: str, player2_name: str) -> dict:
    """Résumé H2H : victoires, sets, dernière rencontre."""
    df = get_h2h(player1_name, player2_name)
    if df.empty:
        return {"matches": 0, "p1_wins": 0, "p2_wins": 0, "p1_sets": 0, "p2_sets": 0}

    p1_wins = 0
    p2_wins = 0
    p1_sets = 0
    p2_sets = 0

    for _, row in df.iterrows():
        if row["player1"] == player1_name:
            p1_sets += int(row["score_p1"] or 0)
            p2_sets += int(row["score_p2"] or 0)
            if row["winner"] == 1:
                p1_wins += 1
            else:
                p2_wins += 1
        else:
            p1_sets += int(row["score_p2"] or 0)
            p2_sets += int(row["score_p1"] or 0)
            if row["winner"] == 2:
                p1_wins += 1
            else:
                p2_wins += 1

    return {
        "matches": len(df),
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "p1_sets": p1_sets,
        "p2_sets": p2_sets,
    }


def get_player_stats(player_name: str) -> dict:
    """Stats globales d'un joueur."""
    df = _query("""
        SELECT COUNT(DISTINCT m.id) AS total_matches,
               SUM(CASE WHEN (m.winner = 1 AND m.player1_id = p.id)
                          OR (m.winner = 2 AND m.player2_id = p.id)
                        THEN 1 ELSE 0 END) AS wins,
               MIN(m.played_at) AS first_match,
               MAX(m.played_at) AS last_match
        FROM players p
        JOIN matches m ON p.id = m.player1_id OR p.id = m.player2_id
        WHERE p.name = :name
    """, {"name": player_name})
    if df.empty or df.iloc[0]["total_matches"] == 0:
        return {}
    row = df.iloc[0]
    total = int(row["total_matches"] or 0)
    wins = int(row["wins"] or 0)
    return {
        "total_matches": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "first_match": row["first_match"],
        "last_match": row["last_match"],
    }


def get_player_match_history(player_name: str, limit: int = 100) -> pd.DataFrame:
    """Historique des derniers matchs d'un joueur."""
    return _query("""
        SELECT m.played_at, c.name AS competition,
               p1.name AS player1, p2.name AS player2,
               m.score_p1, m.score_p2, m.winner,
               m.round_name
        FROM matches m
        JOIN players p1 ON m.player1_id = p1.id
        JOIN players p2 ON m.player2_id = p2.id
        JOIN competitions c ON m.competition_id = c.id
        WHERE p1.name = :name OR p2.name = :name
        ORDER BY m.played_at DESC
        LIMIT :limit
    """, {"name": player_name, "limit": limit})


# ── ELO HISTORY ───────────────────────────────────────────────────────────────

def get_player_elo_history(player_name: str) -> pd.DataFrame:
    """
    Historique Elo mensuel d'un joueur depuis data/elo_history.csv.
    Disponible uniquement après l'entraînement du modèle.
    """
    path = Path("data/elo_history.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        player_df = df[df["name"] == player_name][["played_at", "elo_rating"]].copy()
        player_df["played_at"] = pd.to_datetime(player_df["played_at"], errors="coerce")
        return player_df.sort_values("played_at")
    except Exception:
        return pd.DataFrame()


def get_latest_elo(player_name: str) -> float:
    """Dernière valeur Elo connue d'un joueur (depuis elo_history.csv)."""
    df = get_player_elo_history(player_name)
    if df.empty:
        return 1500.0
    return float(df["elo_rating"].iloc[-1])


def get_player_rolling_winrate(player_name: str, window: int = 10) -> pd.DataFrame:
    """
    Win rate glissant sur les N derniers matchs (proxy de forme, toujours disponible).
    """
    df = _query("""
        SELECT m.played_at,
               CASE WHEN (m.winner = 1 AND m.player1_id = p.id)
                      OR (m.winner = 2 AND m.player2_id = p.id)
                    THEN 1 ELSE 0 END AS won
        FROM players p
        JOIN matches m ON p.id = m.player1_id OR p.id = m.player2_id
        WHERE p.name = :name
        ORDER BY m.played_at ASC
    """, {"name": player_name})
    if df.empty:
        return df
    df["played_at"] = pd.to_datetime(df["played_at"])
    df["rolling_winrate"] = df["won"].rolling(window, min_periods=3).mean() * 100
    return df


# ── PRÉDICTION / EDGE CALCULATOR ──────────────────────────────────────────────

def get_player_form_value(player_name: str, last_n: int = 5) -> float:
    """Win rate sur les N derniers matchs [0, 1]."""
    df = _query("""
        SELECT m.winner, m.player1_id, m.player2_id, p.id AS pid
        FROM players p
        JOIN matches m ON p.id = m.player1_id OR p.id = m.player2_id
        WHERE p.name = :name
        ORDER BY m.played_at DESC
        LIMIT :n
    """, {"name": player_name, "n": last_n})
    if df.empty:
        return 0.5
    wins = sum(
        1 for _, r in df.iterrows()
        if (r["winner"] == 1 and r["player1_id"] == r["pid"])
        or (r["winner"] == 2 and r["player2_id"] == r["pid"])
    )
    return wins / len(df)


def get_player_ittf_rank(player_name: str) -> int:
    """Dernier rang ITTF connu du joueur."""
    df = _query("""
        SELECT ir.rank
        FROM players p
        JOIN ittf_rankings ir ON ir.player_id = p.id
        WHERE p.name = :name
        ORDER BY ir.snapshot_date DESC
        LIMIT 1
    """, {"name": player_name})
    if df.empty or df.iloc[0]["rank"] is None:
        return 9999
    return int(df.iloc[0]["rank"])


def get_player_wtt_rank(player_name: str) -> tuple[int, float | None]:
    """Dernier rang WTT + points YTD connus du joueur. Retourne (rank, points_ytd)."""
    df = _query("""
        SELECT wr.rank, wr.points_ytd
        FROM players p
        JOIN wtt_rankings wr ON wr.player_id = p.id
        WHERE p.name = :name
        ORDER BY wr.snapshot_date DESC
        LIMIT 1
    """, {"name": player_name})
    if df.empty or df.iloc[0]["rank"] is None:
        return 9999, None
    pts = df.iloc[0]["points_ytd"]
    return int(df.iloc[0]["rank"]), (float(pts) if pts is not None else None)


def get_player_info(player_name: str) -> dict:
    """Infos de base : age, gender, country."""
    df = _query("""
        SELECT date_of_birth, gender, country
        FROM players
        WHERE name = :name
        LIMIT 1
    """, {"name": player_name})
    if df.empty:
        return {"age": 25.0, "gender": None, "country": None}
    row = df.iloc[0]
    age = 25.0
    if row.get("date_of_birth"):
        try:
            dob = pd.to_datetime(row["date_of_birth"])
            age = (datetime.utcnow() - dob).days / 365.25
        except Exception:
            pass
    return {
        "age": round(age, 1),
        "gender": row.get("gender"),
        "country": row.get("country"),
    }


def get_features_for_prediction(p1_name: str, p2_name: str) -> dict:
    """
    Construit le vecteur de features pour prédire un match entre deux joueurs.
    Utilise les dernières valeurs disponibles en DB + elo_history.csv.
    """
    elo_p1 = get_latest_elo(p1_name)
    elo_p2 = get_latest_elo(p2_name)
    from src.features.elo import expected_score
    elo_diff = elo_p1 - elo_p2
    elo_win_prob = expected_score(elo_p1, elo_p2)

    h2h = get_h2h_summary(p1_name, p2_name)
    h2h_matches = h2h["matches"]
    h2h_winrate = h2h["p1_wins"] / h2h_matches if h2h_matches >= 3 else 0.5
    h2h_recent_winrate = h2h_winrate  # approximation

    form_p1 = get_player_form_value(p1_name, 5)
    form_p2 = get_player_form_value(p2_name, 5)

    rank_p1 = get_player_ittf_rank(p1_name)
    rank_p2 = get_player_ittf_rank(p2_name)

    info_p1 = get_player_info(p1_name)
    info_p2 = get_player_info(p2_name)

    return {
        "elo_diff": elo_diff,
        "elo_win_prob_p1": elo_win_prob,
        "h2h_matches": h2h_matches,
        "h2h_winrate_p1": h2h_winrate,
        "h2h_recent_winrate_p1": h2h_recent_winrate,
        "form_p1": form_p1,
        "form_p2": form_p2,
        "form_diff": form_p1 - form_p2,
        "avg_sets_p1": 2.0,
        "avg_sets_p2": 2.0,
        "rest_hours_p1": 48.0,
        "rest_hours_p2": 48.0,
        "fatigue_p1": 0,
        "fatigue_p2": 0,
        "ittf_rank_p1": rank_p1,
        "ittf_rank_p2": rank_p2,
        "rank_diff": rank_p1 - rank_p2,
        "age_p1": info_p1["age"],
        "age_p2": info_p2["age"],
        "age_diff": info_p1["age"] - info_p2["age"],
        "implied_prob_p1": elo_win_prob,
        # Infos complémentaires pour affichage
        "_elo_p1": elo_p1,
        "_elo_p2": elo_p2,
        "_p1_name": p1_name,
        "_p2_name": p2_name,
    }


# ── COMPARAISON MODÈLES ───────────────────────────────────────────────────────

def get_all_model_metrics() -> dict[str, dict]:
    """Charge les métriques de tous les modèles disponibles."""
    result = {}
    for model_name in ("lgbm", "xgb"):
        path = Path(f"data/{model_name}_metrics.json")
        if path.exists():
            try:
                result[model_name.upper()] = json.loads(path.read_text())
            except Exception:
                pass
    # Baseline Elo
    elo_path = Path("data/elo_baseline_metrics.json")
    if elo_path.exists():
        try:
            result["Elo baseline"] = json.loads(elo_path.read_text())
        except Exception:
            pass
    return result


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
