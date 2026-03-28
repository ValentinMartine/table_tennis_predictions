"""
Prédit les chances de victoire pour les matchs WTT/internationaux à venir.

Récupère les prochains matchs depuis Sofascore (7 jours),
reconstruit les features depuis la DB, et affiche les prédictions.

Usage :
    python scripts/predict_upcoming.py
    python scripts/predict_upcoming.py --days 3 --model lgbm
    python scripts/predict_upcoming.py --min-conf 0.60
"""
import argparse
import io
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.database.db import engine
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel

try:
    from curl_cffi import requests as cffi_requests
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

API_BASE = "https://api.sofascore.com/api/v1"
_CFFI_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
}

# Compétitions WTT/internationales ciblées (sous-chaînes dans le nom Sofascore)
TARGET_PATTERNS_INTL = [
    "WTT Champions", "WTT Star Contender", "WTT Contender",
    "WTT Cup Finals", "World Championships", "European Championships",
    "Olympic Games", "Team World Cup",
]
TARGET_PATTERNS_ALL = TARGET_PATTERNS_INTL + [
    "Liga Pro", "Setka Cup", "ETTU Champions",
    "Czech Liga Pro", "TT Cup", "TT Elite Series",
]


# ── Sofascore ────────────────────────────────────────────────────────────────

def _sfetch(session, url: str) -> dict:
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug(f"Sofascore GET {url}: {e}")
        return {}


def fetch_upcoming_matches(days: int = 7, all_leagues: bool = False) -> list[dict]:
    """Retourne les matchs non-commencés sur les `days` prochains jours."""
    if not _HAS_CURL_CFFI:
        logger.error("curl-cffi requis. pip install curl-cffi")
        return []

    patterns = TARGET_PATTERNS_ALL if all_leagues else TARGET_PATTERNS_INTL

    session = cffi_requests.Session(impersonate="chrome120")
    session.headers.update(_CFFI_HEADERS)

    results = []
    for d in range(days + 1):
        day = date.today() + timedelta(days=d)
        data = _sfetch(session, f"{API_BASE}/sport/table-tennis/scheduled-events/{day}")
        for ev in data.get("events", []):
            status = ev.get("status", {}).get("type", "")
            if status not in ("notstarted", "inprogress"):
                continue
            t = ev.get("tournament", {})
            tournament_name = t.get("name", "") or t.get("uniqueTournament", {}).get("name", "")
            if not any(p.lower() in tournament_name.lower() for p in patterns):
                continue
            results.append({
                "event_id": ev["id"],
                "tournament": tournament_name,
                "p1_name": ev.get("homeTeam", {}).get("name", ""),
                "p2_name": ev.get("awayTeam", {}).get("name", ""),
                "start_time": ev.get("startTimestamp", 0),
                "status": status,
            })
        import time; time.sleep(0.5)

    scope = "toutes compétitions" if all_leagues else "WTT/internationaux"
    logger.info(f"{len(results)} matchs {scope} à venir (prochains {days} jours)")
    return results


# ── Player lookup ─────────────────────────────────────────────────────────────

def _load_player_map() -> pd.DataFrame:
    """Charge {player_id, name, ittf_rank, wtt_rank} depuis la DB."""
    query = text("""
        SELECT p.id, p.name,
               COALESCE(ir.rank, 9999) AS ittf_rank,
               COALESCE(wr.rank, 9999) AS wtt_rank
        FROM players p
        LEFT JOIN (
            SELECT player_id, rank,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM ittf_rankings
        ) ir ON ir.player_id = p.id AND ir.rn = 1
        LEFT JOIN (
            SELECT player_id, rank,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM wtt_rankings
        ) wr ON wr.player_id = p.id AND wr.rn = 1
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def _match_player(name: str, player_map: pd.DataFrame) -> int | None:
    """Retourne player_id depuis le nom (exact ou partiel)."""
    name_lower = name.lower().strip()
    # Exact match
    exact = player_map[player_map["name"].str.lower() == name_lower]
    if not exact.empty:
        return int(exact.iloc[0]["id"])
    # Partial match (last name)
    parts = name_lower.split()
    if parts:
        last = parts[-1]
        partial = player_map[player_map["name"].str.lower().str.contains(last, na=False)]
        if len(partial) == 1:
            return int(partial.iloc[0]["id"])
    return None


# ── Feature building ──────────────────────────────────────────────────────────

def _build_player_stats(player_id: int) -> dict:
    """Construit les dernières stats connues pour un joueur depuis la DB."""
    query = text("""
        SELECT
            m.played_at,
            CASE WHEN m.player1_id = :pid THEN m.winner = 1
                 ELSE m.winner = 2 END AS won,
            CASE WHEN m.player1_id = :pid THEN m.score_p1
                 ELSE m.score_p2 END AS sets_won,
            CASE WHEN m.player1_id = :pid THEN m.score_p2
                 ELSE m.score_p1 END AS sets_lost
        FROM matches m
        WHERE (m.player1_id = :pid OR m.player2_id = :pid)
          AND m.is_walkover = 0
        ORDER BY m.played_at DESC
        LIMIT 10
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"pid": player_id})

    if df.empty:
        return {"form": 0.5, "avg_sets": 2.0, "rest_hours": 48.0, "fatigue": 0}

    df["played_at"] = pd.to_datetime(df["played_at"])
    last_match = df.iloc[0]["played_at"]
    rest_hours = min((pd.Timestamp.now() - last_match).total_seconds() / 3600, 168)

    # Forme sur les 5 derniers (pondération exponentielle)
    recent = df.head(5)
    weights = [0.85 ** i for i in range(len(recent))]
    wins = [1 if r else 0 for r in recent["won"]]
    form = sum(w * x for w, x in zip(weights, wins)) / sum(weights) if weights else 0.5

    avg_sets = df.head(5)["sets_won"].mean() if not df.empty else 2.0

    return {
        "form": round(form, 3),
        "avg_sets": round(avg_sets, 2),
        "rest_hours": round(rest_hours, 1),
        "fatigue": 1 if rest_hours < 48 else 0,
    }


def _get_elo(player_id: int) -> float:
    query = text("""
        SELECT rating FROM elo_ratings
        WHERE player_id = :pid
        ORDER BY computed_at DESC LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"pid": player_id}).fetchone()
    return float(row[0]) if row else 1500.0


def _get_h2h(p1_id: int, p2_id: int) -> dict:
    query = text("""
        SELECT winner, player1_id
        FROM matches
        WHERE ((player1_id = :p1 AND player2_id = :p2)
            OR (player1_id = :p2 AND player2_id = :p1))
          AND is_walkover = 0
        ORDER BY played_at DESC
        LIMIT 20
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"p1": p1_id, "p2": p2_id})

    if df.empty or len(df) < 1:
        return {"h2h_matches": 0, "h2h_winrate_p1": 0.5, "h2h_recent_winrate_p1": 0.5}

    p1_wins = sum(
        1 for _, r in df.iterrows()
        if (r["player1_id"] == p1_id and r["winner"] == 1)
        or (r["player1_id"] == p2_id and r["winner"] == 2)
    )
    total = len(df)
    recent = df.head(5)
    p1_wins_recent = sum(
        1 for _, r in recent.iterrows()
        if (r["player1_id"] == p1_id and r["winner"] == 1)
        or (r["player1_id"] == p2_id and r["winner"] == 2)
    )
    return {
        "h2h_matches": total,
        "h2h_winrate_p1": round(p1_wins / total, 3),
        "h2h_recent_winrate_p1": round(p1_wins_recent / len(recent), 3),
    }


def _elo_win_prob(elo1: float, elo2: float) -> float:
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))


def build_features_for_match(p1_id: int, p2_id: int,
                              p1_ittf: int, p2_ittf: int) -> pd.DataFrame:
    elo1 = _get_elo(p1_id)
    elo2 = _get_elo(p2_id)
    stats1 = _build_player_stats(p1_id)
    stats2 = _build_player_stats(p2_id)
    h2h = _get_h2h(p1_id, p2_id)
    elo_prob = _elo_win_prob(elo1, elo2)

    return pd.DataFrame([{
        "elo_diff": round(elo1 - elo2, 1),
        "elo_win_prob_p1": round(elo_prob, 4),
        "h2h_matches": h2h["h2h_matches"],
        "h2h_winrate_p1": h2h["h2h_winrate_p1"],
        "h2h_recent_winrate_p1": h2h["h2h_recent_winrate_p1"],
        "form_p1": stats1["form"],
        "form_p2": stats2["form"],
        "form_diff": round(stats1["form"] - stats2["form"], 3),
        "avg_sets_p1": stats1["avg_sets"],
        "avg_sets_p2": stats2["avg_sets"],
        "rest_hours_p1": stats1["rest_hours"],
        "rest_hours_p2": stats2["rest_hours"],
        "fatigue_p1": stats1["fatigue"],
        "fatigue_p2": stats2["fatigue"],
        "ittf_rank_p1": p1_ittf,
        "ittf_rank_p2": p2_ittf,
        "rank_diff": p1_ittf - p2_ittf,
        "age_p1": 25.0,
        "age_p2": 25.0,
        "age_diff": 0.0,
        "implied_prob_p1": round(elo_prob, 4),
    }])


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prédictions matchs WTT/intl à venir")
    p.add_argument("--days", type=int, default=7, help="Horizon en jours (défaut: 7)")
    p.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="Confiance minimum à afficher (ex: 0.60)")
    p.add_argument("--all-leagues", action="store_true",
                   help="Inclure aussi Liga Pro, Setka Cup, ETTU, etc.")
    return p.parse_args()


def main():
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    # Charger le modèle
    model_path = f"data/{args.model}_model.pkl"
    model = LGBMModel.load(model_path) if args.model == "lgbm" else XGBModel.load(model_path)
    logger.info(f"Modèle {args.model.upper()} chargé")

    # Charger la map joueurs
    player_map = _load_player_map()
    logger.info(f"{len(player_map)} joueurs en DB")

    # Récupérer les matchs à venir
    upcoming = fetch_upcoming_matches(days=args.days, all_leagues=args.all_leagues)
    if not upcoming:
        logger.info("Aucun match WTT/international à venir trouvé")
        return

    # Prédire
    predictions = []
    not_found = []

    for ev in upcoming:
        p1_id = _match_player(ev["p1_name"], player_map)
        p2_id = _match_player(ev["p2_name"], player_map)

        if p1_id is None or p2_id is None:
            not_found.append(f"{ev['p1_name']} vs {ev['p2_name']}")
            continue

        p1_row = player_map[player_map["id"] == p1_id].iloc[0]
        p2_row = player_map[player_map["id"] == p2_id].iloc[0]

        features = build_features_for_match(
            p1_id, p2_id,
            int(p1_row["ittf_rank"]), int(p2_row["ittf_rank"])
        )

        prob_p1 = float(model.predict_proba(features)[0])
        prob_p2 = 1 - prob_p1

        # Détecter le favori
        fav = ev["p1_name"] if prob_p1 >= 0.5 else ev["p2_name"]
        fav_prob = max(prob_p1, prob_p2)

        if fav_prob >= args.min_conf:
            import datetime as _dt
            start = _dt.datetime.fromtimestamp(ev["start_time"]).strftime("%d/%m %H:%M") \
                    if ev["start_time"] else "?"
            predictions.append({
                "Date": start,
                "Tournoi": ev["tournament"][:30],
                "J1": ev["p1_name"],
                "J2": ev["p2_name"],
                "P(J1)": f"{prob_p1:.1%}",
                "P(J2)": f"{prob_p2:.1%}",
                "Favori": fav,
                "Confiance": f"{fav_prob:.1%}",
                "WTT#J1": int(p1_row["wtt_rank"]) if int(p1_row["wtt_rank"]) < 9999 else "-",
                "WTT#J2": int(p2_row["wtt_rank"]) if int(p2_row["wtt_rank"]) < 9999 else "-",
            })

    # Afficher
    if not predictions:
        logger.warning("Aucun match avec confiance suffisante")
        return

    df = pd.DataFrame(predictions).sort_values("Date")
    print(f"\n{'='*100}")
    print(f"  PREDICTIONS {args.model.upper()} — Matchs WTT/Internationaux ({len(df)} matchs)")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}")

    if not_found:
        logger.warning(f"{len(not_found)} matchs ignorés (joueurs non trouvés en DB) :")
        for m in not_found[:10]:
            logger.warning(f"  - {m}")


if __name__ == "__main__":
    main()
