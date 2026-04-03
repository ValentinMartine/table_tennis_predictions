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

from src.database.db import engine, get_session
from src.database.models import BettingRecord
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.features.match_features import build_single_match_features
from src.features.tournament_projections import TournamentSimulator

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
    "WTT Grand Smash", "WTT Finals", "WTT Cup Finals",
    "World Cup", "World Championships", "European Championships",
    "Asian Championships", "Olympic Games", "Team World Cup",
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


def fetch_upcoming_matches(session=None, days: int = 7, all_leagues: bool = False) -> list[dict]:
    """Retourne les matchs non-commencés sur les `days` prochains jours."""
    if not _HAS_CURL_CFFI:
        logger.error("curl-cffi requis. pip install curl-cffi")
        return []

    patterns = TARGET_PATTERNS_ALL if all_leagues else TARGET_PATTERNS_INTL

    # On utilise la session passée en argument ou on en recrée une si besoin (dashboard)
    if session is None:
        session = cffi_requests.Session(impersonate="chrome120")
        session.headers.update(_CFFI_HEADERS)

    results = []
    seen_ids = set()
    all_tournament_names: set[str] = set()
    for d in range(days + 1):
        day = date.today() + timedelta(days=d)
        data = _sfetch(session, f"{API_BASE}/sport/table-tennis/scheduled-events/{day}")
        for ev in data.get("events", []):
            eid = ev.get("id")
            if not eid or eid in seen_ids:
                continue
            
            status = ev.get("status", {}).get("type", "")
            if status not in ("notstarted", "inprogress"):
                continue
            t = ev.get("tournament", {})
            tournament_name = t.get("name", "") or t.get("uniqueTournament", {}).get("name", "")
            all_tournament_names.add(tournament_name)
            if not any(p.lower() in tournament_name.lower() for p in patterns):
                continue
            
            round_name = ev.get("roundInfo", {}).get("name", "")
            group_name = ev.get("roundInfo", {}).get("caption", "")
            
            # Fallback for World Cup groups in tournament name
            if not group_name and "Group" in tournament_name:
                import re
                m_group = re.search(r"Group (\d+)", tournament_name)
                if m_group:
                    group_name = f"Group {m_group.group(1)}"
            if not round_name and group_name:
                round_name = group_name

            seen_ids.add(eid)
            results.append({
                "event_id": eid,
                "tournament": tournament_name,
                "p1_name": ev.get("homeTeam", {}).get("name", ""),
                "p2_name": ev.get("awayTeam", {}).get("name", ""),
                "start_time": ev.get("startTimestamp", 0),
                "status": status,
                "round_name": round_name,
                "group_name": group_name,
            })
        import time; time.sleep(0.5)

    logger.info(f"Tous les tournois détectés sur Sofascore ({days} jours) :")
    for name in sorted(all_tournament_names):
        matched = any(p.lower() in name.lower() for p in patterns)
        logger.info(f"  {'✓' if matched else '✗'} {name}")

    scope = "toutes compétitions" if all_leagues else "WTT/internationaux"
    logger.info(f"{len(results)} matchs {scope} à venir (prochains {days} jours)")
    return results


def fetch_odds_for_event(session, event_id: int) -> tuple[float, float]:
    """Récupère les cotes actuelles pour un match Sofascore."""
    url = f"{API_BASE}/event/{event_id}/odds/1/all"
    data = _sfetch(session, url)
    markets = data.get("markets", [])
    winner_market = next((m for m in markets if m.get("marketName") == "Full time"), markets[0] if markets else None)
    
    if winner_market:
        choices = winner_market.get("choices", [])
        if len(choices) >= 2:
            o1_frac = choices[0].get("fractionalValue")
            o2_frac = choices[1].get("fractionalValue")
            
            def f_to_d(f):
                try:
                    n, d = f.split('/')
                    return round(float(n)/float(d) + 1.0, 3)
                except: return 0.0
            
            return f_to_d(o1_frac), f_to_d(o2_frac)
    return 0.0, 0.0


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
                              p1_ittf: int, p2_ittf: int,
                              p1_wtt: int = 9999, p2_wtt: int = 9999) -> pd.DataFrame:
    feats = build_single_match_features(p1_id, p2_id, p1_ittf, p2_ittf, p1_wtt, p2_wtt)
    return pd.DataFrame([feats])


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
    session = cffi_requests.Session(impersonate="chrome120")
    session.headers.update(_CFFI_HEADERS)
    
    upcoming = fetch_upcoming_matches(session, days=args.days, all_leagues=args.all_leagues)
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
            # --- NOUVEAU : Enregistrement Paper Betting ---
            o1, o2 = fetch_odds_for_event(session, ev["event_id"])
            
            # On calcule l'edge contre les cotes réelles si dispos
            edge_1 = prob_p1 - (1/o1) if o1 > 0 else 0
            edge_2 = prob_p2 - (1/o2) if o2 > 0 else 0
            
            # Enregistrement si Edge > 1% (on conserve même les petits edges pour l'audit)
            if (edge_1 > 0.01 or edge_2 > 0.01) and ev["status"] == "notstarted":
                # On cherche s'il n'existe pas déjà un match en DB pour lier l'ID si possible
                # Mais souvent le match n'est pas encore créé par le scraper principal.
                # On enregistre le pari en 'PENDING'.
                # Paramètres Kelly
                BANKROLL = 1000.0
                FRACTIONAL_KELLY = 0.1
                
                bet_p = 1 if edge_1 >= edge_2 else 2
                edge = max(edge_1, edge_2)
                odds = o1 if bet_p == 1 else o2
                prob = prob_p1 if bet_p == 1 else prob_p2
                
                # Kelly Formula: f = (p*o - 1) / (o - 1)
                f_star = (prob * odds - 1) / (odds - 1)
                stake = max(2.0, round(BANKROLL * f_star * FRACTIONAL_KELLY, 2))
                
                # Recherche du match_id dans la table 'matches' par external_id (sfs_{id})
                with get_session() as db_session:
                    exists = db_session.execute(
                        text("SELECT id FROM matches WHERE external_id = :ext"),
                        {"ext": f"sfs_{ev['event_id']}"}
                    ).fetchone()
                    
                    if exists:
                        mid = exists[0]
                        # Vérifier si on n'a pas déjà un pari pour ce match
                        already_bet = db_session.execute(
                            text("SELECT id FROM betting_records WHERE match_id = :mid AND is_paper = 1"),
                            {"mid": mid}
                        ).fetchone()
                        
                        if not already_bet:
                            rec = BettingRecord(
                                match_id=mid, bet_player=bet_p, stake=stake, odds=odds,
                                predicted_prob=round(prob, 3), model_edge=round(edge, 3),
                                result="PENDING", is_paper=True, placed_at=pd.Timestamp.now()
                            )
                            db_session.add(rec)
                            logger.info(f"   [Paper Bet] Enregistré PENDING: {ev['p1_name']} vs {ev['p2_name']} (@{odds}) | Stake: {stake}€ (Kelly)")

            predictions.append({
                "Date": start,
                "Tournoi": ev["tournament"][:30],
                "J1": ev["p1_name"],
                "J2": ev["p2_name"],
                "P(J1)": f"{prob_p1:.1%}",
                "P(J2)": f"{prob_p2:.1%}",
                "Favori": fav,
                "Confiance": f"{fav_prob:.1%}",
                "Edge": f"{max(edge_1, edge_2):+.1%}" if (o1 > 0 and o2 > 0) else "-",
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
    
    # --- SIMULATION DE TOURNOI (NOUVEAU) ---
    logger.info("Simulation des phases finales (World Cup / Brackets)...")
    sim = TournamentSimulator(model, player_map)
    leaders = sim.simulate_world_cup_groups(upcoming)
    projections = sim.project_knockout_stage(leaders)
    
    if projections:
        df_proj = pd.DataFrame(projections)
        df_proj.to_csv("data/tournament_projections.csv", index=False)
        logger.info(f"Projections de tournois sauvegardées ({len(projections)} rounds projetés)")
        print(f"\nProjections (Knockout Stage) :\n{df_proj.to_string(index=False)}")

    # Save to CSV for easy reading
    df.to_csv("data/upcoming_predictions.csv", index=False)
    logger.info(f"Prédictions sauvegardées dans data/upcoming_predictions.csv")

    if not_found:
        logger.warning(f"{len(not_found)} matchs ignorés (joueurs non trouvés en DB) :")
        for m in not_found[:10]:
            logger.warning(f"  - {m}")


if __name__ == "__main__":
    main()
