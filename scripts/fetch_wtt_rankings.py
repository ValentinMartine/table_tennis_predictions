"""
Récupère le classement WTT officiel (SEN_SINGLES) et l'insère en DB.

Endpoint : GET https://wtt-web-frontdoor-withoutcache-cqakg0andqf5hchn.a01.azurefd.net/ranking/SEN_SINGLES.json
Réponse  : {"Result": [{"IttfId": "137237", "PlayerName": "LIN Shidong", "RankingPosition": "1",
                         "RankingPointsYTD": "5905", "RankingYear": "2026", "RankingWeek": "13", ...}]}

Usage :
    python scripts/fetch_wtt_rankings.py
    python scripts/fetch_wtt_rankings.py --dry-run
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.database.db import engine, init_db

try:
    from curl_cffi import requests as cffi_requests
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

API_URL = "https://wtt-web-frontdoor-withoutcache-cqakg0andqf5hchn.a01.azurefd.net/ranking/SEN_SINGLES.json"

_CFFI_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://worldtabletennis.com/",
    "Origin": "https://worldtabletennis.com",
}


def fetch_wtt_rankings() -> list[dict]:
    """Télécharge le classement WTT depuis l'API officielle."""
    if not _HAS_CURL_CFFI:
        raise RuntimeError("curl-cffi requis. Installez : pip install curl-cffi")

    session = cffi_requests.Session(impersonate="chrome120")
    session.headers.update(_CFFI_HEADERS)
    resp = session.get(API_URL, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("Result", [])


def _iso_week_monday(year: int, week: int) -> date:
    """Retourne le lundi de la semaine ISO (year, week)."""
    return date.fromisocalendar(year, week, 1)


def load_player_ittf_map() -> dict[str, int]:
    """Retourne {ittf_id: player_id} depuis la DB."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ittf_id, id FROM players WHERE ittf_id IS NOT NULL")).fetchall()
    return {str(r[0]): r[1] for r in rows}


def insert_rankings(records: list[dict], dry_run: bool = False) -> tuple[int, int, int]:
    """Insère les rankings WTT en DB. Retourne (insérés, déjà présents, sans joueur)."""
    player_map = load_player_ittf_map()
    inserted = skipped = no_player = 0

    rows_to_insert = []
    for rec in records:
        ittf_id = str(rec.get("IttfId", "")).strip()
        if not ittf_id:
            no_player += 1
            continue

        player_id = player_map.get(ittf_id)
        if player_id is None:
            no_player += 1
            continue

        try:
            rank = int(rec["RankingPosition"])
            points_ytd = float(rec["RankingPointsYTD"]) if rec.get("RankingPointsYTD") else None
            year = int(rec["RankingYear"])
            week = int(rec["RankingWeek"])
            snapshot_date = _iso_week_monday(year, week)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Parsing error pour IttfId={ittf_id}: {e}")
            no_player += 1
            continue

        rows_to_insert.append({
            "player_id": player_id,
            "rank": rank,
            "points_ytd": points_ytd,
            "ranking_year": year,
            "ranking_week": week,
            "snapshot_date": snapshot_date,
        })

    if dry_run:
        logger.info(f"[DRY-RUN] {len(rows_to_insert)} rankings seraient insérés")
        for row in rows_to_insert[:5]:
            logger.info(f"  rank={row['rank']} player_id={row['player_id']} points={row['points_ytd']} date={row['snapshot_date']}")
        return len(rows_to_insert), 0, no_player

    stmt = text("""
        INSERT INTO wtt_rankings (player_id, rank, points_ytd, ranking_year, ranking_week, snapshot_date)
        VALUES (:player_id, :rank, :points_ytd, :ranking_year, :ranking_week, :snapshot_date)
        ON CONFLICT(player_id, snapshot_date) DO NOTHING
    """)

    with engine.begin() as conn:
        for row in rows_to_insert:
            result = conn.execute(stmt, row)
            if result.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

    return inserted, skipped, no_player


def main():
    parser = argparse.ArgumentParser(description="Fetch WTT rankings (SEN_SINGLES)")
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans écrire en DB")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    init_db()

    logger.info("Téléchargement du classement WTT...")
    try:
        records = fetch_wtt_rankings()
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement : {e}")
        sys.exit(1)

    logger.info(f"{len(records)} joueurs récupérés depuis l'API WTT")
    if records:
        sample = records[0]
        logger.info(
            f"Snapshot : semaine {sample.get('RankingWeek')}/{sample.get('RankingYear')} "
            f"| #1 : {sample.get('PlayerName')} ({sample.get('RankingPointsYTD')} pts)"
        )

    inserted, skipped, no_player = insert_rankings(records, dry_run=args.dry_run)

    logger.info(
        f"\n=== RÉSULTAT ===\n"
        f"  Insérés         : {inserted}\n"
        f"  Déjà présents   : {skipped}\n"
        f"  Sans joueur DB  : {no_player}"
    )


if __name__ == "__main__":
    main()
