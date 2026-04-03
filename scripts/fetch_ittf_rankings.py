"""
Synchronise la table ittf_rankings depuis l'API WTT officielle.

L'API WTT retourne RankingPosition = rang ITTF mondial (WTT = ITTF depuis la fusion 2021).
On réutilise le même endpoint que fetch_wtt_rankings.py.

Usage :
    python scripts/fetch_ittf_rankings.py
    python scripts/fetch_ittf_rankings.py --dry-run
    python scripts/fetch_ittf_rankings.py --gender W   # femmes
    python scripts/fetch_ittf_rankings.py --gender both  # hommes + femmes (défaut)
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from sqlalchemy import text

from src.database.db import engine, init_db

try:
    from curl_cffi import requests as cffi_requests

    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

API_URLS = {
    "M": "https://wtt-web-frontdoor-withoutcache-cqakg0andqf5hchn.a01.azurefd.net/ranking/SEN_SINGLES.json",
    "W": "https://wtt-web-frontdoor-withoutcache-cqakg0andqf5hchn.a01.azurefd.net/ranking/SEN_SINGLES_W.json",
}

_CFFI_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://worldtabletennis.com/",
    "Origin": "https://worldtabletennis.com",
}


def fetch_rankings(gender: str) -> list[dict]:
    if not _HAS_CURL_CFFI:
        raise RuntimeError("curl-cffi required. Install: pip install curl-cffi")
    url = API_URLS[gender]
    session = cffi_requests.Session(impersonate="chrome120")
    session.headers.update(_CFFI_HEADERS)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json().get("Result", [])


def _iso_week_monday(year: int, week: int) -> date:
    return date.fromisocalendar(year, week, 1)


def insert_ittf_rankings(
    records: list[dict], dry_run: bool = False
) -> tuple[int, int, int]:
    """Insert into ittf_rankings. Returns (inserted, skipped, no_player)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT ittf_id, id FROM players WHERE ittf_id IS NOT NULL")
        ).fetchall()
    player_map = {str(r[0]): r[1] for r in rows}

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
            points = (
                float(rec["RankingPointsYTD"]) if rec.get("RankingPointsYTD") else 0.0
            )
            year = int(rec["RankingYear"])
            week = int(rec["RankingWeek"])
            snapshot_date = _iso_week_monday(year, week)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Parse error IttfId={ittf_id}: {e}")
            no_player += 1
            continue

        rows_to_insert.append(
            {
                "player_id": player_id,
                "rank": rank,
                "points": points,
                "snapshot_date": str(snapshot_date),
            }
        )

    if dry_run:
        logger.info(f"[DRY-RUN] {len(rows_to_insert)} ittf_rankings would be inserted")
        for row in rows_to_insert[:5]:
            logger.info(
                f"  rank={row['rank']} player_id={row['player_id']} date={row['snapshot_date']}"
            )
        return len(rows_to_insert), 0, no_player

    stmt = text("""
        INSERT INTO ittf_rankings (player_id, rank, points, snapshot_date)
        VALUES (:player_id, :rank, :points, :snapshot_date)
        ON CONFLICT(player_id, snapshot_date) DO UPDATE SET rank=excluded.rank, points=excluded.points
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
    parser = argparse.ArgumentParser(description="Sync ittf_rankings from WTT API")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gender", choices=["M", "W", "both"], default="both")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    init_db()

    genders = ["M", "W"] if args.gender == "both" else [args.gender]

    for g in genders:
        label = "Men" if g == "M" else "Women"
        logger.info(f"Fetching ITTF rankings ({label})...")
        try:
            records = fetch_rankings(g)
        except Exception as e:
            logger.error(f"Failed to fetch {label}: {e}")
            continue

        logger.info(f"{len(records)} players retrieved")
        if records:
            s = records[0]
            logger.info(
                f"Snapshot: week {s.get('RankingWeek')}/{s.get('RankingYear')} | #1: {s.get('PlayerName')}"
            )

        ins, skip, nop = insert_ittf_rankings(records, dry_run=args.dry_run)
        logger.info(f"[{label}] inserted={ins} skipped={skip} no_player={nop}")


if __name__ == "__main__":
    main()
