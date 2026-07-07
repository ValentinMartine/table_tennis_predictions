"""
Backfill table-tennis odds from OddsPapi into existing matches.

This avoids BetsAPI and uses:
- /v4/sports to resolve table tennis
- /v4/fixtures to find finished fixtures with odds
- /v4/odds to extract winner prices

It updates matches already present in the local database. Matching priority:
1. externalProviders.sofascoreId -> matches.external_id = sfs_{id}
2. participant names + start date fuzzy match
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text

from src.database.db import engine
from src.scraping.oddspapi import (
    DEFAULT_BOOKMAKERS,
    extract_home_away_odds,
    fetch_fixtures,
    fetch_odds,
    is_usable_secret,
    resolve_table_tennis_sport_id,
)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return " ".join(value.split())


def _similarity(left: str, right: str) -> float:
    a = set(_normalize(left).split())
    b = set(_normalize(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def _event_sofascore_external_id(event: dict) -> str | None:
    providers = event.get("externalProviders") or {}
    value = providers.get("sofascoreId")
    return f"sfs_{value}" if value else None


def _as_naive_utc(value) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _build_match_index(conn, start: datetime, end: datetime, window_hours: int) -> dict:
    lo = start.replace(tzinfo=None) - timedelta(hours=window_hours)
    hi = end.replace(tzinfo=None) + timedelta(hours=window_hours)
    rows = conn.execute(
        text("""
            SELECT m.id, m.external_id, m.odds_p1 IS NOT NULL, m.played_at, p1.name, p2.name
            FROM matches m
            JOIN players p1 ON p1.id = m.player1_id
            JOIN players p2 ON p2.id = m.player2_id
            WHERE m.played_at BETWEEN :lo AND :hi
        """),
        {"lo": lo, "hi": hi},
    ).fetchall()

    by_external: dict[str, tuple[int, bool]] = {}
    by_date: dict[str, list[tuple[int, bool, datetime, str, str]]] = {}
    for row in rows:
        played_at = _as_naive_utc(row[3])
        if played_at is None:
            continue
        record = (int(row[0]), bool(row[2]), played_at, row[4], row[5])
        if row[1]:
            by_external[str(row[1])] = (int(row[0]), bool(row[2]))
        for offset in (-1, 0, 1):
            key = (played_at.date() + timedelta(days=offset)).isoformat()
            by_date.setdefault(key, []).append(record)
    logger.info(f"Index local: {len(rows)} matchs charges pour appariement OddsPapi")
    return {"external": by_external, "date": by_date}


def _find_match(index: dict, event: dict, window_hours: int, min_similarity: float) -> tuple[int, bool] | None:
    external_id = _event_sofascore_external_id(event)
    if external_id:
        row = index["external"].get(external_id)
        if row:
            return row

    start = _parse_dt(event.get("startTime"))
    if start is None:
        return None
    p1 = event.get("participant1Name", "")
    p2 = event.get("participant2Name", "")
    start_naive = start.astimezone(timezone.utc).replace(tzinfo=None)
    lo = start_naive - timedelta(hours=window_hours)
    hi = start_naive + timedelta(hours=window_hours)

    best: tuple[int, bool, float] | None = None
    for row in index["date"].get(start_naive.date().isoformat(), []):
        if not (lo <= row[2] <= hi):
            continue
        direct = _similarity(p1, row[3]) + _similarity(p2, row[4])
        reverse = _similarity(p1, row[4]) + _similarity(p2, row[3])
        score = max(direct, reverse) / 2
        if score >= min_similarity and (best is None or score > best[2]):
            best = (int(row[0]), bool(row[1]), score)
    if not best:
        return None
    return best[0], best[1]


def _record_snapshot(conn, event: dict, odds_p1: float, odds_p2: float, source: str, match_id: int | None) -> None:
    conn.execute(
        text("""
            INSERT OR IGNORE INTO odds_snapshots
            (match_id, match_external_id, player1_name, player2_name, source,
             odds_p1, odds_p2, match_start, snapshot_type, captured_at)
            VALUES (:match_id, :external_id, :p1, :p2, :source,
                    :odds_p1, :odds_p2, :match_start, 'historical', :captured_at)
        """),
        {
            "match_id": match_id,
            "external_id": event.get("fixtureId"),
            "p1": event.get("participant1Name"),
            "p2": event.get("participant2Name"),
            "source": source,
            "odds_p1": odds_p1,
            "odds_p2": odds_p2,
            "match_start": _parse_dt(event.get("startTime")),
            "captured_at": datetime.now(timezone.utc).replace(tzinfo=None),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD, exclusive")
    parser.add_argument("--bookmakers", default=DEFAULT_BOOKMAKERS)
    parser.add_argument("--window-hours", type=int, default=18)
    parser.add_argument("--min-similarity", type=float, default=0.78)
    parser.add_argument("--delay", type=float, default=0.7)
    parser.add_argument("--day-delay", type=float, default=1.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(".env")
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    api_key = os.getenv("ODDSPAPI_KEY", "")
    if not is_usable_secret(api_key):
        logger.error("ODDSPAPI_KEY manquant ou placeholder")
        return

    sport_id = resolve_table_tennis_sport_id(api_key)
    if sport_id is None:
        logger.error("Tennis de table introuvable dans la couverture OddsPapi de ce compte")
        return
    logger.info(f"OddsPapi sportId tennis de table = {sport_id}")

    start = _parse_date(args.from_date)
    end = _parse_date(args.to_date)
    cursor = start
    scanned = covered = matched = updated = skipped_existing = 0

    # Index construit une fois avec une connexion en lecture (fermée ensuite).
    with engine.connect() as conn:
        match_index = _build_match_index(conn, start, end, args.window_hours)

    # Une transaction PAR JOUR : la progression est committée au fil de l'eau,
    # donc un arrêt/timeout ne perd pas tout et le backfill est reprenable.
    while cursor < end:
        day_end = min(cursor + timedelta(days=1), end)
        fixtures = fetch_fixtures(
            api_key,
            sport_id,
            cursor,
            day_end,
            status_id=2,
            bookmakers=args.bookmakers,
            has_odds=None,
        )
        logger.info(f"{cursor.date()}: {len(fixtures)} fixtures terminees")

        with engine.begin() as conn:
            for event in fixtures:
                if args.limit and scanned >= args.limit:
                    break
                scanned += 1
                fixture_id = event.get("fixtureId")
                if not fixture_id:
                    continue

                found = _find_match(match_index, event, args.window_hours, args.min_similarity)
                if not found:
                    continue
                match_id, already_has_odds = found
                matched += 1
                if already_has_odds:
                    skipped_existing += 1
                    continue

                odds_payload = fetch_odds(api_key, fixture_id, bookmakers=args.bookmakers)
                time.sleep(args.delay)
                if not odds_payload:
                    continue
                odds_p1, odds_p2, source = extract_home_away_odds(odds_payload)
                if not odds_p1 or not odds_p2:
                    continue
                covered += 1

                if not args.dry_run:
                    _record_snapshot(conn, event, odds_p1, odds_p2, f"oddspapi:{source or 'unknown'}", match_id)
                    conn.execute(
                        text("""
                            UPDATE matches
                            SET odds_p1 = :odds_p1,
                                odds_p2 = :odds_p2,
                                odds_source = :source
                            WHERE id = :match_id
                        """),
                        {
                            "odds_p1": odds_p1,
                            "odds_p2": odds_p2,
                            "source": f"oddspapi:{source or 'unknown'}",
                            "match_id": match_id,
                        },
                    )
                updated += 1

        if args.limit and scanned >= args.limit:
            break
        cursor = day_end
        time.sleep(args.day_delay)

    logger.info(
        f"{'[DRY-RUN] ' if args.dry_run else ''}"
        f"scannes={scanned}, avec_cotes={covered}, apparies={matched}, "
        f"deja_cotes={skipped_existing}, maj={updated}"
    )


if __name__ == "__main__":
    main()
