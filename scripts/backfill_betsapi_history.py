"""
Importe un historique BetsAPI avec cotes pré-match quand elles sont disponibles.

Ce script complète la base locale avec des matchs BetsAPI distincts des matchs Sofascore.
Il ne remplace pas les lignes existantes et respecte la déduplication du pipeline.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import yaml
from dotenv import load_dotenv
from loguru import logger

from src.scraping.base import RawMatch
from src.scraping.pipeline import insert_matches

SPORT_ID = 18
BASE_V1 = "https://api.betsapi.com/v1"
BASE_V2 = "https://api.betsapi.com/v2"


def _get_json(url: str, params: dict, delay: float) -> dict:
    time.sleep(delay)
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _parse_score(score: str) -> tuple[int, int] | None:
    try:
        p1, p2 = str(score).replace("-", ":").split(":")[:2]
        return int(p1), int(p2)
    except (ValueError, AttributeError):
        return None


def _fetch_odds(event_id: str, token: str, delay: float) -> tuple[float | None, float | None]:
    try:
        data = _get_json(
            f"{BASE_V2}/event/odds/summary",
            {"event_id": event_id, "token": token},
            delay,
        )
    except requests.RequestException:
        return None, None

    if data.get("success") != 1:
        return None, None

    market = (data.get("results", {}) or {}).get("1_1", {})
    odds_p1 = odds_p2 = None
    for outcome in market.get("odds", []):
        try:
            value = float(outcome.get("odds", 0) or 0)
        except (TypeError, ValueError):
            continue
        if outcome.get("id") == "1" and value > 1:
            odds_p1 = value
        elif outcome.get("id") == "2" and value > 1:
            odds_p2 = value
    return odds_p1, odds_p2


def _parse_event(event: dict, competition_id: str, token: str, delay: float) -> RawMatch | None:
    event_id = str(event.get("id") or "")
    home = event.get("home") or {}
    away = event.get("away") or {}
    score = _parse_score(event.get("ss", ""))
    if not event_id or not home.get("name") or not away.get("name") or not score:
        return None

    try:
        played_at = datetime.utcfromtimestamp(int(event.get("time")))
    except (TypeError, ValueError):
        return None

    odds_p1, odds_p2 = _fetch_odds(event_id, token, delay)
    if odds_p1 is None or odds_p2 is None:
        return None

    score_p1, score_p2 = score
    if score_p1 == score_p2:
        return None

    return RawMatch(
        external_id=f"bapi_{event_id}",
        competition_id=competition_id,
        player1_name=home.get("name", ""),
        player2_name=away.get("name", ""),
        player1_country=home.get("cc"),
        player2_country=away.get("cc"),
        played_at=played_at,
        winner=1 if score_p1 > score_p2 else 2,
        score_p1=score_p1,
        score_p2=score_p2,
        sets_detail=str(event.get("scores") or ""),
        odds_p1=odds_p1,
        odds_p2=odds_p2,
        odds_source="betsapi",
    )


def _iter_days(start: datetime, end: datetime):
    day = start.date()
    last = end.date()
    while day <= last:
        yield day
        day += timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", default="2024-01-01")
    parser.add_argument("--to-date", default=datetime.utcnow().date().isoformat())
    parser.add_argument("--competition-id", default="betsapi_table_tennis")
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--max-days", type=int, default=0, help="0 = toute la plage")
    parser.add_argument("--max-events", type=int, default=0, help="0 = tous les événements")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    import os
    token = os.getenv("BETSAPI_TOKEN", "")
    if not token:
        logger.error("BETSAPI_TOKEN manquant")
        return 1

    with open("config/settings.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    start = datetime.fromisoformat(args.from_date)
    end = datetime.fromisoformat(args.to_date)
    raw_matches: list[RawMatch] = []
    days_done = 0
    events_seen = 0

    for day in _iter_days(start, end):
        if args.max_days and days_done >= args.max_days:
            break
        days_done += 1
        page = 1
        day_events = 0
        while True:
            try:
                data = _get_json(
                    f"{BASE_V1}/events/ended",
                    {
                        "sport_id": SPORT_ID,
                        "day": day.strftime("%Y%m%d"),
                        "page": page,
                        "token": token,
                    },
                    args.delay,
                )
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", "?")
                logger.warning(f"{day} page {page}: erreur HTTP {status}")
                break

            if data.get("success") != 1:
                logger.warning(f"{day} page {page}: {data.get('error', 'BetsAPI error')}")
                break

            batch = data.get("results", [])
            if not batch:
                break

            for event in batch:
                events_seen += 1
                day_events += 1
                if args.max_events and events_seen > args.max_events:
                    break
                match = _parse_event(event, args.competition_id, token, args.delay)
                if match:
                    raw_matches.append(match)

            if args.max_events and events_seen >= args.max_events:
                break

            pager = data.get("pager", {})
            if page >= pager.get("total_pages", 1):
                break
            page += 1

        if day_events:
            logger.info(f"{day}: {day_events} événements, {len(raw_matches)} avec cotes cumulés")
        if args.max_events and events_seen >= args.max_events:
            break

    if args.dry_run:
        logger.info(f"[DRY-RUN] {len(raw_matches)} matchs BetsAPI cotés prêts à insérer")
        return 0

    inserted = insert_matches(raw_matches, config)
    logger.info(f"{inserted} nouveaux matchs BetsAPI insérés ({len(raw_matches)} cotés récupérés)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
