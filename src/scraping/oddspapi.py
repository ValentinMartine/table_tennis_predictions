"""
OddsPapi connector for table-tennis odds.

Docs:
- https://oddspapi.io/us/docs/get-sports
- https://oddspapi.io/us/docs/get-fixtures
- https://oddspapi.io/us/docs/get-odds
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from loguru import logger

BASE_URL = "https://api.oddspapi.io/v4"
DEFAULT_BOOKMAKERS = "pinnacle,bet365,1xbet"


def is_usable_secret(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "test", "placeholder", "xxx"} and not normalized.startswith("your_")


def _get(endpoint: str, api_key: str, **params: Any) -> Any:
    params = {k: v for k, v in params.items() if v is not None}
    params["apiKey"] = api_key
    response = None
    for attempt in range(4):
        try:
            response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        except requests.RequestException as exc:
            wait = 2 + attempt * 2
            logger.warning(f"OddsPapi {endpoint}: network error {type(exc).__name__}, retry in {wait}s")
            time.sleep(wait)
            continue
        if response.status_code != 429:
            break
        wait = 2 + attempt * 2
        logger.warning(f"OddsPapi {endpoint}: rate limit, retry in {wait}s")
        time.sleep(wait)
    if response is None:
        return None
    if response.status_code == 404 and endpoint == "fixtures":
        return []
    if response.status_code in {401, 403}:
        logger.warning(f"OddsPapi {endpoint}: invalid or unauthorized key ({response.status_code})")
        return None
    if response.status_code == 429:
        logger.warning("OddsPapi: quota or rate limit reached")
        return None
    if not response.ok:
        logger.warning(f"OddsPapi {endpoint}: HTTP {response.status_code} - {response.text[:160]}")
        return None
    return response.json()


def resolve_table_tennis_sport_id(api_key: str) -> int | None:
    sports = _get("sports", api_key, language="en")
    if not isinstance(sports, list):
        return None
    for sport in sports:
        haystack = f"{sport.get('slug', '')} {sport.get('sportName', '')}".lower()
        if "table" in haystack and "tennis" in haystack:
            return int(sport["sportId"])
    return None


def fetch_fixtures(
    api_key: str,
    sport_id: int,
    start: datetime,
    end: datetime,
    status_id: int | None,
    bookmakers: str = DEFAULT_BOOKMAKERS,
    has_odds: bool | None = True,
) -> list[dict]:
    data = _get(
        "fixtures",
        api_key,
        sportId=sport_id,
        **{
            "from": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "to": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        statusId=status_id,
        hasOdds=str(has_odds).lower() if has_odds is not None else None,
        bookmakers=bookmakers,
        language="en",
    )
    return data if isinstance(data, list) else []


def fetch_odds(api_key: str, fixture_id: str, bookmakers: str = DEFAULT_BOOKMAKERS) -> dict | None:
    data = _get(
        "odds",
        api_key,
        fixtureId=fixture_id,
        bookmakers=bookmakers,
        oddsFormat="decimal",
        language="en",
        verbosity=3,
    )
    return data if isinstance(data, dict) else None


def _iter_prices(odds_payload: dict) -> list[tuple[str, float, str]]:
    prices: list[tuple[str, float, str]] = []
    for bookmaker_name, bookmaker in (odds_payload.get("bookmakerOdds") or {}).items():
        for market in (bookmaker.get("markets") or {}).values():
            market_prices: list[tuple[str, float, str]] = []
            for outcome in (market.get("outcomes") or {}).values():
                for player in (outcome.get("players") or {}).values():
                    candidates = player if isinstance(player, list) else [player]
                    for candidate in candidates:
                        try:
                            price = float(candidate.get("price") or 0)
                        except (AttributeError, TypeError, ValueError):
                            continue
                        if price <= 1:
                            continue
                        outcome_id = str(candidate.get("bookmakerOutcomeId") or "").lower()
                        market_prices.append((outcome_id, price, bookmaker_name))

            labelled = [
                item for item in market_prices
                if item[0] in {"home", "away", "1", "2", "participant1", "participant2"}
            ]
            if labelled:
                prices.extend(labelled)
                continue

            if len(market_prices) == 2:
                prices.append(("home", market_prices[0][1], market_prices[0][2]))
                prices.append(("away", market_prices[1][1], market_prices[1][2]))
    return prices


def extract_home_away_odds(odds_payload: dict) -> tuple[float | None, float | None, str | None]:
    home: tuple[float, str] | None = None
    away: tuple[float, str] | None = None
    for outcome_id, price, bookmaker in _iter_prices(odds_payload):
        if outcome_id in {"home", "1", "participant1"}:
            if home is None or price > home[0]:
                home = (price, bookmaker)
        elif outcome_id in {"away", "2", "participant2"}:
            if away is None or price > away[0]:
                away = (price, bookmaker)
    if not home or not away:
        return None, None, None
    source = home[1] if home[1] == away[1] else "oddspapi_best"
    return home[0], away[0], source


def get_table_tennis_odds_oddspapi(
    api_key: str | None = None,
    days_ahead: int = 7,
    bookmakers: str = DEFAULT_BOOKMAKERS,
    delay: float = 0.6,
) -> list[dict]:
    api_key = api_key or os.getenv("ODDSPAPI_KEY", "")
    if not is_usable_secret(api_key):
        logger.debug("ODDSPAPI_KEY absent - OddsPapi skipped")
        return []

    sport_id = resolve_table_tennis_sport_id(api_key)
    if sport_id is None:
        logger.warning("OddsPapi: table tennis not found in account coverage")
        return []

    now = datetime.now(timezone.utc)
    events = fetch_fixtures(
        api_key,
        sport_id,
        now,
        now + timedelta(days=days_ahead),
        status_id=0,
        bookmakers=bookmakers,
        has_odds=True,
    )

    results: list[dict] = []
    for event in events:
        fixture_id = event.get("fixtureId")
        if not fixture_id:
            continue
        odds_payload = fetch_odds(api_key, fixture_id, bookmakers=bookmakers)
        time.sleep(delay)
        if not odds_payload:
            continue
        odds_home, odds_away, source = extract_home_away_odds(odds_payload)
        if odds_home and odds_away:
            results.append({
                "home": event.get("participant1Name", ""),
                "away": event.get("participant2Name", ""),
                "commence_time": _parse_dt(event.get("startTime")),
                "odds_home": odds_home,
                "odds_away": odds_away,
                "bookmaker": f"OddsPapi:{source or 'unknown'}",
                "fixture_id": fixture_id,
                "external_providers": event.get("externalProviders") or {},
            })

    logger.info(f"OddsPapi: {len(results)} table-tennis matches with odds")
    return results


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
