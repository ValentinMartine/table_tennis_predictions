"""
Récupère les cotes Sofascore pour les matchs val/test sans cotes.

Endpoint : GET https://api.sofascore.com/api/v1/event/{event_id}/odds/1/all
Réponse  : {"markets": [{"marketName": "Full time", "choices": [
               {"name": "1", "fractionalValue": "1/2", ...},
               {"name": "2", "fractionalValue": "3/1", ...}
            ]}]}

Usage :
    python scripts/fetch_sofascore_odds.py
    python scripts/fetch_sofascore_odds.py --since 2024-01-01 --delay 1.5
    python scripts/fetch_sofascore_odds.py --dry-run   # affiche sans écrire en DB
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.database.db import engine

try:
    from curl_cffi import requests as cffi_requests

    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

API_BASE = "https://api.sofascore.com/api/v1"

_CFFI_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "Cache-Control": "no-cache",
}


def fractional_to_decimal(frac: str) -> float | None:
    """Convertit '7/1' → 8.0, '1/16' → 1.0625, '1/1' → 2.0."""
    try:
        parts = frac.strip().split("/")
        if len(parts) != 2:
            return None
        num, den = float(parts[0]), float(parts[1])
        if den == 0:
            return None
        return round(num / den + 1.0, 4)
    except (ValueError, AttributeError):
        return None


def parse_odds_response(data: dict) -> tuple[float | None, float | None]:
    """
    Extrait (odds_p1, odds_p2) depuis la réponse de l'endpoint odds.
    Cherche le marché 'Full time' ou le premier marché disponible.
    Retourne (None, None) si indisponible.
    """
    markets = data.get("markets", [])
    if not markets:
        return None, None

    # Préférer le marché "Full time"
    target = None
    for m in markets:
        if "full time" in (m.get("marketName") or "").lower():
            target = m
            break
    if target is None:
        target = markets[0]

    choices = target.get("choices", [])
    odds_map: dict[str, float] = {}
    for c in choices:
        name = str(c.get("name", "")).strip()
        frac = c.get("fractionalValue") or c.get("initialFractionalValue")
        if name and frac:
            val = fractional_to_decimal(frac)
            if val:
                odds_map[name] = val

    odds_p1 = odds_map.get("1")
    odds_p2 = odds_map.get("2")
    return odds_p1, odds_p2


def fetch_matches_without_odds(since: str, priority_max: int = 99) -> pd.DataFrame:
    if priority_max < 99:
        query = text("""
            SELECT m.id, m.external_id
            FROM matches m
            JOIN competitions c ON m.competition_id = c.id
            WHERE m.external_id LIKE 'sfs_%'
              AND m.odds_p1 IS NULL
              AND m.played_at >= :since
              AND c.priority <= :pmax
            ORDER BY m.played_at DESC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"since": since, "pmax": priority_max})
    else:
        query = text("""
            SELECT id, external_id
            FROM matches
            WHERE external_id LIKE 'sfs_%'
              AND odds_p1 IS NULL
              AND played_at >= :since
            ORDER BY played_at DESC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"since": since})
    return df


def update_odds_in_db(match_id: int, odds_p1: float, odds_p2: float) -> None:
    stmt = text("""
        UPDATE matches
        SET odds_p1 = :odds_p1,
            odds_p2 = :odds_p2,
            odds_source = 'sofascore'
        WHERE id = :match_id
    """)
    with engine.begin() as conn:
        conn.execute(
            stmt, {"odds_p1": odds_p1, "odds_p2": odds_p2, "match_id": match_id}
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Sofascore odds pour matchs WTT/ITTF"
    )
    parser.add_argument(
        "--since",
        default="2022-01-01",
        help="Date minimum YYYY-MM-DD (défaut: 2022-01-01)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.2, help="Délai entre requêtes en secondes"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Affiche sans écrire en DB"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limite le nombre de matchs traités (0 = tous)",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=2,
        help="Priorité max des compétitions (1=WTT top, 2=WTT+ETTU, 99=tout, défaut: 2)",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if not _HAS_CURL_CFFI:
        logger.error("curl-cffi requis. Installez : pip install curl-cffi")
        sys.exit(1)

    logger.info(
        f"Chargement des matchs WTT/ITTF (priority<={args.priority}) sans cotes depuis {args.since}..."
    )
    df = fetch_matches_without_odds(args.since, priority_max=args.priority)
    total = len(df)
    logger.info(f"{total} matchs éligibles trouvés")

    if total == 0:
        logger.info("Aucun match à mettre à jour.")
        return

    if args.limit > 0:
        df = df.head(args.limit)
        logger.info(f"Limité à {len(df)} matchs (--limit {args.limit})")

    session = cffi_requests.Session(impersonate="chrome120")
    session.headers.update(_CFFI_HEADERS)

    updated = 0
    no_odds = 0
    errors = 0
    last_request = 0.0

    for i, row in enumerate(df.itertuples(), 1):
        match_id = row.id
        external_id = row.external_id  # e.g. "sfs_15631518"
        event_id = external_id.replace("sfs_", "")

        url = f"{API_BASE}/event/{event_id}/odds/1/all"

        # Throttle
        elapsed = time.time() - last_request
        if elapsed < args.delay:
            time.sleep(args.delay - elapsed)
        last_request = time.time()

        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 404:
                # Pas de cotes disponibles pour ce match
                no_odds += 1
                if i % 500 == 0 or i <= 5:
                    logger.debug(f"[{i}/{len(df)}] {external_id} : 404 (pas de cotes)")
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            errors += 1
            logger.debug(f"[{i}/{len(df)}] {external_id} erreur : {e}")
            continue

        odds_p1, odds_p2 = parse_odds_response(data)

        if odds_p1 is None or odds_p2 is None:
            no_odds += 1
            continue

        if args.dry_run:
            logger.info(
                f"[DRY-RUN] match_id={match_id} {external_id} → odds_p1={odds_p1} odds_p2={odds_p2}"
            )
        else:
            try:
                update_odds_in_db(match_id, odds_p1, odds_p2)
                updated += 1
            except Exception as e:
                logger.error(f"DB update error match_id={match_id}: {e}")
                errors += 1

        if i % 100 == 0:
            logger.info(
                f"Progression : {i}/{len(df)} traités | "
                f"mis à jour : {updated} | sans cotes : {no_odds} | erreurs : {errors}"
            )

    logger.info(
        f"\n=== TERMINÉ ===\n"
        f"  Traités   : {len(df)}\n"
        f"  Mis à jour: {updated}\n"
        f"  Sans cotes: {no_odds}\n"
        f"  Erreurs   : {errors}"
    )


if __name__ == "__main__":
    main()
