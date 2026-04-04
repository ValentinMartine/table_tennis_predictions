"""
Stocke les cotes bookmaker pré-match dans la DB pour les matchs WTT à venir.

À lancer quotidiennement (ou avant chaque tournoi) pour accumuler des données
d'entraînement avec odds. Une fois les matchs terminés, les lignes existantes
auront odds_p1/odds_p2 renseignés → réentraîner le modèle avec implied_prob_p1.

Usage :
    python scripts/store_pre_match_odds.py
    python scripts/store_pre_match_odds.py --days 14 --dry-run
"""
import argparse
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from loguru import logger
from sqlalchemy import text

from src.database.db import engine
from src.scraping.oddsapi import enrich_with_bookmaker_odds, get_table_tennis_odds
from scripts.predict_upcoming import (
    fetch_upcoming_matches,
    _load_player_map,
    _match_player,
)


def _upsert_odds(external_id: str, odds_p1: float, odds_p2: float, source: str, dry_run: bool) -> bool:
    """Met à jour odds_p1/odds_p2 sur un match existant, ou log si absent."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, odds_p1 FROM matches WHERE external_id = :eid"),
            {"eid": external_id},
        ).fetchone()

        if not row:
            return False

        if row[1] is not None:
            logger.debug(f"Cotes déjà présentes pour {external_id} — skip")
            return False

        if not dry_run:
            conn.execute(
                text("""
                    UPDATE matches
                    SET odds_p1 = :o1, odds_p2 = :o2, odds_source = :src
                    WHERE external_id = :eid
                """),
                {"o1": odds_p1, "o2": odds_p2, "src": source, "eid": external_id},
            )
            conn.commit()
        return True


def _pending_external_ids(player_map, matches: list[dict]) -> dict[str, dict]:
    """
    Pour chaque match upcoming, tente de trouver son external_id en DB
    (match not yet played = winner IS NULL).
    Retourne {external_id: ev_dict}.
    """
    result = {}
    with engine.connect() as conn:
        for ev in matches:
            p1_id = _match_player(ev["p1_name"], player_map)
            p2_id = _match_player(ev["p2_name"], player_map)
            if not p1_id or not p2_id:
                continue

            # Cherche un match futur/récent entre ces deux joueurs sans résultat
            row = conn.execute(
                text("""
                    SELECT external_id FROM matches
                    WHERE ((player1_id = :p1 AND player2_id = :p2)
                        OR (player1_id = :p2 AND player2_id = :p1))
                      AND winner IS NULL
                    ORDER BY played_at DESC
                    LIMIT 1
                """),
                {"p1": p1_id, "p2": p2_id},
            ).fetchone()

            if row:
                result[row[0]] = ev
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans écrire")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    odds_key = os.getenv("ODDS_API_KEY", "")
    if not odds_key:
        logger.error("ODDS_API_KEY non défini — impossible de récupérer les cotes")
        sys.exit(1)

    upcoming, source = fetch_upcoming_matches(days=args.days, all_leagues=False)
    if not upcoming:
        logger.info("Aucun match à venir trouvé")
        return

    logger.info(f"{len(upcoming)} matchs à venir (source: {source})")

    upcoming = enrich_with_bookmaker_odds(upcoming, odds_key)
    covered = [ev for ev in upcoming if ev.get("book_odds_p1")]
    logger.info(f"Cotes bookmaker disponibles pour {len(covered)}/{len(upcoming)} matchs")

    if not covered:
        logger.info("Aucune cote bookmaker — vérifier la couverture The Odds API pour table_tennis")
        return

    player_map = _load_player_map()
    pending = _pending_external_ids(player_map, covered)
    logger.info(f"Matchs en DB sans cotes : {len(pending)}")

    updated = 0
    for external_id, ev in pending.items():
        stored = _upsert_odds(
            external_id,
            ev["book_odds_p1"],
            ev["book_odds_p2"],
            ev.get("bookmaker", "the_odds_api"),
            args.dry_run,
        )
        if stored:
            action = "[DRY-RUN] " if args.dry_run else ""
            logger.info(f"{action}Cotes stockées pour {ev['p1_name']} vs {ev['p2_name']}: "
                        f"{ev['book_odds_p1']:.2f} / {ev['book_odds_p2']:.2f} ({ev.get('bookmaker','')})")
            updated += 1

    logger.info(f"{'[DRY-RUN] ' if args.dry_run else ''}{updated} matchs mis à jour")


if __name__ == "__main__":
    main()
