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
    enrich_with_sofascore_odds,
    _load_player_map,
    _match_player,
)


def _record_snapshots(covered: list[dict], snapshot_type: str, dry_run: bool) -> int:
    """Ajoute une ligne dans odds_snapshots par match couvert (série open→close pour la CLV)."""
    if dry_run:
        return 0
    now = datetime.now(timezone.utc)
    rows = []
    for ev in covered:
        eid = ev.get("event_id")
        rows.append({
            "xid": f"sfs_{eid}" if eid else None,
            "p1": ev.get("p1_name"), "p2": ev.get("p2_name"),
            "src": ev.get("bookmaker", "unknown"),
            "o1": ev.get("book_odds_p1"), "o2": ev.get("book_odds_p2"),
            "st": snapshot_type, "cap": now,
        })
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT OR IGNORE INTO odds_snapshots "
            "(match_external_id, player1_name, player2_name, source, odds_p1, odds_p2, snapshot_type, captured_at) "
            "VALUES (:xid,:p1,:p2,:src,:o1,:o2,:st,:cap)"
        ), rows)
    return len(rows)


def _update_closing_odds(covered: list[dict], dry_run: bool) -> int:
    """Renseigne betting_records.closing_odds pour les paris en attente sur ces matchs (côté parié)."""
    if dry_run:
        return 0
    updated = 0
    with engine.begin() as conn:
        for ev in covered:
            eid = ev.get("event_id")
            if not eid:
                continue
            m = conn.execute(text("SELECT id FROM matches WHERE external_id = :x"),
                             {"x": f"sfs_{eid}"}).fetchone()
            if not m:
                continue
            for bet_player, close in ((1, ev.get("book_odds_p1")), (2, ev.get("book_odds_p2"))):
                if close and close > 1:
                    res = conn.execute(text(
                        "UPDATE betting_records SET closing_odds = :c "
                        "WHERE match_id = :m AND bet_player = :bp AND closing_odds IS NULL"
                    ), {"c": close, "m": m[0], "bp": bet_player})
                    updated += res.rowcount or 0
    return updated


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
    parser.add_argument("--closing", action="store_true",
                        help="Marque ces cotes comme clôture (à lancer juste avant les matchs) "
                             "et renseigne closing_odds des paris en attente")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    odds_key = os.getenv("ODDS_API_KEY", "")
    has_betfair = all(
        os.getenv(k) for k in ("BETFAIR_APP_KEY", "BETFAIR_USERNAME", "BETFAIR_PASSWORD")
    )
    if not odds_key and not has_betfair:
        logger.info(
            "Ni ODDS_API_KEY ni identifiants Betfair — cotes récupérées via Sofascore uniquement"
        )

    upcoming, source = fetch_upcoming_matches(days=args.days, all_leagues=False)
    if not upcoming:
        logger.info("Aucun match à venir trouvé")
        return

    logger.info(f"{len(upcoming)} matchs à venir (source: {source})")

    # 1. Sofascore (public, sans compte, marche depuis la France) puis
    # 2. Betfair / The Odds API en complément sur les matchs restants.
    upcoming = enrich_with_sofascore_odds(upcoming)
    upcoming = enrich_with_bookmaker_odds(upcoming, odds_key)
    covered = [ev for ev in upcoming if ev.get("book_odds_p1")]
    logger.info(f"Cotes bookmaker disponibles pour {len(covered)}/{len(upcoming)} matchs")

    if not covered:
        logger.info("Aucune cote bookmaker — vérifier la couverture Betfair Exchange / The Odds API pour table_tennis")
        return

    # Snapshots de cotes (série open→close pour la Closing Line Value)
    snap_type = "close" if args.closing else "inter"
    n_snap = _record_snapshots(covered, snap_type, args.dry_run)
    if n_snap:
        logger.info(f"{n_snap} snapshots de cotes enregistrés (type={snap_type})")
    if args.closing:
        n_clv = _update_closing_odds(covered, args.dry_run)
        logger.info(f"closing_odds renseignée pour {n_clv} paris en attente")

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
