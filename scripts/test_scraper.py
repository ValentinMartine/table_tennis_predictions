"""
Script de test des scrapers — valide le DOM et les sélecteurs avant le vrai scraping.

Usage :
    python scripts/test_scraper.py --source skillgamesboard --comp setka_cup
    python scripts/test_scraper.py --source flashscore --comp ttnet
    python scripts/test_scraper.py --source flashscore --comp bundesliga
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        choices=["skillgamesboard", "flashscore", "sofascore"],
        required=True,
    )
    p.add_argument("--comp", required=True, help="competition_id (ex: setka_cup, ttnet, wtt_champions)")
    p.add_argument("--days", type=int, default=7, help="Nb de jours à scraper (défaut: 7)")
    return p.parse_args()


def main():
    args = parse_args()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    logger.info(f"Test scraper : {args.source} / {args.comp}")
    logger.info(f"Période : {start_date.date()} → {end_date.date()}")

    if args.source == "skillgamesboard":
        from src.scraping.skillgamesboard import SkillGamesBoardScraper
        scraper = SkillGamesBoardScraper({})
    elif args.source == "sofascore":
        from src.scraping.sofascore import SofascoreScraper
        scraper = SofascoreScraper({})
    else:
        from src.scraping.flashscore import FlashscoreScraper
        scraper = FlashscoreScraper({})

    matches = scraper.scrape_competition(args.comp, start_date, end_date)

    if not matches:
        logger.error("Aucun match extrait — les sélecteurs sont probablement à ajuster.")
        logger.info("Ouvre le site manuellement et inspecte le DOM (F12) pour trouver")
        logger.info("les vraies classes CSS des lignes de matchs, noms et scores.")
        sys.exit(1)

    logger.info(f"\n{'='*50}")
    logger.info(f"{len(matches)} matchs extraits — exemples :")
    for m in matches[:5]:
        logger.info(
            f"  {m.played_at.strftime('%Y-%m-%d')} | "
            f"{m.player1_name} {m.score_p1}-{m.score_p2} {m.player2_name} "
            f"[{m.external_id}]"
        )
    logger.info(f"{'='*50}")
    logger.info("Scraper fonctionnel. Lance maintenant :")
    logger.info(f"  python scripts/run_scraper.py --from 2022-01-01 --to 2024-12-31 --priority 1")


if __name__ == "__main__":
    main()
