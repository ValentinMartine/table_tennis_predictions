"""
Script : scraping des matchs historiques ou récents.

Usage :
    python scripts/run_scraper.py --from 2022-01-01 --to 2024-12-31
    python scripts/run_scraper.py --days 7         # 7 derniers jours
    python scripts/run_scraper.py --priority 1     # ligues haute priorité seulement
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.database.db import init_db
from src.scraping.pipeline import run_scraping


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape les résultats TT")
    parser.add_argument("--from", dest="start", type=str, help="Date début YYYY-MM-DD")
    parser.add_argument("--to", dest="end", type=str, help="Date fin YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=None, help="Derniers N jours")
    parser.add_argument("--priority", type=int, default=2, help="Priorité max des compétitions (1-3)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.days:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=args.days)
    else:
        start_date = datetime.strptime(args.start or "2022-01-01", "%Y-%m-%d")
        end_date = datetime.strptime(args.end or datetime.utcnow().strftime("%Y-%m-%d"), "%Y-%m-%d")

    logger.info(f"Scraping du {start_date.date()} au {end_date.date()} (priorité ≤ {args.priority})")
    init_db()

    stats = run_scraping(start_date, end_date, priority_max=args.priority)

    total = sum(stats.values())
    logger.info(f"Total : {total} nouveaux matchs insérés")
    for source, n in stats.items():
        logger.info(f"  {source}: {n}")


if __name__ == "__main__":
    main()
