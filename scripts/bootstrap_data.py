"""
Script de bootstrap — à lancer une seule fois pour initialiser la base de données.

Ordre des opérations :
  1. Crée les tables DB
  2. Télécharge et importe les infos joueurs ITTF (ittf_player_info.csv)
  3. Télécharge et importe les rankings historiques ITTF hommes + femmes
     (~24 ans de snapshots hebdomadaires, 2001 → 2024)
  4. Importe le dataset Kaggle Setka (si présent dans data/raw/)
  5. Scrape les données récentes des ligues priorité 1 (30 derniers jours)

Usage :
    python scripts/bootstrap_data.py
    python scripts/bootstrap_data.py --skip-ittf    # si déjà importé
    python scripts/bootstrap_data.py --skip-scrape  # CSV seulement
    python scripts/bootstrap_data.py --force        # re-télécharge tout
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db import init_db
from src.scraping.ittf_csv_loader import load_player_info_into_db, load_rankings_into_db
from src.scraping.kaggle_loader import load_kaggle_setka
from src.scraping.pipeline import run_scraping


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap de la base de données TT")
    p.add_argument("--skip-ittf", action="store_true", help="Ne pas importer les CSVs ITTF")
    p.add_argument("--skip-scrape", action="store_true", help="Ne pas scraper les données récentes")
    p.add_argument("--skip-kaggle", action="store_true", help="Ne pas importer le dataset Kaggle")
    p.add_argument("--force", action="store_true", help="Re-télécharge tous les CSV")
    p.add_argument("--days", type=int, default=30, help="Nb jours récents à scraper (défaut: 30)")
    return p.parse_args()


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    args = parse_args()

    logger.info("=== BOOTSTRAP BASE DE DONNÉES ===")

    # 1. Init DB
    logger.info("Étape 1/5 : Initialisation de la DB...")
    init_db()

    # 2. Infos joueurs ITTF
    if not args.skip_ittf:
        logger.info("Étape 2/5 : Import infos joueurs ITTF...")
        n = load_player_info_into_db(force_download=args.force)
        logger.info(f"  → {n} joueurs enrichis")
    else:
        logger.info("Étape 2/5 : [SKIPPED] Infos joueurs ITTF")

    # 3. Rankings ITTF historiques
    if not args.skip_ittf:
        logger.info("Étape 3/5 : Import rankings ITTF hommes (2001→2024)...")
        n_men = load_rankings_into_db(gender="M", force_download=args.force)
        logger.info(f"  → {n_men} snapshots hommes insérés")

        logger.info("Étape 3/5 : Import rankings ITTF femmes (2001→2024)...")
        n_women = load_rankings_into_db(gender="F", force_download=args.force)
        logger.info(f"  → {n_women} snapshots femmes insérés")
    else:
        logger.info("Étape 3/5 : [SKIPPED] Rankings ITTF")

    # 4. Kaggle dataset
    if not args.skip_kaggle:
        logger.info("Étape 4/5 : Import dataset Kaggle Setka (juin-juillet 2022)...")
        n = load_kaggle_setka()
        if n >= 0:
            logger.info(f"  → {n} matchs insérés")
        else:
            logger.info("  → Fichier absent. Télécharger depuis kaggle.com/datasets/medaxone/one-month-table-tennis-dataset")
            logger.info("     et placer dans data/raw/kaggle_setka_2022.csv")
    else:
        logger.info("Étape 4/5 : [SKIPPED] Kaggle dataset")

    # 5. Scraping récent
    if not args.skip_scrape:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=args.days)
        logger.info(f"Étape 5/5 : Scraping {args.days} derniers jours (priorité 1)...")
        stats = run_scraping(start_date, end_date, priority_max=1)
        total = sum(stats.values())
        logger.info(f"  → {total} matchs récents insérés")
        for src, n in stats.items():
            logger.info(f"     {src}: {n}")
    else:
        logger.info("Étape 5/5 : [SKIPPED] Scraping récent")

    logger.info("=== BOOTSTRAP TERMINÉ ===")
    logger.info("Prochaines étapes :")
    logger.info("  python scripts/run_scraper.py --from 2022-01-01 --to 2024-12-31 --priority 2")
    logger.info("  python scripts/train_model.py --model lgbm --shap")
    logger.info("  python scripts/backtest.py --plot")


if __name__ == "__main__":
    main()
