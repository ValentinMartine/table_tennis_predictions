"""
Script : prédiction live — à lancer via cron toutes les 15 minutes.

Usage :
    python scripts/live_predict.py
    # Cron : */15 * * * * cd /app && python scripts/live_predict.py
"""
import sys
from pathlib import Path

from loguru import logger

# Ajoute le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment.predictor import LivePredictor


def main():
    logger.remove()
    logger.add(
        "data/logs/live_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
    )
    logger.add(sys.stderr, level="INFO")

    try:
        predictor = LivePredictor()
        predictor.run()
    except Exception as e:
        logger.exception(f"Erreur critique : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
