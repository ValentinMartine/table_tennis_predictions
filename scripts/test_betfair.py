"""
Test de connexion à Betfair Exchange et aperçu des marchés tennis de table.

À lancer une fois les identifiants renseignés dans .env :
    BETFAIR_APP_KEY, BETFAIR_USERNAME, BETFAIR_PASSWORD

Usage :
    python scripts/test_betfair.py
    python scripts/test_betfair.py --days 30

Ne révèle jamais le mot de passe : lit uniquement les variables d'environnement.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from src.scraping.betfair import get_table_tennis_odds_betfair


def _load_env():
    """Charge .env si python-dotenv est présent, sinon parsing minimal."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="Horizon en jours")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    _load_env()
    app_key = os.getenv("BETFAIR_APP_KEY", "")
    username = os.getenv("BETFAIR_USERNAME", "")
    password = os.getenv("BETFAIR_PASSWORD", "")

    missing = [n for n, v in [
        ("BETFAIR_APP_KEY", app_key),
        ("BETFAIR_USERNAME", username),
        ("BETFAIR_PASSWORD", password),
    ] if not v]
    if missing:
        logger.error(f"Variables manquantes dans .env : {', '.join(missing)}")
        sys.exit(1)

    logger.info(f"App key : {app_key[:4]}… | compte : {username}")
    markets = get_table_tennis_odds_betfair(app_key, username, password, days_ahead=args.days)

    if not markets:
        logger.warning(
            "Connexion OK mais aucun marché tennis de table à venir "
            f"sur {args.days} jours (ou identifiants/App Key à vérifier)."
        )
        return

    logger.info(f"✓ {len(markets)} marchés avec cotes exchange :")
    for m in markets[:20]:
        print(f"  {m['home']:<24} {m['odds_home']:>6.2f}  |  "
              f"{m['odds_away']:>6.2f}  {m['away']}")


if __name__ == "__main__":
    main()
