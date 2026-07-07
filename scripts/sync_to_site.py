"""
Synchronise l'artefact web du modèle vers le site (valentinmartine.com).

Copie data/predictions.json → <site>/public/data/tt_predictions.json, où <site>
est le dépôt du portfolio React (déployé sur Vercel). Avec --push, committe et
pousse le dépôt site (Vercel redéploie automatiquement au push).

Usage :
    python scripts/sync_to_site.py                    # copie seulement
    python scripts/sync_to_site.py --push             # copie + commit + push du site
    python scripts/sync_to_site.py --site ../mon-site # chemin du dépôt site
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SITE = REPO.parent / "valentinmartine"
SRC = REPO / "data" / "predictions.json"
DEST_REL = Path("public") / "data" / "tt_predictions.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(DEFAULT_SITE), help="Chemin du dépôt site")
    ap.add_argument("--push", action="store_true", help="Committer et pousser le dépôt site")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if not SRC.exists():
        logger.error(f"Source introuvable : {SRC} — lancer `python -m src.export` d'abord")
        return 1

    site = Path(args.site).resolve()
    dest = site / DEST_REL
    if not dest.parent.exists():
        logger.error(f"Dossier site introuvable : {dest.parent} (dépôt site correct ?)")
        return 1

    shutil.copyfile(SRC, dest)
    logger.info(f"Copié : {SRC.name} → {dest}")

    if not args.push:
        logger.info("Copie effectuée (sans push). Utiliser --push pour déployer.")
        return 0

    def git(*a):
        return subprocess.run(["git", "-C", str(site), *a], capture_output=True, text=True)

    git("add", str(DEST_REL))
    diff = git("diff", "--staged", "--quiet")
    if diff.returncode == 0:
        logger.info("Aucun changement à committer (site déjà à jour).")
        return 0
    git("commit", "-m", "chore(data): sync prédictions tennis de table depuis le modèle")
    push = git("push")
    if push.returncode != 0:
        logger.error(f"Push échoué : {push.stderr.strip()[:200]}")
        return 1
    logger.info("Site poussé → Vercel va redéployer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
