"""Chaque import tiers doit figurer dans requirements.txt.

Le 28/09/2026, `src/scraping/base.py` importait tenacity, absent des dépendances : la collecte
des résultats plantait à chaque exécution de la CI, masquée par continue-on-error.
"""
import ast
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
LOCAUX = {"src", "scripts", "dashboard", "tests", "config", "conftest"}
# Module importé -> nom du paquet pip quand ils diffèrent.
PAQUETS = {
    "sklearn": "scikit-learn", "curl_cffi": "curl-cffi", "yaml": "pyyaml", "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv", "dateutil": "python-dateutil", "PIL": "pillow",
}
# Outils locaux absents de la CI : l'API FastAPI et les alertes par python-telegram-bot.
OPTIONNELS = {"fastapi", "telegram"}


def imports_tiers() -> dict[str, str]:
    trouves = {}
    for dossier in ("src", "scripts"):
        for fichier in (RACINE / dossier).rglob("*.py"):
            arbre = ast.parse(fichier.read_text(encoding="utf-8"))
            for noeud in ast.walk(arbre):
                if isinstance(noeud, ast.Import):
                    noms = [a.name for a in noeud.names]
                elif isinstance(noeud, ast.ImportFrom) and noeud.level == 0 and noeud.module:
                    noms = [noeud.module]
                else:
                    continue
                for nom in noms:
                    racine = nom.split(".")[0]
                    if racine not in sys.stdlib_module_names and racine not in LOCAUX:
                        trouves.setdefault(racine, str(fichier.relative_to(RACINE)))
    return trouves


def declares(fichier: str) -> set[str]:
    noms = set()
    for ligne in (RACINE / fichier).read_text(encoding="utf-8").splitlines():
        ligne = ligne.split("#")[0].strip()
        if ligne:
            noms.add(re.split(r"[<>=!~\[; ]", ligne)[0].lower().replace("_", "-"))
    return noms


def test_imports_tiers_declares_dans_requirements():
    requis = declares("requirements.txt")
    manquants = {
        module: fichier for module, fichier in imports_tiers().items()
        if module not in OPTIONNELS and PAQUETS.get(module, module).lower().replace("_", "-") not in requis
    }
    assert manquants == {}
