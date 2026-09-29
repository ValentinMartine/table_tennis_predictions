from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _charger(nom):
    return yaml.safe_load((WORKFLOWS / nom).read_text(encoding="utf-8"))


def test_predictions_quotidiennes_sans_reentrainement():
    # Seul le lundi régénérait les matchs à venir : un tournoi commencé en semaine n'apparaissait pas.
    flux = _charger("predictions-quotidiennes.yml")
    commandes = " ".join(e.get("run", "") for e in flux["jobs"]["predictions"]["steps"])
    assert "predict_upcoming.py" in commandes and "src.export" in commandes
    assert "train_model.py" not in commandes and "run_scraper.py" not in commandes


def test_meme_groupe_de_concurrence_que_l_autonomie():
    assert _charger("predictions-quotidiennes.yml")["concurrency"]["group"] == _charger("autonomy.yml")["concurrency"]["group"]


def test_workflows_sans_emoji():
    for fichier in WORKFLOWS.glob("*.yml"):
        texte = fichier.read_text(encoding="utf-8")
        assert not any(0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF for c in texte), fichier.name
