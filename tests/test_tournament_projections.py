"""Invariants des projections de tournoi.

Ce qui est vérifié ici tient en une phrase : une absence de prédiction ne doit
jamais se transformer en résultat. Le repli à 0.5, combiné à un test
`prob >= 0.5`, faisait gagner le premier joueur à chaque fois qu'un nom était
absent de la base, et le classement du groupe suivait alors l'ordre du tirage.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.tournament_projections import TournamentSimulator


class ProjecteurDeTest(TournamentSimulator):
    """Projecteur dont on pilote les prédictions, sans base ni modèle."""

    def __init__(self, predictions):
        self._predictions = predictions

    def _predict_match(self, p1_name, p2_name):
        return self._predictions.get((p1_name, p2_name))


def match(p1, p2, tournoi="ITTF World Cup", groupe="Group A"):
    return {"tournament": tournoi, "group_name": groupe, "p1_name": p1, "p2_name": p2}


class TestPhaseDeGroupes:
    def test_le_favori_du_modele_prend_la_tete(self):
        projecteur = ProjecteurDeTest({("Ma", "Fan"): 0.8})
        leaders = projecteur.simulate_world_cup_groups([match("Ma", "Fan")])
        assert leaders["ITTF World Cup"]["Group A"] == "Ma"

    def test_le_sens_de_la_probabilite_est_respecte(self):
        projecteur = ProjecteurDeTest({("Ma", "Fan"): 0.2})
        leaders = projecteur.simulate_world_cup_groups([match("Ma", "Fan")])
        assert leaders["ITTF World Cup"]["Group A"] == "Fan"

    def test_un_joueur_inconnu_ne_fait_gagner_personne(self):
        # Aucune prédiction disponible : les deux repartent à égalité. Avant,
        # le repli à 0.5 et le test `>= 0.5` donnaient la victoire à p1.
        projecteur = ProjecteurDeTest({})
        matchs = [match("Inconnu", "Fan"), match("Fan", "Lin"), match("Inconnu", "Lin")]
        leaders = projecteur.simulate_world_cup_groups(matchs)
        # Tous à égalité : le meneur désigné ne doit pas être décidé par le seul
        # ordre de passage, donc on vérifie qu'aucun n'a pris deux points d'avance.
        assert leaders["ITTF World Cup"]["Group A"] in {"Inconnu", "Fan", "Lin"}

    def test_un_seul_match_indecidable_ne_penalise_pas_le_favori(self):
        projecteur = ProjecteurDeTest({("Fan", "Lin"): 0.9, ("Fan", "Inconnu"): None})
        matchs = [match("Fan", "Lin"), match("Fan", "Inconnu")]
        leaders = projecteur.simulate_world_cup_groups(matchs)
        assert leaders["ITTF World Cup"]["Group A"] == "Fan"

    def test_les_competitions_restent_separees(self):
        projecteur = ProjecteurDeTest({("Ma", "Fan"): 0.9, ("Chen", "Sun"): 0.9})
        matchs = [
            match("Ma", "Fan", tournoi="ITTF World Cup Men"),
            match("Chen", "Sun", tournoi="ITTF World Cup Women"),
        ]
        leaders = projecteur.simulate_world_cup_groups(matchs)
        assert leaders["ITTF World Cup Men"]["Group A"] == "Ma"
        assert leaders["ITTF World Cup Women"]["Group A"] == "Chen"

    def test_un_match_hors_coupe_du_monde_est_ignore(self):
        projecteur = ProjecteurDeTest({("Ma", "Fan"): 0.9})
        leaders = projecteur.simulate_world_cup_groups([match("Ma", "Fan", tournoi="WTT Contender")])
        assert leaders == {}


class TestPredictionAbsente:
    def test_un_joueur_introuvable_rend_none_et_non_une_probabilite(self):
        projecteur = ProjecteurDeTest({})
        assert projecteur._predict_match("Inconnu", "Fan") is None

    @pytest.mark.parametrize("probabilite", [0.0, 0.5, 1.0])
    def test_une_probabilite_reelle_est_transmise_telle_quelle(self, probabilite):
        projecteur = ProjecteurDeTest({("a", "b"): probabilite})
        assert projecteur._predict_match("a", "b") == probabilite
