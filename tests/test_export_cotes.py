"""Chaque match à venir exporte les cotes bookmaker et la probabilité implicite du marché."""

import pandas as pd
import pytest

from src.export import _prochain_match

LIGNE = {
    "Date": "01/10 10:00", "Tournoi": "WTT Champions Macao", "J1": "WANG Chuqin", "WTT#1": 1,
    "J2": "LIN Shidong", "WTT#2": 3, "P(J1)": "62.0%", "P(J2)": "38.0%", "Favori": "WANG Chuqin",
    "Confiance": "62.0%", "Δ Modèle/Elo": "+1.0%", "Edge(book)": "+4.0%", "Cote J1": 1.70, "Cote J2": 2.20,
}


def test_les_cotes_et_la_probabilite_du_marche_sont_exportees():
    m = _prochain_match(pd.Series(LIGNE))
    assert (m["odds_p1"], m["odds_p2"]) == (1.70, 2.20)
    assert m["market_p1_prob"] == pytest.approx((1 / 1.7) / (1 / 1.7 + 1 / 2.2), abs=1e-3)


def test_l_edge_et_les_classements_wtt_sont_relus():
    m = _prochain_match(pd.Series(LIGNE))
    assert m["edge"] == 0.04
    assert (m["wtt_p1"], m["wtt_p2"]) == (1, 3)


def test_sans_cotes_les_champs_restent_vides():
    m = _prochain_match(pd.Series({**LIGNE, "Cote J1": "-", "Cote J2": "-", "Edge(book)": "-"}))
    assert m["odds_p1"] is None and m["market_p1_prob"] is None and m["edge"] is None
