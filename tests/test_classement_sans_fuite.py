"""Le classement d'un match est celui connu à sa date, jamais un instantané postérieur.

Régression du 30/09/2026 : avec `ranking_as_of: false`, chaque match depuis 2021 recevait
le classement du dernier instantané (septembre 2026), ce qui gonflait la log-loss de test.
"""

import pandas as pd
import yaml
from sqlalchemy import create_engine, text

from src.features import pipeline


def test_la_configuration_utilise_le_classement_a_la_date_du_match():
    with open("config/settings.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["features"]["ranking_as_of"] is True


def test_un_match_anterieur_au_premier_instantane_n_a_pas_de_classement(monkeypatch):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        for table in ("ittf_rankings", "wtt_rankings"):
            conn.execute(text(f"CREATE TABLE {table} (player_id INTEGER, rank INTEGER, snapshot_date TEXT)"))
            conn.execute(
                text(f"INSERT INTO {table} VALUES (1, 40, '2026-03-30'), (1, 12, '2026-09-28'), (2, 80, '2026-03-30')")
            )
    monkeypatch.setattr(pipeline, "engine", engine)
    df = pd.DataFrame(
        {
            "player1_id": [1, 1],
            "player2_id": [2, 2],
            "played_at": pd.to_datetime(["2025-06-01", "2026-04-01"]),
            "ittf_rank_p1": [12, 12],
            "ittf_rank_p2": [80, 80],
            "wtt_rank_p1": [12, 12],
            "wtt_rank_p2": [80, 80],
        }
    )
    sortie = pipeline._add_asof_rankings(df, enabled=True)
    assert sortie["ittf_rank_p1"].tolist() == [9999.0, 40.0]
    assert sortie["wtt_rank_p2"].tolist() == [9999.0, 80.0]
