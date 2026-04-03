import streamlit as st
import pandas as pd
import sqlite3
import yaml
import sys
import os
import numpy as np
from pathlib import Path

# Pathing and namespace mapping for unpickling
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "chess_src"))

import chess_src.models.lgbm_model

sys.modules["src.models.lgbm_model"] = chess_src.models.lgbm_model

from chess_src.models.lgbm_model import ChessLGBMModel
from chess_src.features.pipeline import ChessFeaturePipeline
from chess_src.simulation.monte_carlo import CandidatesSimulator

# Configuration
DB_PATH = str(PROJECT_ROOT / "data" / "chess_matches.db")
CONFIG_PATH = str(PROJECT_ROOT / "config" / "settings.yaml")
MODEL_PATH = str(PROJECT_ROOT / "data" / "chess_lgbm.pkl")

# The 8 official FIDE IDs
CANDIDATE_FIDE_IDS = [
    2004887,
    2020009,
    8603405,
    24116068,
    14205481,
    25059650,
    4661654,
    24175439,
]
FIDE_IDS_SQL = ",".join(str(i) for i in CANDIDATE_FIDE_IDS)
TOURNAMENT_NAME = "Candidates 2026"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def get_current_standings():
    """Returns standings for the 8 candidates only, based on tournament results."""
    conn = get_db_connection()
    query = f"""
        SELECT
            p.name AS Joueur,
            p.country AS Pays,
            CAST(p.rating_initial AS INTEGER) AS Elo,
            ROUND(
                COALESCE(SUM(
                    CASE
                        WHEN m.result = 1.0 AND m.white_id = p.id THEN 1.0
                        WHEN m.result = 0.0 AND m.black_id = p.id THEN 1.0
                        WHEN m.result = 0.5                        THEN 0.5
                        ELSE 0.0
                    END
                ), 0), 1
            ) AS Points
        FROM players p
        LEFT JOIN matches m
            ON (p.id = m.white_id OR p.id = m.black_id)
            AND m.result IS NOT NULL
            AND m.tournament = '{TOURNAMENT_NAME}'
        WHERE p.id IN ({FIDE_IDS_SQL})
        GROUP BY p.id
        ORDER BY Points DESC, Elo DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_tournament_completed():
    """Returns only completed TOURNAMENT matches (not CSV history)."""
    conn = get_db_connection()
    query = f"""
        SELECT m.id, m.round, m.white_id, m.black_id, m.result, m.played_at, m.tournament
        FROM matches m
        WHERE m.result IS NOT NULL
          AND m.tournament = '{TOURNAMENT_NAME}'
          AND m.white_id IN ({FIDE_IDS_SQL})
          AND m.black_id IN ({FIDE_IDS_SQL})
        ORDER BY m.round ASC, m.id ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_upcoming_matches():
    """Returns upcoming (unplayed) tournament matches among the 8 candidates."""
    conn = get_db_connection()
    query = f"""
        SELECT
            m.id,
            m.round,
            pw.name AS white,
            pb.name AS black,
            m.white_id,
            m.black_id,
            m.played_at,
            m.tournament
        FROM matches m
        JOIN players pw ON m.white_id = pw.id
        JOIN players pb ON m.black_id = pb.id
        WHERE m.result IS NULL
          AND m.tournament = '{TOURNAMENT_NAME}'
          AND m.white_id IN ({FIDE_IDS_SQL})
          AND m.black_id IN ({FIDE_IDS_SQL})
        ORDER BY m.round ASC, m.id ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ─── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FIDE Candidates 2026", layout="wide")
st.title("♟️ FIDE Candidates 2026 — Dashboard Prédictif")
st.markdown("Classement en temps réel et prédictions IA pour les rondes restantes.")

config = load_config()
col1, col2 = st.columns([1, 2])

# ── Standings ─────────────────────────────────────────────────────────────────
with col1:
    st.subheader("📊 Classement actuel")
    standings = get_current_standings()
    standings.insert(0, "#", range(1, len(standings) + 1))
    st.dataframe(standings, hide_index=True, use_container_width=True)

# ── Upcoming Match Probabilities ──────────────────────────────────────────────
with col2:
    st.subheader("🔮 Probabilités — Prochains matchs")

    if not os.path.exists(MODEL_PATH):
        st.warning(
            "Modèle introuvable. Patientez pendant le ré-entraînement du modèle..."
        )
    else:
        try:
            model = ChessLGBMModel.load(MODEL_PATH)
            pipeline = ChessFeaturePipeline(config, db_path=DB_PATH)

            completed = get_tournament_completed()
            upcoming = get_upcoming_matches()

            if upcoming.empty:
                st.info("Tournoi terminé !")
            else:
                upcoming_for_pipeline = upcoming[
                    ["id", "round", "white_id", "black_id", "played_at", "tournament"]
                ].copy()
                upcoming_for_pipeline["result"] = np.nan

                all_matches = pd.concat(
                    [completed, upcoming_for_pipeline], ignore_index=True
                )
                all_features = pipeline.process(all_matches)

                # Upcoming matches have NaN result after pipeline (result column preserved)
                upcoming_features = all_features[all_features["result"].isna()].copy()

                if upcoming_features.empty:
                    st.warning("Aucune prédiction disponible.")
                else:
                    probs = model.predict_proba(upcoming_features)
                    pred_df = pd.DataFrame(
                        probs, columns=["P_Noir", "P_Nulle", "P_Blanc"]
                    )
                    pred_df["id"] = upcoming_features["id"].values
                    pred_df["Ronde"] = upcoming_features["round"].values.astype(int)

                    final_display = pd.merge(
                        upcoming[["id", "white", "black"]], pred_df, on="id"
                    ).sort_values("Ronde").reset_index(drop=True)

                    display_df = final_display[
                        ["Ronde", "white", "black", "P_Blanc", "P_Nulle", "P_Noir"]
                    ].rename(columns={"white": "Blancs", "black": "Noirs"})
                    display_df.columns = ["Ronde", "Blancs", "Noirs", "Victoire Blancs", "Nulle", "Victoire Noirs"]

                    # Alternating round background + outcome colours
                    rounds = display_df["Ronde"].unique()
                    round_parity = {r: i % 2 for i, r in enumerate(sorted(rounds))}
                    ROW_ODD  = "background-color: #f0f2f6; color: #000000"
                    ROW_EVEN = "background-color: #ffffff; color: #000000"
                    GREEN  = "background-color: #c8f0c8; color: #1a5c1a; font-weight: bold"
                    ORANGE = "background-color: #ffe5a0; color: #7a4f00; font-weight: bold"
                    RED    = "background-color: #f0c8c8; color: #7a1a1a; font-weight: bold"
                    DIM    = "color: #999999"

                    def style_row(row):
                        base = ROW_ODD if round_parity.get(row["Ronde"], 0) else ROW_EVEN
                        pb, pn, pk = row["Victoire Blancs"], row["Nulle"], row["Victoire Noirs"]
                        best = max(pb, pn, pk)
                        if best == pb:
                            wb, wn, wk = GREEN, ORANGE, RED
                        elif best == pn:
                            wb, wn, wk = ORANGE, GREEN, ORANGE
                        else:
                            wb, wn, wk = RED, ORANGE, GREEN
                        return [base, base, base, wb, wn, wk]

                    styled = (
                        display_df.style
                        .apply(style_row, axis=1)
                        .format({
                            "Victoire Blancs": "{:.1%}",
                            "Nulle": "{:.1%}",
                            "Victoire Noirs": "{:.1%}",
                        })
                    )
                    st.dataframe(styled, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur d'affichage : {e}")
            import traceback

            st.code(traceback.format_exc())

# ── Monte Carlo Forecast ───────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Prévision du tournoi — Simulation Monte Carlo")

if st.button("🚀 Lancer 1 000 simulations"):
    with st.spinner("Simulation en cours…"):
        try:
            _model = ChessLGBMModel.load(MODEL_PATH)
            _pipeline = ChessFeaturePipeline(config, db_path=DB_PATH)
            _players = config["players"]

            completed_tourney = get_tournament_completed()
            upcoming_tourney = get_upcoming_matches()
            upcoming_tourney["result"] = np.nan

            simulator = CandidatesSimulator(
                _model, _pipeline, _players, num_simulations=1000
            )
            raw_results = simulator.simulate(completed_tourney, upcoming_tourney)

            # Map FIDE IDs → player names
            id_to_name = {p["fide_id"]: p["name"] for p in _players}
            res_df = (
                pd.DataFrame(
                    [
                        (id_to_name.get(pid, str(pid)), round(prob * 100, 1))
                        for pid, prob in raw_results.items()
                    ],
                    columns=["Joueur", "% Victoire tournoi"],
                )
                .sort_values("% Victoire tournoi", ascending=False)
                .reset_index(drop=True)
            )
            res_df.insert(0, "#", range(1, len(res_df) + 1))

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.dataframe(res_df, hide_index=True, use_container_width=True)
            with col_b:
                st.bar_chart(res_df.set_index("Joueur")["% Victoire tournoi"])

        except Exception as e:
            st.error(f"Erreur de simulation : {e}")
            import traceback

            st.code(traceback.format_exc())
