import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from chess_src.models.lgbm_model import ChessLGBMModel
from chess_src.features.pipeline import ChessFeaturePipeline
from chess_src.simulation.monte_carlo import CandidatesSimulator
from chess_src.scraping.chess_fetcher import ChessDataFetcher
from chess_src.database import DEFAULT_DB
import sqlite3

import argparse


def run_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=1000)
    args = parser.parse_args()

    # 1. Load Config & Resources
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        config = yaml.safe_load(f)

    players = config["players"]
    player_map = {p["fide_id"]: p["name"] for p in players}
    elo_map = {p["fide_id"]: p["rating_april_2006"] for p in players}

    model_path = PROJECT_ROOT / "data" / "chess_lgbm.pkl"
    if not model_path.exists():
        logger.error("Model not found. Run train.py first.")
        return

    model = ChessLGBMModel.load(str(model_path))
    pipeline = ChessFeaturePipeline(config)
    fetcher = ChessDataFetcher(players)

    # 2. Get Current Matches (Real rounds 1-4)
    conn = sqlite3.connect(DEFAULT_DB)
    current_matches = pd.read_sql_query(
        "SELECT * FROM matches WHERE tournament = 'Candidates 2026'", conn
    )
    conn.close()

    # Enrich current matches with Elo (needed by pipeline)
    current_matches["white_elo"] = current_matches["white_id"].map(elo_map)
    current_matches["black_elo"] = current_matches["black_id"].map(elo_map)

    # 3. Form Analysis (TPR)
    logger.info("--- FORM ANALYSIS (TPR) ---")
    standings = []
    for pid, name in player_map.items():
        # Games involving this player
        p_matches = current_matches[
            (current_matches["white_id"] == pid) | (current_matches["black_id"] == pid)
        ]
        score = 0.0
        opp_elos = []
        for _, m in p_matches.iterrows():
            if m["white_id"] == pid:
                score += m["result"]
                opp_elos.append(m["black_elo"])
            else:
                score += 1.0 - m["result"]
                opp_elos.append(m["white_elo"])

        # Simple TPR formula: AvgOppElo + 400 * (2*Score/N - 1)
        if len(p_matches) > 0:
            avg_opp = np.mean(opp_elos)
            tpr = avg_opp + 400 * (2 * (score / len(p_matches)) - 1)
            standings.append(
                {
                    "Player": name,
                    "Score": score,
                    "Played": len(p_matches),
                    "TPR": int(tpr),
                    "Elo": elo_map[pid],
                    "Momentum": int(tpr - elo_map[pid]),
                }
            )

    df_form = pd.DataFrame(standings).sort_values("Score", ascending=False)
    print("\nCURRENT STANDINGS & FORM MOMENTUM:")
    print(df_form.to_string(index=False))

    # 4. Run Final Forecast
    logger.info("--- TOURNAMENT FORECAST (10,000 simulations) ---")
    all_pairings = fetcher.fetch_candidates_pairings()
    # Remaining matches are those with round > 4
    remaining_pairings = all_pairings[all_pairings["round"] > 4].copy()

    # Enrich remaining pairings with Elo
    remaining_pairings["white_elo"] = remaining_pairings["white_id"].map(elo_map)
    remaining_pairings["black_elo"] = remaining_pairings["black_id"].map(elo_map)
    remaining_pairings["result"] = 0.5  # Placeholder
    remaining_pairings["tournament"] = "Candidates 2026"
    remaining_pairings["played_at"] = pd.Timestamp.now()

    simulator = CandidatesSimulator(model, pipeline, players, num_simulations=args.sims)
    win_probs = simulator.simulate(current_matches, remaining_pairings)

    forecast = []
    for pid, prob in win_probs.items():
        forecast.append({"Player": player_map[pid], "Win Prob": f"{prob * 100:.1f}%"})

    df_forecast = pd.DataFrame(forecast).sort_values("Win Prob", ascending=False)
    print("\nFINAL WIN PROBABILITIES (Monte Carlo):")
    print(df_forecast.to_string(index=False))


if __name__ == "__main__":
    run_analysis()
