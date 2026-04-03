import numpy as np
import pandas as pd
from chess_src.models.lgbm_model import ChessLGBMModel
from chess_src.features.pipeline import ChessFeaturePipeline

TOTAL_ROUNDS = 14
STATIC_FEATURE_COLS = [
    "elo_diff",
    "elo_prob_white",
    "h2h_matches",
    "h2h_points_white",
    "h2h_recent_points_white",
    "form_white",
    "form_black",
    "form_diff",
]
CONTEXT_FEATURE_COLS = [
    "white_tournament_points",
    "black_tournament_points",
    "tournament_points_diff",
    "round_norm",
    "is_closing_stage",
]


class CandidatesSimulator:
    def __init__(
        self,
        model: ChessLGBMModel,
        pipeline: ChessFeaturePipeline,
        players_config: list,
        num_simulations: int = 1000,
    ):
        self.model = model
        self.pipeline = pipeline
        self.players = players_config
        self.num_simulations = num_simulations

    def simulate(
        self, current_matches: pd.DataFrame, remaining_matches: pd.DataFrame
    ) -> dict:
        """
        Simulate the remaining tournament rounds.
        Tournament context features (standings) are updated round-by-round within each simulation.
        Static features (elo, h2h, form) are computed once.
        """
        p_ids = [p["fide_id"] for p in self.players]
        win_counts = {pid: 0 for pid in p_ids}

        # Build history points from completed matches
        history_points = {pid: 0.0 for pid in p_ids}
        for _, r in current_matches.iterrows():
            w_id = int(r["white_id"])
            b_id = int(r["black_id"])
            history_points[w_id] = history_points.get(w_id, 0.0) + float(r["result"])
            history_points[b_id] = history_points.get(b_id, 0.0) + (
                1.0 - float(r["result"])
            )

        # Compute static features (elo, h2h, form) once using the full match history
        full_df = pd.concat([current_matches, remaining_matches], ignore_index=True)
        full_df["white_id"] = full_df["white_id"].astype(int)
        full_df["black_id"] = full_df["black_id"].astype(int)
        full_df["result"] = pd.to_numeric(full_df["result"], errors="coerce")
        df_proc = self.pipeline.process(full_df)

        # Extract only remaining match rows, sorted by round
        rem_df = df_proc[df_proc["result"].isna()].copy()
        rem_df = rem_df.sort_values(["round", "white_id"]).reset_index(drop=True)

        if rem_df.empty:
            winner_id = max(history_points, key=history_points.get)
            return {pid: (1.0 if pid == winner_id else 0.0) for pid in p_ids}

        # Group rounds for batch prediction
        rounds_grouped = [
            (rnd, grp.reset_index(drop=True)) for rnd, grp in rem_df.groupby("round")
        ]
        outcomes = [0.0, 0.5, 1.0]

        for _ in range(self.num_simulations):
            sim_scores = history_points.copy()

            for rnd, rnd_matches in rounds_grouped:
                # Build feature rows with dynamic tournament context for this round
                batch = rnd_matches[STATIC_FEATURE_COLS].copy()
                batch["white_tournament_points"] = (
                    rnd_matches["white_id"].map(sim_scores).fillna(0.0)
                )
                batch["black_tournament_points"] = (
                    rnd_matches["black_id"].map(sim_scores).fillna(0.0)
                )
                batch["tournament_points_diff"] = (
                    batch["white_tournament_points"] - batch["black_tournament_points"]
                )
                batch["round_norm"] = rnd / TOTAL_ROUNDS
                batch["is_closing_stage"] = int(rnd >= TOTAL_ROUNDS - 3)

                probs_batch = self.model.predict_proba(batch)

                for i, row in rnd_matches.iterrows():
                    p = np.array(probs_batch[i], dtype=np.float64)
                    p = p / p.sum()
                    res = np.random.choice(outcomes, p=p)
                    sim_scores[int(row["white_id"])] += res
                    sim_scores[int(row["black_id"])] += 1.0 - res

            max_s = max(sim_scores[pid] for pid in p_ids)
            winners = [pid for pid in p_ids if sim_scores[pid] == max_s]
            win_counts[np.random.choice(winners)] += 1

        return {pid: win_counts[pid] / self.num_simulations for pid in p_ids}
