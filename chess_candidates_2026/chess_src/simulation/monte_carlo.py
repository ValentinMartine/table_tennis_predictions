import numpy as np
import pandas as pd
from chess_src.models.lgbm_model import ChessLGBMModel
from chess_src.features.pipeline import ChessFeaturePipeline

TOTAL_ROUNDS = 14

# Context features recomputed dynamically per simulation round
DYNAMIC_CONTEXT_COLS = {
    "white_tournament_points",
    "black_tournament_points",
    "tournament_points_diff",
    "white_gap_to_leader",
    "black_gap_to_leader",
    "white_last2_score",
    "black_last2_score",
}


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
            sim_round_history: dict[int, list] = {}  # pid -> [(rnd, score)]

            for rnd, rnd_matches in rounds_grouped:
                # Start from fully-computed pipeline features (includes all new features)
                batch = rnd_matches[self.model.feature_cols].copy()

                # Override dynamic context features with current simulation state
                w_pts = rnd_matches["white_id"].map(sim_scores).fillna(0.0).values
                b_pts = rnd_matches["black_id"].map(sim_scores).fillna(0.0).values
                leader = max(sim_scores.values()) if sim_scores else 0.0

                batch["white_tournament_points"] = w_pts
                batch["black_tournament_points"] = b_pts
                batch["tournament_points_diff"] = w_pts - b_pts
                batch["white_gap_to_leader"] = leader - w_pts
                batch["black_gap_to_leader"] = leader - b_pts

                def last2(pid):
                    h = sorted(sim_round_history.get(pid, []), key=lambda x: x[0])
                    return sum(s for _, s in h[-2:])

                batch["white_last2_score"] = [
                    last2(int(r["white_id"])) for _, r in rnd_matches.iterrows()
                ]
                batch["black_last2_score"] = [
                    last2(int(r["black_id"])) for _, r in rnd_matches.iterrows()
                ]

                probs_batch = self.model.predict_proba(batch)

                for i, row in rnd_matches.iterrows():
                    p = np.array(probs_batch[i], dtype=np.float64)
                    p = p / p.sum()
                    res = np.random.choice(outcomes, p=p)
                    w_id = int(row["white_id"])
                    b_id = int(row["black_id"])
                    sim_scores[w_id] += res
                    sim_scores[b_id] += 1.0 - res
                    sim_round_history.setdefault(w_id, []).append((rnd, res))
                    sim_round_history.setdefault(b_id, []).append((rnd, 1.0 - res))

            max_s = max(sim_scores[pid] for pid in p_ids)
            winners = [pid for pid in p_ids if sim_scores[pid] == max_s]
            win_counts[np.random.choice(winners)] += 1

        return {pid: win_counts[pid] / self.num_simulations for pid in p_ids}
