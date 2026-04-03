import pandas as pd


def _color_streak(history: list) -> int:
    """Returns length of current color streak: positive = whites, negative = blacks."""
    if not history:
        return 0
    last = history[-1]
    streak = 0
    for c in reversed(history):
        if c == last:
            streak += 1
        else:
            break
    return streak if last == "W" else -streak


class ChessContextCalculator:
    def __init__(self, total_rounds: int = 14):
        self.total_rounds = total_rounds

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds tournament-specific context features.
        - round_number: normalize current round
        - is_closing_stage: flag for final rounds (11-14)
        - color: white/black advantage (already inherent in white_elo vs black_elo but we can add it)
        - points_delta: difference in current tournament standings
        """
        df = df.copy()
        # Clean round numbers (TWIC uses 1.29, ?, etc.)
        df["round"] = (
            pd.to_numeric(df["round"], errors="coerce").fillna(1).astype(float)
        )

        df = df.sort_values(["tournament", "round"])

        # Standings tracking per tournament
        for col in [
            "white_tournament_points",
            "black_tournament_points",
            "white_last2_score",
            "black_last2_score",
            "white_color_balance",
            "black_color_balance",
            "white_gap_to_leader",
            "black_gap_to_leader",
            "white_color_streak",
            "black_color_streak",
        ]:
            df[col] = 0.0

        for tourney in df["tournament"].unique():
            t_mask = df["tournament"] == tourney
            t_df = df[t_mask].sort_values(["round", "played_at"])

            points = {}  # player_id -> cumulative points
            rounds_history = {}  # player_id -> [(round_num, score)]
            colors = {}  # player_id -> (whites_count, blacks_count)

            color_history = {}  # player_id -> list of 'W'/'B' in order

            for idx, row in t_df.iterrows():
                w_id = row["white_id"]
                b_id = row["black_id"]
                round_num = row["round"]

                # Snapshot BEFORE this match
                df.at[idx, "white_tournament_points"] = points.get(w_id, 0.0)
                df.at[idx, "black_tournament_points"] = points.get(b_id, 0.0)

                w_hist = sorted(rounds_history.get(w_id, []), key=lambda x: x[0])
                b_hist = sorted(rounds_history.get(b_id, []), key=lambda x: x[0])
                df.at[idx, "white_last2_score"] = sum(s for _, s in w_hist[-2:])
                df.at[idx, "black_last2_score"] = sum(s for _, s in b_hist[-2:])

                w_col = colors.get(w_id, (0, 0))
                b_col = colors.get(b_id, (0, 0))
                df.at[idx, "white_color_balance"] = w_col[0] - w_col[1]
                df.at[idx, "black_color_balance"] = b_col[0] - b_col[1]

                leader_score = max(points.values()) if points else 0.0
                df.at[idx, "white_gap_to_leader"] = leader_score - points.get(w_id, 0.0)
                df.at[idx, "black_gap_to_leader"] = leader_score - points.get(b_id, 0.0)

                df.at[idx, "white_color_streak"] = _color_streak(
                    color_history.get(w_id, [])
                )
                df.at[idx, "black_color_streak"] = _color_streak(
                    color_history.get(b_id, [])
                )

                # Update state AFTER this match
                res = row["result"]
                if pd.notna(res):
                    res = float(res)
                    points[w_id] = points.get(w_id, 0.0) + res
                    points[b_id] = points.get(b_id, 0.0) + (1.0 - res)
                    rounds_history.setdefault(w_id, []).append((round_num, res))
                    rounds_history.setdefault(b_id, []).append((round_num, 1.0 - res))
                    w_whites, w_blacks = colors.get(w_id, (0, 0))
                    b_whites, b_blacks = colors.get(b_id, (0, 0))
                    colors[w_id] = (w_whites + 1, w_blacks)
                    colors[b_id] = (b_whites, b_blacks + 1)
                    color_history.setdefault(w_id, []).append("W")
                    color_history.setdefault(b_id, []).append("B")

        df["tournament_points_diff"] = (
            df["white_tournament_points"] - df["black_tournament_points"]
        )
        df["round_norm"] = df["round"] / self.total_rounds
        df["is_closing_stage"] = (df["round"] >= (self.total_rounds - 3)).astype(int)

        return df
