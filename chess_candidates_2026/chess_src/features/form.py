import numpy as np
import pandas as pd
import math


class ChessFormCalculator:
    def __init__(self, window: int = 15):
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates recent form (TPR - Tournament Performance Rating).
        For each match, look at the last N games of each player and calculate their TPR.
        """
        df = df.sort_values("played_at").reset_index(drop=True).copy()

        form_white_list = []
        form_black_list = []

        if "white_elo" not in df.columns or "black_elo" not in df.columns:
            raise KeyError("ELO columns must be computed before Form.")

        for pos, (idx, row) in enumerate(df.iterrows()):
            w_id = int(row["white_id"])
            b_id = int(row["black_id"])

            form_white_list.append(self._calculate_tpr(df, w_id, pos))
            form_black_list.append(self._calculate_tpr(df, b_id, pos))

        df["form_white"] = form_white_list
        df["form_black"] = form_black_list
        df["form_diff"] = df["form_white"] - df["form_black"]

        return df

    def _calculate_tpr(self, df, player_id, current_pos):
        past_slice = df.iloc[:current_pos]
        past = past_slice.loc[
            (past_slice["white_id"] == player_id)
            | (past_slice["black_id"] == player_id)
        ].tail(self.window)

        n = len(past)
        if n == 0:
            # No history yet: use the player's current Elo as neutral TPR estimate
            cur = df.iloc[current_pos]
            if int(cur["white_id"]) == player_id:
                return float(cur["white_elo"])
            return float(cur["black_elo"])

        scores = []
        opponent_elos = []
        for _, m in past.iterrows():
            if pd.isna(m["result"]):
                continue  # skip prediction rows
            if m["white_id"] == player_id:
                scores.append(float(m["result"]))
                opponent_elos.append(m["black_elo"])
            else:
                scores.append(1.0 - float(m["result"]))
                opponent_elos.append(m["white_elo"])

        avg_opponent_elo = np.mean(opponent_elos)
        avg_score = np.mean(scores)

        if avg_score >= 1.0:
            return avg_opponent_elo + 400
        if avg_score <= 0.0:
            return avg_opponent_elo - 400

        tpr = avg_opponent_elo + 400 * math.log10(avg_score / (1 - avg_score))
        return tpr
