import numpy as np
import pandas as pd
from datetime import timedelta


class ChessH2HCalculator:
    def __init__(
        self, min_matches: int = 1, max_age_days: int = 1825, decay_rate: float = 0.0005
    ):
        self.min_matches = min_matches
        self.max_age_days = max_age_days
        self.decay_rate = decay_rate

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates head-to-head features for each match.
        Result is points scored by the WHITE player in the current match index context.
        """
        df = df.sort_values("played_at").reset_index(drop=True).copy()

        h2h_matches_list = []
        h2h_points_white_list = []
        h2h_recent_points_white_list = []

        for pos, (idx, row) in enumerate(df.iterrows()):
            w_id = int(row["white_id"])
            b_id = int(row["black_id"])
            match_date = pd.Timestamp(row["played_at"])
            cutoff = match_date - timedelta(days=self.max_age_days)

            past_slice = df.iloc[:pos]
            past = past_slice.loc[
                (past_slice["played_at"] >= cutoff)
                & (
                    (
                        (past_slice["white_id"] == w_id)
                        & (past_slice["black_id"] == b_id)
                    )
                    | (
                        (past_slice["white_id"] == b_id)
                        & (past_slice["black_id"] == w_id)
                    )
                )
            ].copy()

            # Only count past matches with known results
            past = past[pd.notna(past["result"])].copy()
            n = len(past)
            h2h_matches_list.append(n)

            if n < self.min_matches:
                h2h_points_white_list.append(0.5)
                h2h_recent_points_white_list.append(0.5)
                continue

            # Points scored by WHITE player in previous matches
            p_white_points = []
            for _, p_match in past.iterrows():
                if p_match["white_id"] == w_id:
                    p_white_points.append(float(p_match["result"]))
                else:
                    p_white_points.append(1.0 - float(p_match["result"]))

            h2h_points_white_list.append(np.mean(p_white_points))

            # Weighted H2H (recent > ancient)
            days_ago = (match_date - past["played_at"]).dt.days.values
            weights = np.exp(-self.decay_rate * days_ago)
            if np.sum(weights) > 0:
                recent_score = np.average(p_white_points, weights=weights)
            else:
                recent_score = np.mean(p_white_points)
            h2h_recent_points_white_list.append(recent_score)

        df["h2h_matches"] = h2h_matches_list
        df["h2h_points_white"] = h2h_points_white_list
        df["h2h_recent_points_white"] = h2h_recent_points_white_list

        return df
