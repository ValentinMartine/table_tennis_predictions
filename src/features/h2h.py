"""
Features head-to-head (H2H).

Features produites :
- h2h_matches          : nombre de confrontations précédentes
- h2h_winrate_p1       : taux de victoire P1 sur matchs H2H
- h2h_recent_winrate   : H2H pondéré exponentiel (récent > ancien)
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class H2HCalculator:
    def __init__(
        self,
        min_matches: int = 1,
        max_age_days: int = 1825,
        decay_rate: float = 0.001,   # par jour
    ):
        self.min_matches = min_matches
        self.max_age_days = max_age_days
        self.decay_rate = decay_rate

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les features H2H pour chaque match.
        Le DataFrame doit être trié chronologiquement.
        """
        df = df.sort_values("played_at").copy()

        h2h_matches_list = []
        h2h_winrate_list = []
        h2h_recent_winrate_list = []

        for idx, row in df.iterrows():
            p1_id = int(row["player1_id"])
            p2_id = int(row["player2_id"])
            match_date = pd.Timestamp(row["played_at"])
            cutoff = match_date - timedelta(days=self.max_age_days)

            # Matchs H2H antérieurs (dans les deux sens)
            past = df.loc[
                (df.index < idx)
                & (df["played_at"] >= cutoff)
                & (
                    ((df["player1_id"] == p1_id) & (df["player2_id"] == p2_id))
                    | ((df["player1_id"] == p2_id) & (df["player2_id"] == p1_id))
                )
            ].copy()

            n = len(past)
            h2h_matches_list.append(n)

            if n < self.min_matches:
                h2h_winrate_list.append(0.5)
                h2h_recent_winrate_list.append(0.5)
                continue

            # Victoires de P1
            p1_wins = (
                ((past["player1_id"] == p1_id) & (past["winner"] == 1))
                | ((past["player2_id"] == p1_id) & (past["winner"] == 2))
            ).sum()
            h2h_winrate_list.append(p1_wins / n)

            # H2H pondéré exponentiel
            days_ago = (match_date - past["played_at"]).dt.days.clip(lower=0).values
            weights = np.exp(-self.decay_rate * days_ago)
            p1_win_mask = (
                ((past["player1_id"] == p1_id) & (past["winner"] == 1))
                | ((past["player2_id"] == p1_id) & (past["winner"] == 2))
            ).values.astype(float)
            recent_winrate = np.average(p1_win_mask, weights=weights)
            h2h_recent_winrate_list.append(recent_winrate)

        df["h2h_matches"] = h2h_matches_list
        df["h2h_winrate_p1"] = h2h_winrate_list
        df["h2h_recent_winrate_p1"] = h2h_recent_winrate_list

        return df
