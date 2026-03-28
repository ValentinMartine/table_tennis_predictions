"""
Features de forme récente et fatigue.

Features produites :
- form_p1, form_p2             : win rate sur les N derniers matchs
- form_diff                    : form_p1 - form_p2
- avg_sets_p1, avg_sets_p2     : sets gagnés en moyenne (dernier N)
- rest_hours_p1, rest_hours_p2 : heures depuis le dernier match
- fatigue_p1, fatigue_p2       : 1 si <48h depuis dernier match
"""
from datetime import timedelta

import numpy as np
import pandas as pd


class FormCalculator:
    def __init__(
        self,
        window: int = 5,
        decay: float = 0.85,
        fatigue_threshold_hours: int = 48,
    ):
        self.window = window
        self.decay = decay
        self.fatigue_threshold_hours = fatigue_threshold_hours

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("played_at").copy()

        form_p1, form_p2 = [], []
        avg_sets_p1, avg_sets_p2 = [], []
        rest_p1, rest_p2 = [], []

        for idx, row in df.iterrows():
            p1_id = int(row["player1_id"])
            p2_id = int(row["player2_id"])
            match_date = pd.Timestamp(row["played_at"])

            past = df[df.index < idx].copy()

            form_p1.append(self._form(past, p1_id, match_date))
            form_p2.append(self._form(past, p2_id, match_date))
            avg_sets_p1.append(self._avg_sets(past, p1_id))
            avg_sets_p2.append(self._avg_sets(past, p2_id))
            rest_p1.append(self._rest_hours(past, p1_id, match_date))
            rest_p2.append(self._rest_hours(past, p2_id, match_date))

        df["form_p1"] = form_p1
        df["form_p2"] = form_p2
        df["form_diff"] = df["form_p1"] - df["form_p2"]
        df["avg_sets_p1"] = avg_sets_p1
        df["avg_sets_p2"] = avg_sets_p2
        df["rest_hours_p1"] = rest_p1
        df["rest_hours_p2"] = rest_p2
        df["fatigue_p1"] = (df["rest_hours_p1"] < self.fatigue_threshold_hours).astype(int)
        df["fatigue_p2"] = (df["rest_hours_p2"] < self.fatigue_threshold_hours).astype(int)

        return df

    def _player_past_matches(self, df: pd.DataFrame, player_id: int) -> pd.DataFrame:
        """Tous les matchs passés impliquant ce joueur."""
        mask = (df["player1_id"] == player_id) | (df["player2_id"] == player_id)
        return df[mask].sort_values("played_at").tail(self.window)

    def _form(self, past: pd.DataFrame, player_id: int, match_date) -> float:
        player_matches = self._player_past_matches(past, player_id)
        if player_matches.empty:
            return 0.5

        wins = []
        for _, m in player_matches.iterrows():
            if m["player1_id"] == player_id:
                wins.append(1.0 if m["winner"] == 1 else 0.0)
            else:
                wins.append(1.0 if m["winner"] == 2 else 0.0)

        # Pondération exponentielle (plus récent = plus de poids)
        n = len(wins)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        return float(np.average(wins, weights=weights))

    def _avg_sets(self, past: pd.DataFrame, player_id: int) -> float:
        player_matches = self._player_past_matches(past, player_id)
        if player_matches.empty:
            return 0.0

        sets = []
        for _, m in player_matches.iterrows():
            if m["player1_id"] == player_id:
                sets.append(m.get("score_p1", 0) or 0)
            else:
                sets.append(m.get("score_p2", 0) or 0)

        return float(np.mean(sets))

    def _rest_hours(
        self, past: pd.DataFrame, player_id: int, match_date: pd.Timestamp
    ) -> float:
        player_matches = self._player_past_matches(past, player_id)
        if player_matches.empty:
            return 999.0  # jamais joué → pas de fatigue
        last_match_time = pd.Timestamp(player_matches.iloc[-1]["played_at"])
        delta = match_date - last_match_time
        return max(0.0, delta.total_seconds() / 3600)
