"""
Features de forme récente et fatigue.

Features produites :
- form_p1, form_p2             : win rate sur les N derniers matchs
- form_diff                    : form_p1 - form_p2
- avg_sets_p1, avg_sets_p2     : sets gagnés en moyenne (dernier N)
- avg_set_margin_p1/p2         : marge de points moyenne par set (domination)
- close_sets_rate_p1/p2        : fraction de sets serrés (écart <= 3 pts)
- set_margin_diff              : avg_set_margin_p1 - avg_set_margin_p2
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
        set_margin_p1, set_margin_p2 = [], []
        close_sets_p1, close_sets_p2 = [], []
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
            sm1, cs1 = self._set_stats(past, p1_id)
            sm2, cs2 = self._set_stats(past, p2_id)
            set_margin_p1.append(sm1)
            set_margin_p2.append(sm2)
            close_sets_p1.append(cs1)
            close_sets_p2.append(cs2)
            rest_p1.append(self._rest_hours(past, p1_id, match_date))
            rest_p2.append(self._rest_hours(past, p2_id, match_date))

        df["form_p1"] = form_p1
        df["form_p2"] = form_p2
        df["form_diff"] = df["form_p1"] - df["form_p2"]
        df["avg_sets_p1"] = avg_sets_p1
        df["avg_sets_p2"] = avg_sets_p2
        df["avg_set_margin_p1"] = set_margin_p1
        df["avg_set_margin_p2"] = set_margin_p2
        df["set_margin_diff"] = df["avg_set_margin_p1"] - df["avg_set_margin_p2"]
        df["close_sets_rate_p1"] = close_sets_p1
        df["close_sets_rate_p2"] = close_sets_p2
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

    @staticmethod
    def _parse_sets_detail(detail) -> list[tuple[int, int]]:
        """Parse '11-8,9-11,11-6' → [(11,8), (9,11), (11,6)]. Retourne [] si invalide."""
        if not detail or (isinstance(detail, float) and detail != detail):
            return []
        try:
            result = []
            for s in str(detail).strip().split(","):
                parts = s.strip().split("-")
                if len(parts) == 2:
                    result.append((int(parts[0]), int(parts[1])))
            return result
        except (ValueError, AttributeError):
            return []

    def _set_stats(self, past: pd.DataFrame, player_id: int) -> tuple[float, float]:
        """
        Calcule la marge moyenne par set et le taux de sets serrés sur les N derniers matchs.
        Retourne (avg_set_margin, close_sets_rate).
        - avg_set_margin : écart de points moyen par set (tous sets confondus)
        - close_sets_rate : fraction de sets avec écart <= 3 points
        """
        player_matches = self._player_past_matches(past, player_id)
        all_margins = []

        for _, m in player_matches.iterrows():
            is_p1 = (m["player1_id"] == player_id)
            sets = self._parse_sets_detail(m.get("sets_detail", ""))
            for a, b in sets:
                # On normalise toujours du point de vue du joueur étudié
                score_self, score_opp = (a, b) if is_p1 else (b, a)
                all_margins.append(abs(score_self - score_opp))

        if not all_margins:
            return 2.5, 0.3   # valeurs neutres par défaut

        avg_margin = float(np.mean(all_margins))
        close_rate = float(sum(1 for m in all_margins if m <= 3) / len(all_margins))
        return avg_margin, close_rate

    def _rest_hours(
        self, past: pd.DataFrame, player_id: int, match_date: pd.Timestamp
    ) -> float:
        player_matches = self._player_past_matches(past, player_id)
        if player_matches.empty:
            return 999.0  # jamais joué → pas de fatigue
        last_match_time = pd.Timestamp(player_matches.iloc[-1]["played_at"])
        delta = match_date - last_match_time
        return max(0.0, delta.total_seconds() / 3600)
