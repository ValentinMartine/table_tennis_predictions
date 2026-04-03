from dataclasses import dataclass
import pandas as pd


@dataclass
class EloState:
    rating: float = 1500.0
    matches_played: int = 0


def expected_score(ra: float, rb: float) -> float:
    """Calculates the expected score for player A."""
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(
    ra: float,
    rb: float,
    score: float,  # 1.0, 0.5, 0.0
    k: float = 20.0,
) -> tuple[float, float]:
    """Updates Elo ratings for two players based on a match result."""
    ea = expected_score(ra, rb)
    eb = 1 - ea

    new_ra = ra + k * (score - ea)
    new_rb = rb + k * ((1 - score) - eb)

    return new_ra, new_rb


class ChessEloCalculator:
    def __init__(
        self,
        initial_rating: float = 1500.0,
        k: float = 20.0,
        initial_ratings_dict: dict = None,
    ):
        self.default_initial_rating = initial_rating
        self.k = k
        self.initial_ratings_dict = initial_ratings_dict or {}
        self.ratings: dict[int, EloState] = {}

    def _k_for(self, rating: float) -> float:
        if rating >= 2700:
            return 10.0
        if rating >= 2500:
            return 15.0
        return self.k

    def _get_state(self, player_id: int) -> EloState:
        # Force integer casting for dictionary lookup to avoid str/int mismatches
        pid = int(player_id)
        if pid not in self.ratings:
            # Use provided initial rating from config if available, otherwise default to 1500
            start_rating = self.initial_ratings_dict.get(
                pid, self.default_initial_rating
            )
            self.ratings[pid] = EloState(rating=float(start_rating))
        return self.ratings[pid]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich a dataframe with Elo features.
        Expects columns: [white_id, black_id, result, played_at]
        Result: 1.0 (White Win), 0.5 (Draw), 0.0 (Black Win)
        """
        df = df.sort_values("played_at").copy()

        white_elo_list, black_elo_list = [], []

        for idx, row in df.iterrows():
            w_id = int(row["white_id"])
            b_id = int(row["black_id"])
            result = row["result"]

            s_w = self._get_state(w_id)
            s_b = self._get_state(b_id)

            white_elo_list.append(s_w.rating)
            black_elo_list.append(s_b.rating)

            # Only update ELO for completed matches (result known)
            if pd.notna(result):
                res = float(result)
                k = (self._k_for(s_w.rating) + self._k_for(s_b.rating)) / 2
                new_rw, new_rb = update_elo(s_w.rating, s_b.rating, res, k)
                s_w.rating = new_rw
                s_b.rating = new_rb
                s_w.matches_played += 1
                s_b.matches_played += 1

        df["white_elo"] = white_elo_list
        df["black_elo"] = black_elo_list
        df["elo_diff"] = df["white_elo"] - df["black_elo"]
        df["elo_prob_white"] = df.apply(
            lambda r: expected_score(r["white_elo"], r["black_elo"]), axis=1
        )

        return df
