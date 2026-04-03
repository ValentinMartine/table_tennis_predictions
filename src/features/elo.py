"""
Calcul des ratings Elo par joueur, chronologiquement.

Features produites :
- elo_p1, elo_p2               : rating global avant le match
- elo_diff                     : elo_p1 - elo_p2
- elo_win_prob_p1              : probabilité Elo de victoire de P1
- elo_intl_p1, elo_intl_p2    : rating calculé uniquement sur matchs priority <= 1
- elo_intl_diff                : elo_intl_p1 - elo_intl_p2
- elo_intl_win_prob_p1         : probabilité Elo international de victoire de P1
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class EloState:
    rating: float = 1500.0
    matches_played: int = 0


def expected_score(ra: float, rb: float) -> float:
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(
    winner_rating: float,
    loser_rating: float,
    k_winner: float,
    k_loser: float,
    comp_weight: float = 1.0,
) -> tuple[float, float]:
    e_winner = expected_score(winner_rating, loser_rating)
    e_loser = 1 - e_winner
    new_winner = winner_rating + k_winner * comp_weight * (1 - e_winner)
    new_loser = loser_rating + k_loser * comp_weight * (0 - e_loser)
    return new_winner, new_loser


class EloCalculator:
    """
    Calcule les ratings Elo pour tous les matchs d'un DataFrame.
    Le DataFrame doit être trié chronologiquement.

    Deux tracks Elo en parallèle :
    - global : mis à jour sur tous les matchs (toutes compétitions)
    - intl   : mis à jour uniquement sur les matchs priority <= 1
               (WTT Champions, Star Contender, Cup Finals, Worlds, JO)
    """

    INTL_PRIORITY_MAX = 1

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_default: float = 32.0,
        k_new_player: float = 40.0,
        new_player_threshold: int = 30,
        competition_weights: dict[str, float] | None = None,
    ):
        self.initial_rating = initial_rating
        self.k_default = k_default
        self.k_new_player = k_new_player
        self.new_player_threshold = new_player_threshold
        self.comp_weights = competition_weights or {}
        self._ratings: dict[int, EloState] = {}
        self._intl_ratings: dict[int, EloState] = {}

    def _get_state(self, player_id: int) -> EloState:
        if player_id not in self._ratings:
            self._ratings[player_id] = EloState(rating=self.initial_rating)
        return self._ratings[player_id]

    def _get_intl_state(self, player_id: int) -> EloState:
        if player_id not in self._intl_ratings:
            self._intl_ratings[player_id] = EloState(rating=self.initial_rating)
        return self._intl_ratings[player_id]

    def _k_factor(self, player_id: int) -> float:
        return self._k_for_state(self._get_state(player_id))

    def _k_for_state(self, state: EloState) -> float:
        return (
            self.k_new_player
            if state.matches_played < self.new_player_threshold
            else self.k_default
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: DataFrame avec colonnes
                [player1_id, player2_id, winner, competition_id, comp_priority, played_at]
        Returns:
            df enrichi des colonnes Elo global et Elo international
        """
        df = df.sort_values("played_at").copy()

        elo_p1_list, elo_p2_list = [], []
        n1_list, n2_list = [], []
        elo_intl_p1_list, elo_intl_p2_list = [], []

        for _, row in df.iterrows():
            p1_id = int(row["player1_id"])
            p2_id = int(row["player2_id"])
            winner = int(row["winner"])
            comp_id = str(row.get("competition_id", ""))
            comp_weight = self.comp_weights.get(comp_id, 1.0)
            comp_priority = int(row.get("comp_priority", 99))

            # ── Elo global ────────────────────────────────────────────────────
            s1 = self._get_state(p1_id)
            s2 = self._get_state(p2_id)

            elo_p1_list.append(s1.rating)
            elo_p2_list.append(s2.rating)
            n1_list.append(s1.matches_played)
            n2_list.append(s2.matches_played)

            k1 = self._k_for_state(s1)
            k2 = self._k_for_state(s2)
            if winner == 1:
                new_r1, new_r2 = update_elo(s1.rating, s2.rating, k1, k2, comp_weight)
            else:
                new_r2, new_r1 = update_elo(s2.rating, s1.rating, k2, k1, comp_weight)
            s1.rating = new_r1
            s2.rating = new_r2
            s1.matches_played += 1
            s2.matches_played += 1

            # ── Elo international (priority <= 1 uniquement) ──────────────────
            si1 = self._get_intl_state(p1_id)
            si2 = self._get_intl_state(p2_id)

            elo_intl_p1_list.append(si1.rating)
            elo_intl_p2_list.append(si2.rating)

            if comp_priority <= self.INTL_PRIORITY_MAX:
                ki1 = self._k_for_state(si1)
                ki2 = self._k_for_state(si2)
                if winner == 1:
                    new_ri1, new_ri2 = update_elo(si1.rating, si2.rating, ki1, ki2, comp_weight)
                else:
                    new_ri2, new_ri1 = update_elo(si2.rating, si1.rating, ki2, ki1, comp_weight)
                si1.rating = new_ri1
                si2.rating = new_ri2
                si1.matches_played += 1
                si2.matches_played += 1

        df["elo_p1"] = elo_p1_list
        df["elo_p2"] = elo_p2_list
        df["matches_played_p1"] = n1_list
        df["matches_played_p2"] = n2_list
        df["elo_diff"] = df["elo_p1"] - df["elo_p2"]
        df["elo_win_prob_p1"] = df.apply(
            lambda r: expected_score(r["elo_p1"], r["elo_p2"]), axis=1
        )

        df["elo_intl_p1"] = elo_intl_p1_list
        df["elo_intl_p2"] = elo_intl_p2_list
        df["elo_intl_diff"] = df["elo_intl_p1"] - df["elo_intl_p2"]
        df["elo_intl_win_prob_p1"] = df.apply(
            lambda r: expected_score(r["elo_intl_p1"], r["elo_intl_p2"]), axis=1
        )

        return df

    def get_current_ratings(self) -> dict[int, float]:
        return {pid: s.rating for pid, s in self._ratings.items()}

    def get_current_intl_ratings(self) -> dict[int, float]:
        return {pid: s.rating for pid, s in self._intl_ratings.items()}
