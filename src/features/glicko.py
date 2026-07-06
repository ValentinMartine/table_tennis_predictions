"""
Ratings Glicko-2 par joueur, calculés chronologiquement (causal, sans fuite).

Glicko-2 enrichit l'Elo de deux quantités absentes d'un rating simple :
- RD (rating deviation) : incertitude sur le niveau estimé. Élevé pour un joueur
  nouveau ou inactif, décroît avec les matchs.
- volatilité (sigma)    : degré d'irrégularité récente des performances.

Chaque match est traité comme une période de notation d'un seul jeu pour chaque
joueur (variante incrémentale usuelle), avec inflation du RD selon le temps écoulé
depuis le dernier match (un joueur inactif redevient incertain).

Features produites (état AVANT le match, donc utilisables en prédiction) :
  glicko_p1, glicko_p2, glicko_diff, glicko_win_prob_p1,
  glicko_rd_p1, glicko_rd_p2, glicko_rd_diff,
  glicko_vol_p1, glicko_vol_p2

Réf. : Glickman, "Example of the Glicko-2 system" (2013).
"""
import math
from dataclasses import dataclass

import pandas as pd

_SCALE = 173.7178  # conversion échelle Glicko <-> Glicko-2
_PI2 = math.pi ** 2


@dataclass
class Glicko2State:
    rating: float = 1500.0
    rd: float = 350.0
    vol: float = 0.06
    last_match_ns: int = 0
    matches_played: int = 0


class Glicko2Calculator:
    """
    Calcule les ratings Glicko-2 pour tous les matchs d'un DataFrame trié
    chronologiquement. Un seul track global (tous matchs confondus).
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        initial_vol: float = 0.06,
        tau: float = 0.5,
        rating_period_days: float = 30.0,
        max_rd: float = 350.0,
    ):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_vol = initial_vol
        self.tau = tau
        self.rating_period_days = rating_period_days
        self.max_rd = max_rd
        self._states: dict[int, Glicko2State] = {}

    def _get_state(self, player_id: int) -> Glicko2State:
        if player_id not in self._states:
            self._states[player_id] = Glicko2State(
                rating=self.initial_rating, rd=self.initial_rd, vol=self.initial_vol
            )
        return self._states[player_id]

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / _PI2)

    @staticmethod
    def _E(mu: float, mu_j: float, phi_j: float) -> float:
        return 1.0 / (1.0 + math.exp(-Glicko2Calculator._g(phi_j) * (mu - mu_j)))

    def _inflate_rd(self, state: Glicko2State, match_ns: int) -> float:
        """RD gonflé selon l'inactivité : phi = sqrt(phi^2 + vol^2 * t)."""
        phi = state.rd / _SCALE
        if state.last_match_ns and match_ns > state.last_match_ns:
            days = (match_ns - state.last_match_ns) / 8.64e13  # ns -> jours
            t = max(0.0, days / self.rating_period_days)
            phi = math.sqrt(phi * phi + (state.vol ** 2) * t)
        rd = min(phi * _SCALE, self.max_rd)
        return rd

    def _win_prob(self, sa: Glicko2State, sb: Glicko2State) -> float:
        """Probabilité de victoire de A intégrant l'incertitude combinée des deux RD."""
        mu_a = (sa.rating - 1500.0) / _SCALE
        mu_b = (sb.rating - 1500.0) / _SCALE
        phi = math.sqrt((sa.rd / _SCALE) ** 2 + (sb.rd / _SCALE) ** 2)
        return 1.0 / (1.0 + math.exp(-self._g(phi) * (mu_a - mu_b)))

    def _update_one(self, state: Glicko2State, rd_now: float, opp: Glicko2State, score: float) -> None:
        """Met à jour un joueur après un match (période d'un jeu)."""
        mu = (state.rating - 1500.0) / _SCALE
        phi = rd_now / _SCALE
        mu_j = (opp.rating - 1500.0) / _SCALE
        phi_j = opp.rd / _SCALE

        g_j = self._g(phi_j)
        e = 1.0 / (1.0 + math.exp(-g_j * (mu - mu_j)))
        v = 1.0 / (g_j * g_j * e * (1.0 - e))
        delta = v * g_j * (score - e)

        # Nouvelle volatilité par itération d'Illinois
        a = math.log(state.vol ** 2)
        d2 = delta * delta
        phi2 = phi * phi

        def f(x: float) -> float:
            ex = math.exp(x)
            num = ex * (d2 - phi2 - v - ex)
            den = 2.0 * (phi2 + v + ex) ** 2
            return num / den - (x - a) / (self.tau ** 2)

        A = a
        if d2 > phi2 + v:
            B = math.log(d2 - phi2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        fA, fB = f(A), f(B)
        for _ in range(100):
            if abs(B - A) <= 1e-6:
                break
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A, fA = B, fB
            else:
                fA /= 2.0
            B, fB = C, fC
        new_vol = math.exp(A / 2.0)

        phi_star = math.sqrt(phi2 + new_vol ** 2)
        new_phi = 1.0 / math.sqrt(1.0 / (phi_star ** 2) + 1.0 / v)
        new_mu = mu + new_phi ** 2 * g_j * (score - e)

        state.rating = 1500.0 + _SCALE * new_mu
        state.rd = min(_SCALE * new_phi, self.max_rd)
        state.vol = new_vol
        state.matches_played += 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: colonnes [player1_id, player2_id, winner, played_at] triées ASC.
        Returns:
            df enrichi des colonnes Glicko-2 (état avant match).
        """
        df = df.sort_values("played_at").copy()
        match_ns = df["played_at"].values.astype("int64")
        p1 = df["player1_id"].values
        p2 = df["player2_id"].values
        winner = df["winner"].values

        g1, g2, rd1, rd2, vol1, vol2, wp = ([] for _ in range(7))

        for i in range(len(df)):
            s1 = self._get_state(int(p1[i]))
            s2 = self._get_state(int(p2[i]))
            m_ns = int(match_ns[i])

            # RD gonflé par l'inactivité, appliqué avant lecture des features
            rd1_now = self._inflate_rd(s1, m_ns)
            rd2_now = self._inflate_rd(s2, m_ns)
            s1.rd, s2.rd = rd1_now, rd2_now

            g1.append(s1.rating)
            g2.append(s2.rating)
            rd1.append(rd1_now)
            rd2.append(rd2_now)
            vol1.append(s1.vol)
            vol2.append(s2.vol)
            wp.append(self._win_prob(s1, s2))

            s1_score = 1.0 if int(winner[i]) == 1 else 0.0
            # snapshot des adversaires avant mise à jour croisée
            opp1 = Glicko2State(rating=s2.rating, rd=rd2_now, vol=s2.vol)
            opp2 = Glicko2State(rating=s1.rating, rd=rd1_now, vol=s1.vol)
            self._update_one(s1, rd1_now, opp1, s1_score)
            self._update_one(s2, rd2_now, opp2, 1.0 - s1_score)
            s1.last_match_ns = m_ns
            s2.last_match_ns = m_ns

        df["glicko_p1"] = g1
        df["glicko_p2"] = g2
        df["glicko_diff"] = df["glicko_p1"] - df["glicko_p2"]
        df["glicko_rd_p1"] = rd1
        df["glicko_rd_p2"] = rd2
        df["glicko_rd_diff"] = df["glicko_rd_p1"] - df["glicko_rd_p2"]
        df["glicko_vol_p1"] = vol1
        df["glicko_vol_p2"] = vol2
        df["glicko_win_prob_p1"] = wp
        return df
