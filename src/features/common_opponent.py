"""
Features "common-opponent" (Knottenbelt et al., 2012), calculées chronologiquement.

Plutôt que le H2H direct (souvent absent ou à très petit échantillon), on compare
les deux joueurs via les adversaires qu'ils ont TOUS DEUX affrontés dans le passé :
si P1 domine leurs adversaires communs et pas P2, c'est un signal de supériorité
même s'ils ne se sont jamais rencontrés.

Features produites (état AVANT le match) :
  co_n_common          : nombre d'adversaires communs
  co_winrate_diff      : (winrate P1 vs communs) - (winrate P2 vs communs)
  co_winrate_diff_shr  : version shrinkée vers 0 quand peu d'adversaires communs
"""
import pandas as pd


class CommonOpponentCalculator:
    def __init__(self, shrinkage_k: float = 5.0):
        # k contrôle la force du shrinkage : diff * n/(n+k)
        self.shrinkage_k = shrinkage_k
        # player_id -> {opponent_id: [wins, games]}
        self._hist: dict[int, dict[int, list]] = {}

    def _winrate_vs(self, pid: int, opponents: set[int]) -> float:
        rec = self._hist.get(pid, {})
        wins = games = 0
        for o in opponents:
            wg = rec.get(o)
            if wg:
                wins += wg[0]
                games += wg[1]
        return wins / games if games else 0.0

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("played_at").copy()
        p1 = df["player1_id"].values
        p2 = df["player2_id"].values
        winner = df["winner"].values

        n_common, wr_diff, wr_diff_shr = [], [], []

        for i in range(len(df)):
            a, b = int(p1[i]), int(p2[i])
            opp_a = set(self._hist.get(a, {}).keys())
            opp_b = set(self._hist.get(b, {}).keys())
            common = opp_a & opp_b

            n = len(common)
            if n:
                diff = self._winrate_vs(a, common) - self._winrate_vs(b, common)
            else:
                diff = 0.0
            n_common.append(n)
            wr_diff.append(diff)
            wr_diff_shr.append(diff * n / (n + self.shrinkage_k))

            # mise à jour de l'historique avec ce match
            a_win = int(winner[i]) == 1
            self._hist.setdefault(a, {}).setdefault(b, [0, 0])
            self._hist.setdefault(b, {}).setdefault(a, [0, 0])
            self._hist[a][b][1] += 1
            self._hist[b][a][1] += 1
            if a_win:
                self._hist[a][b][0] += 1
            else:
                self._hist[b][a][0] += 1

        df["co_n_common"] = n_common
        df["co_winrate_diff"] = wr_diff
        df["co_winrate_diff_shr"] = wr_diff_shr
        return df
