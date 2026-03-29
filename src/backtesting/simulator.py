"""
Simulateur de backtesting.

Rejoue les paris sur des données historiques out-of-sample
et calcule les métriques de performance (ROI, Sharpe, drawdown).
"""
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from .kelly import compute_stake, model_edge


class BettingSimulator:
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        bet_cfg = config.get("betting", {})

        self.kelly_fraction = bet_cfg.get("kelly_fraction", 0.25)
        self.min_edge = bet_cfg.get("min_edge", 0.03)
        self.min_confidence = bet_cfg.get("min_confidence", 0.55)
        self.max_stake_pct = bet_cfg.get("max_stake_pct", 0.02)
        self.min_odds = bet_cfg.get("min_odds", 1.60)
        self.max_odds = bet_cfg.get("max_odds", 3.50)

    def run(
        self,
        df: pd.DataFrame,
        initial_bankroll: float = 1000.0,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Simule les paris sur le DataFrame.

        Le DataFrame doit contenir :
            played_at, target (winner=P1 → 1),
            pred_prob_p1, odds_p1, odds_p2

        Returns:
            (df_bets, stats)
        """
        df = df.sort_values("played_at").copy()
        bankroll = initial_bankroll
        records = []

        for _, row in df.iterrows():
            if pd.isna(row.get("odds_p1")) or pd.isna(row.get("odds_p2")):
                continue

            prob_p1 = float(row["pred_prob_p1"])
            odds_p1 = float(row["odds_p1"])
            odds_p2 = float(row["odds_p2"])
            actual_winner = int(row["target"])  # 1 si P1 gagne

            # Cherche la meilleure opportunité (P1 ou P2)
            best = self._find_best_bet(prob_p1, odds_p1, odds_p2)
            if best is None:
                continue

            bet_player, prob, odds = best
            stake = compute_stake(
                bankroll, prob, odds, self.kelly_fraction, self.max_stake_pct
            )
            if stake < 0.01:
                continue

            won = (bet_player == 1 and actual_winner == 1) or (
                bet_player == 2 and actual_winner == 0
            )
            profit = stake * (odds - 1) if won else -stake
            bankroll += profit

            records.append({
                "played_at": row["played_at"],
                "bet_player": bet_player,
                "prob": prob,
                "odds": odds,
                "edge": model_edge(prob, odds),
                "stake": stake,
                "won": won,
                "profit": profit,
                "bankroll": bankroll,
            })

        if not records:
            logger.warning("Aucun pari simulé — vérifier les filtres (edge, odds, confidence)")
            return pd.DataFrame(), {}

        df_bets = pd.DataFrame(records)
        stats = self._compute_stats(df_bets, initial_bankroll)
        return df_bets, stats

    def _find_best_bet(
        self, prob_p1: float, odds_p1: float, odds_p2: float
    ) -> tuple[int, float, float] | None:
        """Retourne (bet_player, prob, odds) ou None si aucune valeur."""
        candidates = []

        edge_p1 = model_edge(prob_p1, odds_p1)
        if (
            prob_p1 >= self.min_confidence
            and edge_p1 >= self.min_edge
            and self.min_odds <= odds_p1 <= self.max_odds
        ):
            candidates.append((1, prob_p1, odds_p1, edge_p1))

        prob_p2 = 1 - prob_p1
        edge_p2 = model_edge(prob_p2, odds_p2)
        if (
            prob_p2 >= self.min_confidence
            and edge_p2 >= self.min_edge
            and self.min_odds <= odds_p2 <= self.max_odds
        ):
            candidates.append((2, prob_p2, odds_p2, edge_p2))

        if not candidates:
            return None
        # Prend le meilleur edge
        best = max(candidates, key=lambda x: x[3])
        return best[0], best[1], best[2]

    @staticmethod
    def _compute_stats(df_bets: pd.DataFrame, initial_bankroll: float) -> dict:
        profits = df_bets["profit"].values
        bankrolls = df_bets["bankroll"].values
        n = len(df_bets)

        roi = (bankrolls[-1] - initial_bankroll) / initial_bankroll
        win_rate = df_bets["won"].mean()
        avg_odds = df_bets["odds"].mean()

        # Sharpe ratio (annualisé via nb de paris)
        mean_return = profits.mean()
        std_return = profits.std()
        sharpe = (mean_return / std_return * np.sqrt(n)) if std_return > 0 else 0.0

        # Drawdown max
        peak = np.maximum.accumulate(bankrolls)
        drawdowns = (bankrolls - peak) / peak
        max_drawdown = drawdowns.min()

        return {
            "n_bets": n,
            "roi_pct": round(roi * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_odds": round(avg_odds, 3),
            "avg_edge_pct": round(df_bets["edge"].mean() * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "final_bankroll": round(bankrolls[-1], 2),
            "profit_total": round(bankrolls[-1] - initial_bankroll, 2),
        }
