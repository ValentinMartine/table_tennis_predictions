"""
Pipeline de prédiction live.

Récupère les matchs à venir depuis BetsAPI,
calcule les features, prédit, filtre par edge,
et envoie les alertes.
"""
import asyncio
from datetime import datetime, timedelta

import yaml
from loguru import logger

from ..backtesting.kelly import compute_stake, model_edge
from ..database.db import get_session, init_db
from ..database.models import BettingRecord
from ..models.lgbm_model import LGBMModel
from ..scraping.betsapi import BetsAPIScraper
from .alerts import TelegramAlerter


class LivePredictor:
    def __init__(self, config_path: str = "config/settings.yaml", model_path: str = "data/lgbm_model.pkl"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.bet_cfg = self.config.get("betting", {})
        self.model = LGBMModel.load(model_path)
        self.alerter = TelegramAlerter(config_path)
        self.betsapi = BetsAPIScraper(self.config.get("scraping", {}))
        self.bankroll = self._get_current_bankroll()

    def _get_current_bankroll(self) -> float:
        """Récupère le bankroll actuel depuis les enregistrements de paris."""
        initial = 100.0  # bankroll initial — à configurer
        with get_session() as session:
            from sqlalchemy import func
            total_pl = session.query(
                func.sum(BettingRecord.profit_loss)
            ).filter(BettingRecord.is_paper == False).scalar()
        return initial + (total_pl or 0.0)

    def run(self) -> None:
        """Point d'entrée principal — à appeler via cron."""
        logger.info("Démarrage du pipeline de prédiction live")
        init_db()

        live_events = self.betsapi.get_live_events()
        if not live_events:
            logger.info("Aucun match en cours")
            return

        bets = []
        for event in live_events:
            try:
                bet = self._process_event(event)
                if bet:
                    bets.append(bet)
            except Exception as e:
                logger.error(f"Erreur traitement événement : {e}")

        if bets:
            asyncio.run(self._send_alerts(bets))

        logger.info(f"{len(bets)} opportunités trouvées")

    def _process_event(self, event: dict) -> dict | None:
        """Traite un événement live et retourne un pari si edge suffisant."""
        event_id = str(event.get("id", ""))
        odds = self.betsapi.get_odds(event_id)
        if not odds.get("p1") or not odds.get("p2"):
            return None

        p1_name = event.get("home", {}).get("name", "P1")
        p2_name = event.get("away", {}).get("name", "P2")

        # Prédiction via features actuelles
        # NOTE : en prod, on reconstruit les features depuis la DB pour ces joueurs
        # Ici on utilise une approximation via elo_win_prob implicite des odds
        implied_prob_p1 = (1 / odds["p1"]) / (1 / odds["p1"] + 1 / odds["p2"])

        # Construction d'un sample minimal pour la prédiction
        import pandas as pd
        sample = pd.DataFrame([{
            "elo_diff": 0.0,
            "elo_win_prob_p1": implied_prob_p1,
            "h2h_matches": 0,
            "h2h_winrate_p1": 0.5,
            "h2h_recent_winrate_p1": 0.5,
            "form_p1": 0.5,
            "form_p2": 0.5,
            "form_diff": 0.0,
            "avg_sets_p1": 2.0,
            "avg_sets_p2": 2.0,
            "rest_hours_p1": 48.0,
            "rest_hours_p2": 48.0,
            "fatigue_p1": 0,
            "fatigue_p2": 0,
            "ittf_rank_p1": 9999,
            "ittf_rank_p2": 9999,
            "rank_diff": 0,
            "age_p1": 25.0,
            "age_p2": 25.0,
            "age_diff": 0.0,
            "implied_prob_p1": implied_prob_p1,
        }])
        pred_prob = float(self.model.predict_proba(sample)[0])

        edge = model_edge(pred_prob, odds["p1"])
        if abs(edge) < self.bet_cfg.get("min_edge", 0.03):
            return None

        bet_player = 1 if edge > 0 else 2
        bet_prob = pred_prob if bet_player == 1 else 1 - pred_prob
        bet_odds = odds["p1"] if bet_player == 1 else odds["p2"]
        bet_edge = model_edge(bet_prob, bet_odds)

        if bet_edge < self.bet_cfg.get("min_edge", 0.03):
            return None

        stake = compute_stake(
            self.bankroll, bet_prob, bet_odds,
            self.bet_cfg.get("kelly_fraction", 0.25),
            self.bet_cfg.get("max_stake_pct", 0.02),
        )

        return {
            "event_id": event_id,
            "match": f"{p1_name} vs {p2_name}",
            "competition": event.get("league", {}).get("name", "TT"),
            "bet_on": p1_name if bet_player == 1 else p2_name,
            "bet_player": bet_player,
            "prob": bet_prob,
            "odds": bet_odds,
            "edge": bet_edge,
            "stake": stake,
            "stake_pct": round(stake / self.bankroll * 100, 2),
            "played_at": datetime.utcnow().strftime("%H:%M UTC"),
        }

    async def _send_alerts(self, bets: list[dict]) -> None:
        for bet in bets:
            await self.alerter.send_bet_alert(bet)
