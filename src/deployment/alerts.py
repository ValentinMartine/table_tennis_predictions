"""
Alertes Telegram pour les opportunités de paris.
"""
import os

import yaml
from loguru import logger
from telegram import Bot
from telegram.constants import ParseMode


class TelegramAlerter:
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        tg_cfg = config.get("telegram", {})

        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.min_edge = tg_cfg.get("alert_min_edge", 0.05)
        self._bot: Bot | None = None

    def _get_bot(self) -> Bot:
        if not self._bot:
            if not self.token:
                raise ValueError("TELEGRAM_BOT_TOKEN non configuré")
            self._bot = Bot(token=self.token)
        return self._bot

    async def send_bet_alert(self, bet: dict) -> None:
        """
        Envoie une alerte pari.

        bet: {
            match: "Player A vs Player B",
            competition: "Setka Cup",
            bet_on: "Player A",
            prob: 0.62,
            odds: 1.85,
            edge: 0.08,
            stake_pct: 1.5,
            played_at: datetime
        }
        """
        if bet.get("edge", 0) < self.min_edge:
            return

        edge_pct = round(bet["edge"] * 100, 1)
        prob_pct = round(bet["prob"] * 100, 1)

        msg = (
            f"🏓 *PARI TT* — {bet.get('competition', 'N/A')}\n"
            f"Match : {bet['match']}\n"
            f"Pari : *{bet['bet_on']}*\n"
            f"Proba : {prob_pct}% | Cote : {bet['odds']:.2f}\n"
            f"Edge : +{edge_pct}%\n"
            f"Mise : {bet.get('stake_pct', 'N/A')}% bankroll\n"
            f"Heure : {bet.get('played_at', 'N/A')}"
        )

        try:
            bot = self._get_bot()
            await bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.info(f"Alerte Telegram envoyée : {bet['match']}")
        except Exception as e:
            logger.error(f"Erreur Telegram : {e}")

    async def send_daily_report(self, stats: dict) -> None:
        """Rapport journalier."""
        msg = (
            f"📊 *Rapport journalier TT*\n"
            f"Paris : {stats.get('n_bets_today', 0)}\n"
            f"ROI aujourd'hui : {stats.get('roi_today_pct', 0):.1f}%\n"
            f"ROI total : {stats.get('roi_total_pct', 0):.1f}%\n"
            f"Bankroll : {stats.get('bankroll', 0):.0f}€\n"
            f"Win rate (30j) : {stats.get('win_rate_pct', 0):.1f}%"
        )
        try:
            bot = self._get_bot()
            await bot.send_message(
                chat_id=self.chat_id, text=msg, parse_mode=ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"Erreur rapport Telegram : {e}")
