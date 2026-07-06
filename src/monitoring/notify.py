"""
Notifications d'alerte (Telegram) pour le pipeline autonome.

Sans configuration (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID absents), les appels
sont des no-op silencieux — le pipeline ne casse jamais à cause des alertes.

Usage programmatique :
    from src.monitoring.notify import send_alert
    send_alert("Retrain refusé par l'eval-gate (régression log_loss)")

Usage CLI (pour les workflows CI) :
    python -m src.monitoring.notify "message d'alerte"
"""
import os
import sys

import requests
from loguru import logger


def send_alert(message: str, prefix: str = "🏓 TT-pred") -> bool:
    """Envoie une alerte Telegram. Retourne True si envoyée, False sinon (no-op)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.info(f"[notify désactivé] {message}")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": f"{prefix} — {message}", "disable_notification": False},
            timeout=10,
        )
        if resp.ok:
            return True
        logger.warning(f"Telegram HTTP {resp.status_code} : {resp.text[:150]}")
        return False
    except requests.RequestException as e:
        logger.warning(f"Échec envoi Telegram : {e}")
        return False


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "alerte sans message"
    ok = send_alert(msg)
    sys.exit(0 if ok else 0)  # ne jamais faire échouer le workflow à cause de l'alerte
