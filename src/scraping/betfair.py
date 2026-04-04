"""
Betfair Exchange API — cotes de marché sans marge pour le tennis de table.
Docs : https://developer.betfair.com/exchange-api/

Auth non-interactive : username + password + app_key → session token.
L'endpoint de login Betfair ne nécessite pas de certificat SSL pour les comptes
sans 2FA — suffisant pour un usage programmatique.

Variables d'environnement requises :
    BETFAIR_APP_KEY   — clé d'application (Developer Portal → My API Keys)
    BETFAIR_USERNAME  — email du compte Betfair
    BETFAIR_PASSWORD  — mot de passe

Event Type ID Table Tennis : 9997 (Exchange)
"""
import os
import time
from datetime import datetime, timezone, timedelta

import requests
from loguru import logger

_LOGIN_URL   = "https://identitysso.betfair.com/api/login"
_API_BASE    = "https://api.betfair.com/exchange/betting/rest/v1.0"
_TT_EVENT_ID = "9997"          # Table Tennis sur Betfair Exchange
_SESSION_TTL = 3600 * 7        # token valide ~8h, on renouvelle toutes les 7h

_session_token: str | None = None
_session_expiry: float = 0.0


def _login(app_key: str, username: str, password: str) -> str | None:
    """Obtient un session token via login username/password."""
    try:
        resp = requests.post(
            _LOGIN_URL,
            data={"username": username, "password": password},
            headers={
                "X-Application": app_key,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            timeout=10,
        )
    except requests.RequestException as e:
        logger.warning(f"Betfair login réseau : {e}")
        return None

    if not resp.ok:
        logger.warning(f"Betfair login HTTP {resp.status_code}")
        return None

    data = resp.json()
    if data.get("status") != "SUCCESS":
        logger.warning(f"Betfair login échec : {data.get('error', data)}")
        return None

    token = data.get("token")
    logger.info("Betfair session token obtenu")
    return token


def _get_session(app_key: str, username: str, password: str) -> str | None:
    """Retourne un token valide, en le renouvelant si nécessaire."""
    global _session_token, _session_expiry
    if _session_token and time.time() < _session_expiry:
        return _session_token
    token = _login(app_key, username, password)
    if token:
        _session_token = token
        _session_expiry = time.time() + _SESSION_TTL
    return token


def _api_post(endpoint: str, payload: dict, app_key: str, token: str) -> dict | list | None:
    """Appel générique à l'API Exchange (JSON-RPC REST)."""
    try:
        resp = requests.post(
            f"{_API_BASE}/{endpoint}/",
            json=payload,
            headers={
                "X-Application": app_key,
                "X-Authentication": token,
                "content-type": "application/json",
                "Accept": "application/json",
            },
            timeout=15,
        )
    except requests.RequestException as e:
        logger.warning(f"Betfair API {endpoint} réseau : {e}")
        return None

    if not resp.ok:
        logger.warning(f"Betfair API {endpoint} HTTP {resp.status_code} : {resp.text[:200]}")
        return None

    return resp.json()


def get_tt_upcoming_markets(app_key: str, token: str, days_ahead: int = 14) -> list[dict]:
    """
    Liste les marchés MATCH_ODDS TT à venir.
    Retourne une liste de {marketId, event, runners: [{selectionId, runnerName}]}.
    """
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)

    catalogue = _api_post("listMarketCatalogue", {
        "filter": {
            "eventTypeIds": [_TT_EVENT_ID],
            "marketTypeCodes": ["MATCH_ODDS"],
            "marketStartTime": {
                "from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to":   end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        },
        "marketProjection": ["EVENT", "RUNNER_DESCRIPTION"],
        "maxResults": 200,
        "locale": "en",
    }, app_key, token)

    if not isinstance(catalogue, list):
        logger.warning(f"listMarketCatalogue réponse inattendue : {catalogue}")
        return []

    markets = []
    for m in catalogue:
        runners = [
            {"selectionId": r["selectionId"], "runnerName": r["runnerName"]}
            for r in m.get("runners", [])
            if r.get("runnerName") != "The Draw"
        ]
        if len(runners) == 2:
            markets.append({
                "marketId": m["marketId"],
                "event": m.get("event", {}),
                "runners": runners,
            })

    logger.info(f"Betfair : {len(markets)} marchés MATCH_ODDS TT trouvés")
    return markets


def get_best_prices(market_ids: list[str], app_key: str, token: str) -> dict[str, dict]:
    """
    Récupère les meilleures cotes back disponibles pour une liste de marchés.
    Retourne {marketId: {selectionId: best_back_price}}.
    """
    if not market_ids:
        return {}

    # Betfair limite à 40 marchés par appel
    result = {}
    for i in range(0, len(market_ids), 40):
        chunk = market_ids[i:i + 40]
        books = _api_post("listMarketBook", {
            "marketIds": chunk,
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS"],
                "exBestOffersOverrides": {"bestPricesDepth": 1},
            },
        }, app_key, token)

        if not isinstance(books, list):
            continue

        for book in books:
            mid = book.get("marketId")
            runners_prices = {}
            for runner in book.get("runners", []):
                sid = runner.get("selectionId")
                ex = runner.get("ex", {})
                back = ex.get("availableToBack", [])
                price = float(back[0]["price"]) if back else 0.0
                if price > 1.0:
                    runners_prices[sid] = price
            if runners_prices:
                result[mid] = runners_prices

    return result


def get_table_tennis_odds_betfair(
    app_key: str, username: str, password: str, days_ahead: int = 14
) -> list[dict]:
    """
    Point d'entrée principal. Retourne les marchés TT avec cotes exchange.

    Chaque élément :
        {
          "home": str,   "away": str,
          "odds_home": float,  "odds_away": float,
          "market_id": str,
          "source": "betfair_exchange",
        }
    """
    if not all([app_key, username, password]):
        logger.warning("Betfair : BETFAIR_APP_KEY / USERNAME / PASSWORD manquants")
        return []

    token = _get_session(app_key, username, password)
    if not token:
        return []

    markets = get_tt_upcoming_markets(app_key, token, days_ahead)
    if not markets:
        return []

    market_ids = [m["marketId"] for m in markets]
    prices = get_best_prices(market_ids, app_key, token)

    results = []
    for m in markets:
        mid = m["marketId"]
        if mid not in prices:
            continue
        runners = m["runners"]
        p1_sel, p2_sel = runners[0]["selectionId"], runners[1]["selectionId"]
        o1 = prices[mid].get(p1_sel, 0.0)
        o2 = prices[mid].get(p2_sel, 0.0)
        if o1 > 1.0 and o2 > 1.0:
            results.append({
                "home": runners[0]["runnerName"],
                "away": runners[1]["runnerName"],
                "odds_home": o1,
                "odds_away": o2,
                "market_id": mid,
                "source": "betfair_exchange",
            })

    logger.info(f"Betfair Exchange : {len(results)} marchés avec cotes")
    return results
