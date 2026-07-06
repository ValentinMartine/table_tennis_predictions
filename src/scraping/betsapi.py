"""
Scraper BetsAPI - odds historiques et résultats en temps réel.
Doc: https://betsapi.com/docs/

Endpoints utilisés :
- GET /v1/bet365/result        → résultats matchs
- GET /v2/event/odds/summary   → odds pré-match
- GET /v1/events/inplay        → matchs en cours (live)
"""
import os
from datetime import datetime

from loguru import logger

from .base import BaseScraper, RawMatch

SPORT_ID = 18  # Table Tennis sur BetsAPI


class BetsAPIScraper(BaseScraper):
    SOURCE_NAME = "betsapi"

    BASE_URL = "https://api.betsapi.com/v1"
    BASE_URL_V2 = "https://api.betsapi.com/v2"

    def __init__(self, config: dict):
        super().__init__(config)
        self.token = os.getenv("BETSAPI_TOKEN", "")
        if not self.token:
            logger.warning("BETSAPI_TOKEN non défini - les appels API échoueront")

    def _api_get(self, endpoint: str, version: int = 1, **params) -> dict:
        base = self.BASE_URL if version == 1 else self.BASE_URL_V2
        url = f"{base}/{endpoint}"
        params["token"] = self.token
        resp = self._get(url, params=params)
        data = resp.json()
        if data.get("success") != 1:
            raise ValueError(f"BetsAPI error: {data.get('error', 'unknown')}")
        return data

    def scrape_competition(
        self, competition_id: str, start_date: datetime, end_date: datetime
    ) -> list[RawMatch]:
        """Récupère les résultats via l'endpoint ended events."""
        matches: list[RawMatch] = []
        page = 1

        while True:
            try:
                data = self._api_get(
                    "events/ended",
                    sport_id=SPORT_ID,
                    day=start_date.strftime("%Y%m%d"),
                    page=page,
                )
            except Exception as e:
                logger.error(f"BetsAPI erreur page {page} : {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for event in results:
                try:
                    match = self._parse_event(event, competition_id)
                    if match and start_date <= match.played_at <= end_date:
                        matches.append(match)
                except Exception as e:
                    logger.debug(f"BetsAPI parse error : {e}")

            pager = data.get("pager", {})
            if page >= pager.get("total_pages", 1):
                break
            page += 1

        return matches

    def get_live_events(self) -> list[dict]:
        """Matchs TT en cours - pour alertes live."""
        try:
            data = self._api_get("events/inplay", sport_id=SPORT_ID)
            return data.get("results", [])
        except Exception as e:
            logger.error(f"BetsAPI live error : {e}")
            return []

    def get_odds(self, event_id: str) -> dict[str, float]:
        """Récupère les odds pré-match pour un événement."""
        try:
            data = self._api_get("event/odds/summary", version=2, event_id=event_id)
            results = data.get("results", {})
            # Odds 1X2 → on prend le marché "winner"
            winner_market = results.get("1_1", {})  # Full time - moneyline
            odds = {}
            for outcome in winner_market.get("odds", []):
                if outcome.get("id") == "1":
                    odds["p1"] = float(outcome.get("odds", 0))
                elif outcome.get("id") == "2":
                    odds["p2"] = float(outcome.get("odds", 0))
            return odds
        except Exception as e:
            logger.debug(f"BetsAPI odds error pour event {event_id} : {e}")
            return {}

    def _parse_event(self, event: dict, competition_id: str) -> RawMatch | None:
        home = event.get("home", {})
        away = event.get("away", {})
        p1_name = home.get("name", "")
        p2_name = away.get("name", "")
        if not p1_name or not p2_name:
            return None

        # Score (sets)
        ss = event.get("ss", "")  # format "3:1"
        if not ss:
            return None
        try:
            parts = ss.split(":")
            score_p1, score_p2 = int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            return None

        time_raw = event.get("time")
        try:
            played_at = datetime.utcfromtimestamp(int(time_raw)) if time_raw else None
        except (ValueError, TypeError):
            played_at = None
        if not played_at:
            return None

        winner = 1 if score_p1 > score_p2 else 2

        return RawMatch(
            external_id=f"bapi_{event.get('id', '')}",
            competition_id=competition_id,
            player1_name=p1_name,
            player2_name=p2_name,
            player1_country=home.get("cc"),
            player2_country=away.get("cc"),
            played_at=played_at,
            winner=winner,
            score_p1=score_p1,
            score_p2=score_p2,
            sets_detail=event.get("scores"),
            odds_source="betsapi",
        )


def get_table_tennis_odds_betsapi(token: str, days_ahead: int = 7) -> list[dict]:
    """
    Point d'entrée pour l'enrichissement de cotes : matchs TT à venir + cotes bet365.

    Retourne une liste de {home, away, odds_home, odds_away, source:'betsapi'},
    au même format que les autres sources (Betfair, Sofascore, The Odds API).

    Source commerciale fiable (fonctionne en CI, sans géo-blocage ni throttling) ;
    nécessite un BETSAPI_TOKEN valide. Défensif : renvoie [] à la moindre erreur.
    """
    import time as _t
    import requests

    if not token:
        return []

    base_v1 = "https://api.betsapi.com/v1"
    base_v2 = "https://api.betsapi.com/v2"
    cutoff = _t.time() + days_ahead * 86400
    results: list[dict] = []
    try:
        page = 1
        events: list[dict] = []
        while page <= 10:
            r = requests.get(f"{base_v1}/events/upcoming",
                             params={"sport_id": SPORT_ID, "token": token, "page": page}, timeout=15)
            if not r.ok:
                break
            data = r.json()
            if data.get("success") != 1:
                break
            batch = data.get("results", [])
            events.extend(batch)
            pager = data.get("pager", {})
            if page >= pager.get("total_pages", 1) or not batch:
                break
            page += 1

        for ev in events:
            t_raw = ev.get("time")
            try:
                if t_raw and float(t_raw) > cutoff:
                    continue
            except (TypeError, ValueError):
                pass
            eid = ev.get("id")
            home = (ev.get("home") or {}).get("name", "")
            away = (ev.get("away") or {}).get("name", "")
            if not eid or not home or not away:
                continue
            ro = requests.get(f"{base_v2}/event/odds/summary",
                              params={"event_id": eid, "token": token}, timeout=15)
            if not ro.ok:
                continue
            market = (ro.json().get("results", {}) or {}).get("1_1", {})
            o1 = o2 = 0.0
            for outcome in market.get("odds", []):
                if outcome.get("id") == "1":
                    o1 = float(outcome.get("odds", 0) or 0)
                elif outcome.get("id") == "2":
                    o2 = float(outcome.get("odds", 0) or 0)
            if o1 > 1.0 and o2 > 1.0:
                results.append({"home": home, "away": away,
                                "odds_home": o1, "odds_away": o2, "source": "betsapi"})
    except requests.RequestException as e:
        logger.warning(f"BetsAPI odds à venir : {e}")
        return results

    logger.info(f"BetsAPI : {len(results)} matchs TT avec cotes")
    return results
