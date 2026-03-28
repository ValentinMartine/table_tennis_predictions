"""
Scraper OddsMatrix — source d'odds alternative à BetsAPI.

291 compétitions TT, 16 marchés, ~18k events/an.
1 mois d'essai gratuit : https://oddsmatrix.com/contact/

Documentation API : https://oddsmatrix.com/documentation/
(accès après inscription)

Endpoints utilisés :
  /events?sport=table_tennis&status=ended   → résultats historiques
  /odds?event_id={id}                        → odds pré-match
  /live                                      → événements en cours
"""
import os
from datetime import datetime

from loguru import logger

from .base import BaseScraper, RawMatch


class OddsMatrixScraper(BaseScraper):
    SOURCE_NAME = "oddsmatrix"

    def __init__(self, config: dict):
        super().__init__(config)
        self.token = os.getenv("ODDSMATRIX_TOKEN", "")
        self.base_url = config.get("base_url", "https://api.oddsmatrix.com/v1")
        if not self.token:
            logger.warning("ODDSMATRIX_TOKEN non défini — activer l'essai gratuit sur oddsmatrix.com/contact/")

    def _api_get(self, endpoint: str, **params) -> dict:
        url = f"{self.base_url}/{endpoint}"
        params["token"] = self.token
        resp = self._get(url, params=params)
        return resp.json()

    def scrape_competition(
        self, competition_id: str, start_date: datetime, end_date: datetime
    ) -> list[RawMatch]:
        matches: list[RawMatch] = []
        page = 1

        while True:
            try:
                data = self._api_get(
                    "events",
                    sport="table_tennis",
                    status="ended",
                    competition=competition_id,
                    from_date=start_date.strftime("%Y-%m-%d"),
                    to_date=end_date.strftime("%Y-%m-%d"),
                    page=page,
                    per_page=100,
                )
            except Exception as e:
                logger.error(f"OddsMatrix erreur page {page} pour {competition_id} : {e}")
                break

            events = data.get("events") or data.get("data") or []
            if not events:
                break

            for event in events:
                try:
                    match = self._parse_event(event, competition_id)
                    if match:
                        matches.append(match)
                except Exception as e:
                    logger.debug(f"OddsMatrix parse error : {e}")

            # Pagination
            meta = data.get("meta") or data.get("pagination") or {}
            if page >= meta.get("total_pages", 1):
                break
            page += 1

        return matches

    def get_live_odds(self) -> list[dict]:
        """Récupère les cotes en direct pour tous les matchs TT live."""
        try:
            data = self._api_get("live", sport="table_tennis")
            return data.get("events", [])
        except Exception as e:
            logger.error(f"OddsMatrix live error : {e}")
            return []

    def _parse_event(self, event: dict, competition_id: str) -> RawMatch | None:
        home = event.get("home") or event.get("participant1") or {}
        away = event.get("away") or event.get("participant2") or {}

        p1_name = home.get("name", "")
        p2_name = away.get("name", "")
        if not p1_name or not p2_name:
            return None

        result = event.get("result") or event.get("score") or {}
        score_p1 = int(result.get("home", result.get("p1", 0)))
        score_p2 = int(result.get("away", result.get("p2", 0)))

        dt_raw = event.get("start_time") or event.get("date") or ""
        played_at = self._parse_dt(dt_raw)
        if not played_at:
            return None

        # Odds pré-match si disponibles
        odds = event.get("odds") or {}
        odds_p1 = float(odds.get("1", odds.get("home", 0))) or None
        odds_p2 = float(odds.get("2", odds.get("away", 0))) or None

        winner = 1 if score_p1 > score_p2 else 2

        return RawMatch(
            external_id=f"om_{event.get('id', '')}",
            competition_id=competition_id,
            player1_name=p1_name,
            player2_name=p2_name,
            player1_country=home.get("country"),
            player2_country=away.get("country"),
            played_at=played_at,
            winner=winner,
            score_p1=score_p1,
            score_p2=score_p2,
            sets_detail=None,
            round_name=event.get("round"),
            stage=event.get("phase"),
            odds_p1=odds_p1,
            odds_p2=odds_p2,
            odds_source="oddsmatrix",
        )

    @staticmethod
    def _parse_dt(raw: str) -> datetime | None:
        formats = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
        for fmt in formats:
            try:
                return datetime.strptime(raw.strip(), fmt)
            except (ValueError, AttributeError):
                continue
        return None
