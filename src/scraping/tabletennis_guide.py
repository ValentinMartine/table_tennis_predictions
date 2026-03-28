"""
Scraper pour tabletennis.guide (WTT, ITTF international, championnats).

Le site expose une API JSON non documentée accessible via les endpoints
/api/matches et /api/events — à confirmer lors de l'exploration initiale.
"""
import re
from datetime import datetime

from loguru import logger

from .base import BaseScraper, RawMatch

# Mapping competition_id → slug/ID sur tabletennis.guide
COMP_SLUGS = {
    "wtt_champions": "wtt-champions",
    "wtt_star_contenders": "wtt-star-contenders",
    "wtt_contenders": "wtt-contenders",
    "wtt_cup_finals": "wtt-cup-finals",
    "world_championships": "world-championships",
    "european_championships": "european-championships",
    "asian_championships": "asian-championships",
    "pan_american": "pan-american-championships",
    "ittf_pro_league": "ittf-pro-league",
}


class TableTennisGuideScraper(BaseScraper):
    SOURCE_NAME = "tabletennis_guide"

    BASE_URL = "https://www.tabletennis.guide"
    API_BASE = f"{BASE_URL}/api"

    def scrape_competition(
        self, competition_id: str, start_date: datetime, end_date: datetime
    ) -> list[RawMatch]:
        slug = COMP_SLUGS.get(competition_id)
        if not slug:
            logger.warning(f"Slug inconnu : {competition_id}")
            return []

        matches: list[RawMatch] = []
        page = 1

        while True:
            url = (
                f"{self.API_BASE}/matches"
                f"?event={slug}"
                f"&from={start_date.strftime('%Y-%m-%d')}"
                f"&to={end_date.strftime('%Y-%m-%d')}"
                f"&page={page}"
            )
            try:
                resp = self._get(url)
                data = resp.json()
            except Exception as e:
                logger.error(f"Erreur page {page} pour {competition_id} : {e}")
                break

            items = data.get("matches") or data.get("results") or []
            if not items:
                break

            for item in items:
                try:
                    match = self._parse_match(item, competition_id)
                    if match:
                        matches.append(match)
                except Exception as e:
                    logger.debug(f"Erreur parsing match : {e}")

            # Pagination
            if not data.get("next_page"):
                break
            page += 1

        return matches

    def _parse_match(self, item: dict, competition_id: str) -> RawMatch | None:
        p1 = item.get("player1") or item.get("home") or {}
        p2 = item.get("player2") or item.get("away") or {}

        p1_name = p1.get("name") or p1.get("full_name", "")
        p2_name = p2.get("name") or p2.get("full_name", "")
        if not p1_name or not p2_name:
            return None

        score_p1 = int(item.get("score1") or item.get("sets_p1", 0))
        score_p2 = int(item.get("score2") or item.get("sets_p2", 0))

        played_at_raw = item.get("date") or item.get("start_time") or item.get("played_at")
        played_at = self._parse_datetime(played_at_raw)
        if not played_at:
            return None

        winner = 1 if score_p1 > score_p2 else 2

        # Détail des sets : "11-8,9-11,11-7,11-4"
        sets_detail = None
        if "sets" in item and isinstance(item["sets"], list):
            parts = [f"{s.get('p1', 0)}-{s.get('p2', 0)}" for s in item["sets"]]
            sets_detail = ",".join(parts)

        return RawMatch(
            external_id=f"ttg_{item.get('id', '')}",
            competition_id=competition_id,
            player1_name=p1_name,
            player2_name=p2_name,
            player1_country=p1.get("country"),
            player2_country=p2.get("country"),
            played_at=played_at,
            winner=winner,
            score_p1=score_p1,
            score_p2=score_p2,
            sets_detail=sets_detail,
            round_name=item.get("round") or item.get("stage_name"),
            stage=item.get("stage") or item.get("phase"),
            is_walkover=bool(item.get("walkover") or item.get("retired")),
        )

    @staticmethod
    def _parse_datetime(raw: str | None) -> datetime | None:
        if not raw:
            return None
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue
        return None
