"""
Base scraper class with shared HTTP logic, retry, rate limiting.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class RawMatch:
    """Structure normalisée retournée par tous les scrapers."""
    external_id: str
    competition_id: str           # clé settings.yaml
    player1_name: str
    player2_name: str
    player1_country: str | None
    player2_country: str | None
    played_at: datetime
    winner: int                   # 1 ou 2
    score_p1: int
    score_p2: int
    sets_detail: str | None       # "11-8,9-11,11-7"
    round_name: str | None = None
    stage: str | None = None
    is_walkover: bool = False
    odds_p1: float | None = None
    odds_p2: float | None = None
    odds_source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseScraper(ABC):
    """Classe de base pour tous les scrapers."""

    SOURCE_NAME: str = "base"

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.get("user_agent", "Mozilla/5.0"),
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._request_delay = config.get("request_delay_seconds", 2.0)
        self._last_request_time: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _get(self, url: str, **kwargs) -> requests.Response:
        self._throttle()
        logger.debug(f"[{self.SOURCE_NAME}] GET {url}")
        response = self.session.get(
            url,
            timeout=self.config.get("timeout_seconds", 30),
            **kwargs,
        )
        response.raise_for_status()
        return response

    @abstractmethod
    def scrape_competition(
        self, competition_id: str, start_date: datetime, end_date: datetime
    ) -> list[RawMatch]:
        """Scrape les matchs d'une compétition entre deux dates."""
        ...

    def scrape_all_competitions(
        self, competition_ids: list[str], start_date: datetime, end_date: datetime
    ) -> list[RawMatch]:
        all_matches: list[RawMatch] = []
        for comp_id in competition_ids:
            try:
                logger.info(f"[{self.SOURCE_NAME}] Scraping {comp_id}...")
                matches = self.scrape_competition(comp_id, start_date, end_date)
                logger.info(f"[{self.SOURCE_NAME}] {len(matches)} matchs récupérés pour {comp_id}")
                all_matches.extend(matches)
            except Exception as e:
                logger.error(f"[{self.SOURCE_NAME}] Erreur sur {comp_id}: {e}")
        return all_matches
