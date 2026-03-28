"""
Scraper ITTF World Rankings.
Source: https://www.ittf.com/ranking/ (HTML) ou
        https://github.com/romanzdk/ittf-data-scrape (CSV exports).
"""
import csv
import io
from datetime import datetime

import requests
from loguru import logger

from .base import BaseScraper


class IttfRankingsScraper(BaseScraper):
    SOURCE_NAME = "ittf_rankings"

    RANKINGS_URL = "https://www.ittf.com/ranking/"
    # URL du dernier export CSV du repo romanzdk (fallback)
    GITHUB_CSV_URL = (
        "https://raw.githubusercontent.com/romanzdk/ittf-data-scrape/"
        "main/data/rankings_men.csv"
    )

    def scrape_competition(self, competition_id, start_date, end_date):
        """Non applicable pour les rankings — utiliser scrape_rankings()."""
        raise NotImplementedError("Utiliser scrape_rankings() pour les classements ITTF.")

    def scrape_rankings(self, gender: str = "M") -> list[dict]:
        """
        Retourne le classement ITTF actuel.

        Returns:
            list de dicts : {rank, name, country, points, ittf_id}
        """
        try:
            return self._scrape_html(gender)
        except Exception as e:
            logger.warning(f"Scraping HTML ITTF échoué ({e}), fallback CSV GitHub")
            return self._scrape_github_csv(gender)

    def _scrape_html(self, gender: str) -> list[dict]:
        gender_param = "M" if gender == "M" else "W"
        url = f"{self.RANKINGS_URL}?gender={gender_param}"
        resp = self._get(url)

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")

        table = soup.select_one("table.ranking-table, table#ranking")
        if not table:
            raise ValueError("Table de classement introuvable dans le HTML")

        rankings = []
        for row in table.select("tbody tr"):
            cols = row.select("td")
            if len(cols) < 4:
                continue
            try:
                rankings.append({
                    "rank": int(cols[0].get_text(strip=True)),
                    "name": cols[1].get_text(strip=True),
                    "country": cols[2].get_text(strip=True),
                    "points": float(cols[3].get_text(strip=True).replace(",", "")),
                    "scraped_at": datetime.utcnow(),
                })
            except (ValueError, IndexError):
                continue

        logger.info(f"ITTF rankings HTML : {len(rankings)} joueurs ({gender})")
        return rankings

    def _scrape_github_csv(self, gender: str) -> list[dict]:
        url = self.GITHUB_CSV_URL if gender == "M" else self.GITHUB_CSV_URL.replace("men", "women")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        rankings = []
        reader = csv.DictReader(io.StringIO(resp.text))
        for row in reader:
            try:
                rankings.append({
                    "rank": int(row.get("rank", 0)),
                    "name": row.get("name", ""),
                    "country": row.get("country", ""),
                    "points": float(row.get("points", 0)),
                    "ittf_id": row.get("id"),
                    "scraped_at": datetime.utcnow(),
                })
            except (ValueError, KeyError):
                continue

        logger.info(f"ITTF rankings CSV GitHub : {len(rankings)} joueurs ({gender})")
        return rankings
