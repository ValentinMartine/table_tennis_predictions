"""
Scraper pour tt.skillgamesboard.com (Setka Cup, LigaPro).

Utilise Playwright (navigateur headless) car le site bloque les requêtes HTTP
simples et nécessite l'exécution JavaScript pour afficher les données.

Prérequis :
    python -m playwright install chromium
"""
import re
from datetime import datetime

from loguru import logger
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from .base import BaseScraper, RawMatch

BASE_URL = "https://tt.skillgamesboard.com"

COMP_URLS = {
    "setka_cup": f"{BASE_URL}/tournament/setka-cup/results",
    "liga_pro":  f"{BASE_URL}/tournament/liga-pro/results",
}


class SkillGamesBoardScraper(BaseScraper):
    SOURCE_NAME = "skillgamesboard"

    def scrape_competition(
        self, competition_id: str, start_date: datetime, end_date: datetime
    ) -> list[RawMatch]:
        url = COMP_URLS.get(competition_id)
        if not url:
            logger.warning(f"URL inconnue pour {competition_id}")
            return []

        matches: list[RawMatch] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()

            page_num = 1
            stop = False

            while not stop:
                paginated_url = f"{url}?page={page_num}"
                logger.debug(f"[skillgamesboard] {paginated_url}")

                try:
                    page.goto(paginated_url, wait_until="networkidle", timeout=30_000)
                except PlaywrightTimeout:
                    logger.warning(f"Timeout page {page_num}, on arrête")
                    break

                # Attend que les lignes de matchs apparaissent
                try:
                    page.wait_for_selector(
                        "div.match, tr.match, div.game-row, table.results tbody tr",
                        timeout=10_000,
                    )
                except PlaywrightTimeout:
                    logger.info(f"Aucune ligne trouvée page {page_num} — fin de pagination")
                    break

                raw = self._extract_matches(page, competition_id)
                if not raw:
                    break

                for m in raw:
                    if m.played_at < start_date:
                        stop = True
                        break
                    if m.played_at <= end_date:
                        matches.append(m)

                page_num += 1

            browser.close()

        logger.info(f"[skillgamesboard] {competition_id} : {len(matches)} matchs extraits")
        return matches

    def _extract_matches(self, page, competition_id: str) -> list[RawMatch]:
        """
        Extrait les matchs depuis le DOM rendu.
        Utilise page.evaluate() pour lire directement le DOM côté JS.
        """
        raw_data = page.evaluate("""
            () => {
                const results = [];
                // Sélecteurs larges — couvre plusieurs mises en page possibles
                const rows = document.querySelectorAll(
                    'div.match-row, tr.match, div.game, div.result-row, ' +
                    'table.matches tbody tr, div.matches-list > div'
                );
                rows.forEach(row => {
                    // Noms des joueurs
                    const players = row.querySelectorAll(
                        '.player-name, .name, td.player, span.player, ' +
                        'div.player, a.player-link'
                    );
                    if (players.length < 2) return;

                    // Score
                    const scoreEl = row.querySelector(
                        '.score, td.score, span.score, div.score, .result'
                    );

                    // Date
                    const dateEl = row.querySelector(
                        'time, .date, td.date, span.date, .datetime, .time'
                    );

                    // Lien (pour l'ID externe)
                    const link = row.querySelector('a[href*="/match/"], a[href*="/game/"]');

                    results.push({
                        p1: players[0].innerText.trim(),
                        p2: players[1].innerText.trim(),
                        score: scoreEl ? scoreEl.innerText.trim() : '',
                        date: dateEl ? (dateEl.getAttribute('datetime') || dateEl.innerText.trim()) : '',
                        href: link ? link.getAttribute('href') : '',
                    });
                });
                return results;
            }
        """)

        matches = []
        for item in raw_data:
            try:
                match = self._parse_item(item, competition_id)
                if match:
                    matches.append(match)
            except Exception as e:
                logger.debug(f"Parse error : {e} — {item}")

        return matches

    def _parse_item(self, item: dict, competition_id: str) -> RawMatch | None:
        p1 = item.get("p1", "").strip()
        p2 = item.get("p2", "").strip()
        if not p1 or not p2:
            return None

        # Score — formats possibles : "3:1", "3-1", "3/1"
        score_raw = item.get("score", "")
        score_match = re.search(r"(\d+)\s*[:/-]\s*(\d+)", score_raw)
        if not score_match:
            return None
        score_p1, score_p2 = int(score_match.group(1)), int(score_match.group(2))
        if score_p1 == score_p2:
            return None  # match non terminé ou invalide

        # Date
        played_at = self._parse_date(item.get("date", ""))
        if not played_at:
            return None

        # ID externe depuis l'URL
        href = item.get("href", "")
        id_match = re.search(r"/(?:match|game)/(\d+)", href)
        external_id = f"sgb_{id_match.group(1)}" if id_match else f"sgb_{p1}_{p2}_{played_at.date()}"

        return RawMatch(
            external_id=external_id,
            competition_id=competition_id,
            player1_name=p1,
            player2_name=p2,
            player1_country=None,
            player2_country=None,
            played_at=played_at,
            winner=1 if score_p1 > score_p2 else 2,
            score_p1=score_p1,
            score_p2=score_p2,
            sets_detail=None,
        )

    @staticmethod
    def _parse_date(text: str) -> datetime | None:
        if not text:
            return None
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%d.%m.%Y %H:%M",
            "%d.%m.%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(text.strip()[:19], fmt)
            except ValueError:
                continue
        return None
