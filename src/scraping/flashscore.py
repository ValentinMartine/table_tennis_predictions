"""
Scraper Flashscore pour le tennis de table.

Stratégie : Playwright intercepte les appels réseau internes de Flashscore
pour récupérer le JSON brut directement — plus fiable que parser le DOM.

Compétitions couvertes via Flashscore :
    TTNet (TR), Bundesliga (DE), Superliga (ES), Pro A (FR), Ekstraliga (PL)

Prérequis :
    python -m playwright install chromium
"""
import json
import re
import time
from datetime import datetime
from typing import Any

from loguru import logger
from playwright.sync_api import Route, sync_playwright, TimeoutError as PlaywrightTimeout

from .base import BaseScraper, RawMatch

BASE_URL = "https://www.flashscore.com"

# Slugs Flashscore par competition_id
COMP_SLUGS = {
    "ttnet":         "table-tennis/turkey/ttnet-league",
    "bundesliga":    "table-tennis/germany/bundesliga",
    "superliga_es":  "table-tennis/spain/superliga",
    "pro_a_fr":      "table-tennis/france/pro-a",
    "extraleague_pl":"table-tennis/poland/ekstraliga",
}


class FlashscoreScraper(BaseScraper):
    SOURCE_NAME = "flashscore"

    def scrape_competition(
        self, competition_id: str, start_date: datetime, end_date: datetime
    ) -> list[RawMatch]:
        slug = COMP_SLUGS.get(competition_id)
        if not slug:
            logger.warning(f"Slug Flashscore inconnu pour {competition_id}")
            return []

        url = f"{BASE_URL}/{slug}/results/"
        matches: list[RawMatch] = []
        intercepted: list[dict] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()

            # Intercepte les réponses JSON internes de Flashscore
            def handle_response(response):
                url_r = response.url
                if (
                    ("flashscore" in url_r or "livesport" in url_r)
                    and response.status == 200
                ):
                    ct = response.headers.get("content-type", "")
                    if "json" in ct or "javascript" in ct:
                        try:
                            body = response.body()
                            # Flashscore encode souvent ses données en format propriétaire
                            # On cherche des patterns JSON dans la réponse
                            text = body.decode("utf-8", errors="replace")
                            if '"id"' in text and ('"home"' in text or '"away"' in text):
                                intercepted.append({"url": url_r, "body": text})
                        except Exception:
                            pass

            page.on("response", handle_response)

            try:
                page.goto(url, wait_until="networkidle", timeout=45_000)
                # Scroll pour charger plus de résultats
                for _ in range(5):
                    page.keyboard.press("End")
                    page.wait_for_timeout(1500)

                    show_more = page.query_selector(
                        "a.event__more, button.loadMore, div.show-more"
                    )
                    if show_more:
                        show_more.click()
                        page.wait_for_timeout(2000)

            except PlaywrightTimeout:
                logger.warning(f"Timeout sur {url}")

            # Tente d'abord de parser les données interceptées
            if intercepted:
                for item in intercepted:
                    parsed = self._parse_intercepted(item["body"], competition_id)
                    matches.extend(parsed)

            # Fallback : parse le DOM directement si aucune donnée interceptée
            if not matches:
                logger.info(f"[flashscore] Fallback DOM pour {competition_id}")
                matches = self._parse_dom(page, competition_id)

            browser.close()

        # Filtre par dates
        matches = [
            m for m in matches
            if start_date <= m.played_at <= end_date
        ]

        logger.info(f"[flashscore] {competition_id} : {len(matches)} matchs")
        return matches

    def _parse_intercepted(self, body: str, competition_id: str) -> list[RawMatch]:
        """Tente de parser les données JSON interceptées."""
        matches = []
        try:
            # Flashscore utilise parfois un format non-standard — cherche les objets JSON
            json_objects = re.findall(r'\{[^{}]{50,}\}', body)
            for obj_str in json_objects:
                try:
                    obj = json.loads(obj_str)
                    match = self._parse_event_json(obj, competition_id)
                    if match:
                        matches.append(match)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"Parse intercepted error : {e}")
        return matches

    def _parse_dom(self, page, competition_id: str) -> list[RawMatch]:
        """Parse le DOM rendu de la page results."""
        raw_data = page.evaluate("""
            () => {
                const results = [];
                // Flashscore structure ses events dans des divs data-*
                const events = document.querySelectorAll(
                    'div.event__match, div[id^="g_1_"], div.sportName__sport > div'
                );
                events.forEach(ev => {
                    const homeEl = ev.querySelector(
                        '.event__participant--home, .participant-name.participant-name--home'
                    );
                    const awayEl = ev.querySelector(
                        '.event__participant--away, .participant-name.participant-name--away'
                    );
                    const scoreHome = ev.querySelector(
                        '.event__score--home, .detailScore__wrapper span:first-child'
                    );
                    const scoreAway = ev.querySelector(
                        '.event__score--away, .detailScore__wrapper span:last-child'
                    );
                    const timeEl = ev.querySelector(
                        '.event__time, div.event__stage--block'
                    );

                    if (!homeEl || !awayEl) return;

                    results.push({
                        home: homeEl.innerText.trim(),
                        away: awayEl.innerText.trim(),
                        score_home: scoreHome ? scoreHome.innerText.trim() : '',
                        score_away: scoreAway ? scoreAway.innerText.trim() : '',
                        time: timeEl ? timeEl.innerText.trim() : '',
                        id: ev.getAttribute('id') || '',
                    });
                });
                return results;
            }
        """)

        matches = []
        for item in raw_data:
            try:
                match = self._parse_dom_item(item, competition_id)
                if match:
                    matches.append(match)
            except Exception as e:
                logger.debug(f"DOM parse error : {e}")
        return matches

    def _parse_dom_item(self, item: dict, competition_id: str) -> RawMatch | None:
        p1 = item.get("home", "").strip()
        p2 = item.get("away", "").strip()
        if not p1 or not p2:
            return None

        try:
            score_p1 = int(item.get("score_home", "0").split()[0])
            score_p2 = int(item.get("score_away", "0").split()[0])
        except (ValueError, IndexError):
            return None

        if score_p1 == score_p2 == 0:
            return None

        played_at = self._parse_fs_date(item.get("time", ""))
        if not played_at:
            played_at = datetime.utcnow().replace(hour=0, minute=0, second=0)

        event_id = item.get("id", "").replace("g_1_", "")
        external_id = f"fs_{event_id}" if event_id else f"fs_{p1}_{p2}"

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

    def _parse_event_json(self, obj: dict, competition_id: str) -> RawMatch | None:
        home = obj.get("home") or obj.get("homeTeam") or {}
        away = obj.get("away") or obj.get("awayTeam") or {}
        if isinstance(home, str):
            p1, p2 = home, away
        else:
            p1 = home.get("name", home.get("shortName", ""))
            p2 = away.get("name", away.get("shortName", ""))
        if not p1 or not p2:
            return None

        score = obj.get("score") or obj.get("result") or {}
        try:
            score_p1 = int(score.get("home", score.get("current", {}).get("home", 0)))
            score_p2 = int(score.get("away", score.get("current", {}).get("away", 0)))
        except (ValueError, TypeError):
            return None

        ts = obj.get("startTimestamp") or obj.get("time")
        try:
            played_at = datetime.utcfromtimestamp(int(ts)) if ts else None
        except (ValueError, TypeError):
            played_at = None
        if not played_at:
            return None

        return RawMatch(
            external_id=f"fs_{obj.get('id', '')}",
            competition_id=competition_id,
            player1_name=str(p1),
            player2_name=str(p2),
            player1_country=None,
            player2_country=None,
            played_at=played_at,
            winner=1 if score_p1 > score_p2 else 2,
            score_p1=score_p1,
            score_p2=score_p2,
            sets_detail=None,
        )

    @staticmethod
    def _parse_fs_date(text: str) -> datetime | None:
        """Parse les formats de date Flashscore : '12.03. 18:30' ou '12.03.2024'."""
        text = text.strip()
        # Format courant Flashscore : "12.03. 18:30"
        m = re.match(r"(\d{2})\.(\d{2})\.\s+(\d{2}):(\d{2})", text)
        if m:
            day, month, hour, minute = m.groups()
            year = datetime.utcnow().year
            try:
                return datetime(year, int(month), int(day), int(hour), int(minute))
            except ValueError:
                pass
        # Format complet : "12.03.2024"
        m2 = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", text)
        if m2:
            day, month, year = m2.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except ValueError:
                pass
        return None
