"""
Scraper Sofascore — API interne non-officielle mais stable et accessible.

Sofascore utilise la détection d'empreinte TLS (JA3/JA4) pour bloquer les
requêtes automatisées. On contourne ça avec curl-cffi qui impersonne Chrome.

Compétitions couvertes :
  - Setka Cup (UA) : ~738 matchs/jour, ID stable
  - Liga Pro (RU/BY) : ~380 matchs/jour, ID stable
  - Czech Liga Pro, TT Elite Series (PL), TT Cup (CZ)
  - WTT Champions/Contenders/Star Contenders : via date + filtre nom
  - ETTU Champions League : ID stable

API pattern :
  GET https://api.sofascore.com/api/v1/sport/table-tennis/scheduled-events/{date}
  GET https://api.sofascore.com/api/v1/unique-tournament/{id}/season/{sid}/events/last/{page}
"""
import time
from datetime import datetime, timedelta

from loguru import logger

from .base import BaseScraper, RawMatch

try:
    from curl_cffi import requests as cffi_requests
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False
    logger.warning(
        "curl-cffi non installé — Sofascore sera bloqué. "
        "Installez-le : pip install curl-cffi"
    )

API_BASE = "https://api.sofascore.com/api/v1"

# IDs Sofascore vérifiés (stables — une seule compétition, pas par édition)
TOURNAMENT_IDS: dict[str, dict] = {
    "setka_cup":        {"tournament_id": 15004, "name": "Setka Cup"},
    "liga_pro":         {"tournament_id": 15006, "name": "Liga Pro"},
    "liga_pro_belarus": {"tournament_id": 31462, "name": "Liga Pro Belarus"},
    "czech_liga_pro":   {"tournament_id": 19039, "name": "Czech Liga Pro"},
    "tt_cup_cz":        {"tournament_id": 15005, "name": "TT Cup"},
    "tt_elite_series":  {"tournament_id": 19041, "name": "TT Elite Series"},
    "ettu_champions":   {"tournament_id": 2122,  "name": "ETTU Champions League"},
    "ettu_champions_w": {"tournament_id": 9500,  "name": "ETTU Champions League, Women"},
}

# Patterns pour les compétitions sans ID stable (nouvelles editions chaque tournoi)
# clé = competition_id → pattern à chercher dans uniqueTournament.name
# Pour les compétitions à ID variable, on filtre par sous-chaîne dans le nom du tournoi.
# Un seul pattern par competition_id → prend le premier match trouvé.
NAME_PATTERNS: dict[str, list[str]] = {
    "wtt_champions":        ["WTT Champions", "WTT Singapore Smash", "WTT Macao"],
    "wtt_star_contenders":  ["WTT Star Contender"],
    "wtt_contenders":       ["WTT Contender"],
    "wtt_cup_finals":       ["WTT Cup Finals", "World Tour Cup Finals"],
    "wtt_feeder":           ["WTT Feeder"],
    "world_championships":  ["World Championships", "World Team Championships"],
    "european_championships": ["European Team Championships", "European Championships"],
}

_CFFI_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "Cache-Control": "no-cache",
}


class SofascoreScraper(BaseScraper):
    SOURCE_NAME = "sofascore"

    def __init__(self, config: dict):
        super().__init__(config)
        self._request_delay = max(config.get("request_delay_seconds", 1.0), 1.0)
        self._last_request_time: float = 0.0
        if _HAS_CURL_CFFI:
            self._cffi_session = cffi_requests.Session(impersonate="chrome120")
            self._cffi_session.headers.update(_CFFI_HEADERS)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _cffi_get(self, url: str) -> dict:
        """GET via curl-cffi avec throttling et retries."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

        logger.debug(f"[sofascore] GET {url}")
        for attempt in range(3):
            try:
                resp = self._cffi_session.get(
                    url,
                    timeout=self.config.get("timeout_seconds", 30),
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** (attempt + 1)
                logger.debug(f"[sofascore] retry {attempt+1}/3 après {wait}s — {e}")
                time.sleep(wait)
        return {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def scrape_competition(
        self, competition_id: str, start_date: datetime, end_date: datetime
    ) -> list[RawMatch]:
        if not _HAS_CURL_CFFI:
            logger.error("curl-cffi requis pour Sofascore. pip install curl-cffi")
            return []

        # Compétition avec ID Sofascore stable → scrape par tournoi
        meta = TOURNAMENT_IDS.get(competition_id)
        if meta:
            return self._scrape_by_tournament_id(competition_id, meta, start_date, end_date)

        # Compétition reconnue par nom → scrape par date + filtre
        patterns = NAME_PATTERNS.get(competition_id)
        if patterns:
            return self._scrape_by_date_filtered(competition_id, patterns, start_date, end_date)

        # Fallback : scrape tous les matchs TT de la plage
        logger.warning(
            f"[sofascore] {competition_id} inconnu — scrape par date sans filtre"
        )
        return self._scrape_by_date_filtered(competition_id, None, start_date, end_date)  # type: ignore

    # ------------------------------------------------------------------
    # Strategy A : tournament_id stable (e.g. Setka Cup, Liga Pro)
    # ------------------------------------------------------------------

    def _scrape_by_tournament_id(
        self,
        competition_id: str,
        meta: dict,
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawMatch]:
        t_id = meta["tournament_id"]
        matches: list[RawMatch] = []

        try:
            data = self._cffi_get(f"{API_BASE}/unique-tournament/{t_id}/seasons")
            seasons = data.get("seasons", [])
        except Exception as e:
            logger.error(f"Sofascore seasons error pour {competition_id}: {e}")
            return []

        for season in seasons:
            season_id = season.get("id")
            year = season.get("year", "")
            try:
                if year and int(str(year)[:4]) < start_date.year - 1:
                    continue
            except ValueError:
                pass

            page = 0
            while True:
                try:
                    url = (
                        f"{API_BASE}/unique-tournament/{t_id}"
                        f"/season/{season_id}/events/last/{page}"
                    )
                    data = self._cffi_get(url)
                except Exception as e:
                    logger.debug(
                        f"Sofascore {competition_id} season {season_id} p{page}: {e}"
                    )
                    break

                events = data.get("events", [])
                if not events:
                    break

                for event in events:
                    try:
                        match = self._parse_event(event, competition_id)
                        if match and start_date <= match.played_at <= end_date:
                            matches.append(match)
                    except Exception as e:
                        logger.debug(f"Parse error: {e}")

                oldest = min(
                    (e.get("startTimestamp", 0) for e in events), default=0
                )
                if oldest and datetime.utcfromtimestamp(oldest) < start_date:
                    break

                if not data.get("hasNextPage", False):
                    break
                page += 1

        logger.info(f"[sofascore] {competition_id}: {len(matches)} matchs")
        return matches

    # ------------------------------------------------------------------
    # Strategy B : date-based scrape + filtre par nom de tournoi
    # ------------------------------------------------------------------

    def _scrape_by_date_filtered(
        self,
        competition_id: str,
        name_patterns: list[str] | None,
        start_date: datetime,
        end_date: datetime,
    ) -> list[RawMatch]:
        matches: list[RawMatch] = []
        current = start_date

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            try:
                data = self._cffi_get(
                    f"{API_BASE}/sport/table-tennis/scheduled-events/{date_str}"
                )
                for event in data.get("events", []):
                    try:
                        if name_patterns:
                            ut = (
                                event.get("tournament", {})
                                .get("uniqueTournament", {})
                                .get("name", "")
                            ).lower()
                            if not any(p.lower() in ut for p in name_patterns):
                                continue
                        match = self._parse_event(event, competition_id)
                        if match:
                            matches.append(match)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Sofascore {date_str} error: {e}")

            current += timedelta(days=1)

        logger.info(f"[sofascore] {competition_id} (date-filter): {len(matches)} matchs")
        return matches

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def _parse_event(self, event: dict, competition_id: str) -> RawMatch | None:
        home = event.get("homeTeam") or event.get("home") or {}
        away = event.get("awayTeam") or event.get("away") or {}

        p1 = home.get("name") or home.get("shortName", "")
        p2 = away.get("name") or away.get("shortName", "")
        if not p1 or not p2:
            return None

        status = event.get("status", {})
        status_type = (
            status.get("type", "") if isinstance(status, dict) else str(status)
        )
        if status_type not in ("finished", "ended", "canceled_and_finished"):
            return None

        home_score = event.get("homeScore", {})
        away_score = event.get("awayScore", {})
        score_p1 = int(home_score.get("current", home_score.get("display", 0)) or 0)
        score_p2 = int(away_score.get("current", away_score.get("display", 0)) or 0)

        if score_p1 == score_p2:
            return None

        ts = event.get("startTimestamp")
        try:
            played_at = datetime.utcfromtimestamp(int(ts)) if ts else None
        except (ValueError, TypeError):
            played_at = None
        if not played_at:
            return None

        tournament = event.get("tournament", {})
        comp_id = competition_id or tournament.get("slug", "unknown")

        return RawMatch(
            external_id=f"sfs_{event.get('id', '')}",
            competition_id=comp_id,
            player1_name=str(p1),
            player2_name=str(p2),
            player1_country=home.get("country", {}).get("alpha2"),
            player2_country=away.get("country", {}).get("alpha2"),
            played_at=played_at,
            winner=1 if score_p1 > score_p2 else 2,
            score_p1=score_p1,
            score_p2=score_p2,
            sets_detail=None,
            round_name=event.get("roundInfo", {}).get("name"),
        )
