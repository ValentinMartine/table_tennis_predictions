"""
The Odds API — cotes bookmaker pour les matchs à venir.
Docs : https://the-odds-api.com/liveapi/guides/v4/

Free tier : 500 req/mois. Chaque appel /sports/{sport}/odds compte pour
N_bookmakers × N_markets requêtes (quota usage retourné dans les headers).

Sport key TT : 'table_tennis' (parfois absent sur le free tier selon couverture).
"""
import os
import unicodedata
from datetime import datetime, timezone

import requests
from loguru import logger

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_SPORT_KEY = "table_tennis"
_REGIONS = "eu"          # eu bookmakers (Pinnacle, Bet365, Unibet…)
_MARKETS = "h2h"         # head-to-head = moneyline winner


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return "".join(
        "-" if unicodedata.category(ch) == "Pd" else ch
        for ch in s
    ).lower().strip()


def _name_similarity(a: str, b: str) -> float:
    """Score simple de chevauchement de tokens entre deux noms de joueurs."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _match_event(p1: str, p2: str, api_home: str, api_away: str) -> bool:
    """Vérifie si un événement API correspond à un match (p1 vs p2)."""
    s1 = _name_similarity(p1, api_home) + _name_similarity(p2, api_away)
    s2 = _name_similarity(p1, api_away) + _name_similarity(p2, api_home)
    return max(s1, s2) >= 0.8


def get_table_tennis_odds(api_key: str) -> list[dict]:
    """
    Retourne tous les matchs TT à venir avec les cotes moyennes (best odds).

    Chaque élément :
        {
          "home": str, "away": str,
          "commence_time": datetime (UTC),
          "odds_home": float, "odds_away": float,
          "bookmaker": str,
          "requests_remaining": int,
        }
    """
    if not api_key:
        logger.warning("ODDS_API_KEY non défini — odds bookmaker désactivées")
        return []

    url = f"{ODDS_API_BASE}/sports/{_SPORT_KEY}/odds"
    try:
        resp = requests.get(url, params={
            "apiKey": api_key,
            "regions": _REGIONS,
            "markets": _MARKETS,
            "oddsFormat": "decimal",
        }, timeout=10)
    except requests.RequestException as e:
        logger.warning(f"The Odds API unreachable : {e}")
        return []

    remaining = int(resp.headers.get("x-requests-remaining", -1))

    if resp.status_code == 422:
        logger.warning(f"The Odds API : sport '{_SPORT_KEY}' non disponible (422)")
        return []
    if resp.status_code == 401:
        logger.warning("The Odds API : clé invalide (401)")
        return []
    if not resp.ok:
        logger.warning(f"The Odds API HTTP {resp.status_code} : {resp.text[:200]}")
        return []

    events = resp.json()
    if not isinstance(events, list):
        logger.warning(f"The Odds API format inattendu : {events}")
        return []

    results = []
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        ct_raw = ev.get("commence_time", "")
        try:
            ct = datetime.fromisoformat(ct_raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ct = None

        best_odds_home, best_odds_away, best_book = _best_h2h(ev.get("bookmakers", []), home, away)
        if best_odds_home and best_odds_away:
            results.append({
                "home": home,
                "away": away,
                "commence_time": ct,
                "odds_home": best_odds_home,
                "odds_away": best_odds_away,
                "bookmaker": best_book,
                "requests_remaining": remaining,
            })

    logger.info(f"The Odds API : {len(results)} matchs TT — {remaining} requêtes restantes")
    return results


def _best_h2h(bookmakers: list[dict], home: str, away: str) -> tuple[float, float, str]:
    """Retourne les meilleures cotes disponibles toutes books confondues."""
    best_h, best_a, best_book = 0.0, 0.0, ""
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: float(o["price"]) for o in market.get("outcomes", [])}
            oh = outcomes.get(home, 0.0)
            oa = outcomes.get(away, 0.0)
            if oh > best_h and oa > 1.0:
                best_h, best_a, best_book = oh, oa, bm.get("title", "")
    return best_h, best_a, best_book


def is_table_tennis_available(api_key: str) -> bool:
    """Vérifie si le tennis de table est disponible sur The Odds API."""
    if not api_key:
        return False
    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports",
            params={"apiKey": api_key},
            timeout=10,
        )
        if not resp.ok:
            return False
        return any(_SPORT_KEY in s.get("key", "") for s in resp.json())
    except requests.RequestException:
        return False


def enrich_with_bookmaker_odds(matches: list[dict], api_key: str) -> list[dict]:
    """
    Enrichit chaque match avec book_odds_p1 / book_odds_p2 / book_implied_p1
    depuis The Odds API. Les matchs sans correspondance gardent odds=0.

    Modifie la liste en place et la retourne.
    """
    api_events = get_table_tennis_odds(api_key)
    if not api_events:
        return matches

    matched = 0
    for m in matches:
        p1, p2 = m.get("p1_name", ""), m.get("p2_name", "")
        for ev in api_events:
            if _match_event(p1, p2, ev["home"], ev["away"]):
                # Sens de l'attribution : home→p1 ou home→p2 ?
                if _name_similarity(p1, ev["home"]) >= _name_similarity(p1, ev["away"]):
                    m["book_odds_p1"] = ev["odds_home"]
                    m["book_odds_p2"] = ev["odds_away"]
                else:
                    m["book_odds_p1"] = ev["odds_away"]
                    m["book_odds_p2"] = ev["odds_home"]
                m["bookmaker"] = ev["bookmaker"]

                o1, o2 = m["book_odds_p1"], m["book_odds_p2"]
                raw1, raw2 = 1 / o1, 1 / o2
                m["book_implied_p1"] = raw1 / (raw1 + raw2)  # no-vig
                matched += 1
                break

    logger.info(f"Odds bookmaker appariées : {matched}/{len(matches)} matchs")
    return matches
