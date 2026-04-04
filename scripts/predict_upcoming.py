"""
Prédit les chances de victoire pour les matchs WTT/internationaux à venir.

Récupère les prochains matchs depuis Sofascore (7 jours),
reconstruit les features depuis la DB, et affiche les prédictions.

Usage :
    python scripts/predict_upcoming.py
    python scripts/predict_upcoming.py --days 3 --model lgbm
    python scripts/predict_upcoming.py --min-conf 0.60
"""
import argparse
import io
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.database.db import engine, get_session
from src.database.models import BettingRecord
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.features.match_features import build_single_match_features
from src.features.tournament_projections import TournamentSimulator

try:
    from curl_cffi import requests as cffi_requests
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

API_BASE = "https://api.sofascore.com/api/v1"
_CFFI_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}

# Compétitions WTT/internationales ciblées (sous-chaînes dans le nom Sofascore)
TARGET_PATTERNS_INTL = [
    "WTT Champions", "WTT Star Contender", "WTT Contender", 
    "WTT Grand Smash", "WTT Finals", "WTT Cup Finals",
    "World Cup", "World Championships", "European Championships",
    "Asian Championships", "Olympic Games", "Team World Cup",
]
TARGET_PATTERNS_ALL = TARGET_PATTERNS_INTL + [
    "Liga Pro", "Setka Cup", "ETTU Champions",
    "Czech Liga Pro", "TT Cup", "TT Elite Series",
]


# ── WTT API (source principale -sans protection bot) ────────────────────────

import re
import requests as _requests

_WTT_CDN   = "https://wtt-web-frontdoor-withoutcache-cqakg0andqf5hchn.a01.azurefd.net"
_WTT_API   = "https://wtt-website-api-prod-3-frontdoor-bddnb2haduafdze9.a01.azurefd.net"
_WTT_LIVE  = "https://liveeventsapi.worldtabletennis.com"
_WTT_HDR   = {"Accept": "application/json", "Referer": "https://worldtabletennis.com/"}

# Event types WTT/ITTF à suivre
_WTT_EVENT_TYPES = {
    "WTT Champions", "WTT Star Contender", "WTT Contender",
    "WTT Grand Smash", "WTT Cup Finals", "World Cup", "WTTC",
    "Grand Smash", "WTT Finals",
}

# Stage order pour déterminer le prochain tour
_STAGE_ORDER = ["GRPX", "R16N", "QFNL", "SFNL", "FNLX", "BRNZ"]


def _wtt_get(url: str) -> dict | list:
    try:
        r = _requests.get(url, headers=_WTT_HDR, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug(f"WTT GET {url}: {e}")
    return {}


def _get_active_event_ids(days: int = 30) -> list[dict]:
    """Retourne les events WTT/ITTF actifs ou à venir dans les `days` prochains jours."""
    import json as _json
    today = date.today()
    cutoff = today + timedelta(days=days)

    try:
        body = {"custom_filter": _json.dumps([
            {"name": "StartDateTime", "value": today.year,
             "custom_handling": "multimatch_year_or_filter", "condition": "or_start"},
            {"name": "FromStartDate", "value": today.year,
             "custom_handling": "multimatch_year_or_filter", "condition": "or_end"},
        ])}
        r = _requests.post(f"{_WTT_API}/api/eventcalendar", json=body, headers=_WTT_HDR, timeout=12)
        calendar = r.json()
    except Exception as e:
        logger.warning(f"WTT eventcalendar error: {e}")
        return []

    events = []
    for group in calendar:
        for ev in group.get("rows", [ev] if not isinstance(group, dict) else []):
            etype = ev.get("EventType", "")
            start_str = ev.get("StartDateTime", "")
            end_str   = ev.get("EndDateTime", "") or start_str
            if not start_str:
                continue
            try:
                start = date.fromisoformat(start_str[:10])
                end   = date.fromisoformat(end_str[:10])
            except ValueError:
                continue
            # Garder les événements en cours ou à venir
            if end >= today and start <= cutoff:
                if any(t.lower() in etype.lower() or t.lower() in ev.get("EventName","").lower()
                       for t in _WTT_EVENT_TYPES):
                    events.append({
                        "event_id":   ev.get("EventId"),
                        "event_name": ev.get("EventName", ""),
                        "start":      start,
                        "end":        end,
                        "type":       etype,
                    })
    return events


def _extract_winner(match: dict) -> str | None:
    """Retourne le nom du gagnant depuis un match officiel WTT."""
    mc = match.get("match_card", {})
    if not mc:
        return None
    comps = mc.get("competitiors", [])
    if len(comps) < 2:
        return None
    try:
        h_sets, a_sets = map(int, mc.get("overallScores", "0-0").split("-"))
    except (ValueError, AttributeError):
        return None
    if h_sets > a_sets:
        return comps[0].get("competitiorName")
    elif a_sets > h_sets:
        return comps[1].get("competitiorName")
    return None


def _stage_from_code(doc_code: str) -> str:
    """Extrait le stage (GRPX, R16N, QFNL, SFNL, FNLX) depuis le documentCode."""
    for stage in _STAGE_ORDER:
        if stage in doc_code:
            return stage
    return "UNKN"


def _get_event_schedule_timestamps(event_id) -> dict:
    """Extrait les timestamps de début de match depuis GetOfficialResult (champ startDateLocal).
    Returns dict mapping documentCode → Unix timestamp (best-effort).
    """
    from datetime import datetime as _dt
    result = {}
    data = _wtt_get(
        f"{_WTT_LIVE}/api/cms/GetOfficialResult?EventId={event_id}&include_match_card=true&take=500"
    )
    items = data if isinstance(data, list) else []
    for item in items:
        doc = item.get("documentCode", "")
        if not doc:
            continue
        val = item.get("startDateLocal") or (item.get("match_card") or {}).get("matchDateTime", {})
        if isinstance(val, dict):
            val = val.get("startDateUTC") or val.get("startDateLocal")
        if not val:
            continue
        try:
            dt = _dt.fromisoformat(str(val).replace("Z", "+00:00"))
            result[doc] = int(dt.timestamp())
        except (ValueError, TypeError):
            pass
    if result:
        logger.debug(f"WTT schedule: {len(result)} timestamps depuis GetOfficialResult")
    return result


def _event_end_timestamp(ev_end: date) -> int:
    """Convert event end date to a noon Unix timestamp (rough match proxy)."""
    from datetime import datetime as _dt
    return int(_dt.combine(ev_end, _dt.min.time().replace(hour=12)).timestamp())


def fetch_upcoming_matches_wtt(days: int = 14, gender: str = "M") -> list[dict]:
    """
    Source secondaire : matchs WTT/ITTF depuis l'API officielle worldtabletennis.com.
    Retourne les matchs à venir ou en cours (prochains tours inférables depuis les résultats).
    gender: "M" = Men's Singles only, "F" = Women's Singles only, "all" = both
    """
    gender_keywords = {
        "M":   ("men",),
        "F":   ("women",),
        "all": ("men", "women"),
    }.get(gender.upper() if gender != "all" else "all", ("men",))

    def _is_target_gender(sub_event_type: str) -> bool:
        if not sub_event_type:
            return True  # unknown subtype: keep it to avoid over-filtering
        stl = sub_event_type.lower()
        if "double" in stl or "mixed" in stl:
            return False
        return any(kw in stl for kw in gender_keywords)

    def _gender_from_sub(sub_event_type: str) -> str:
        stl = sub_event_type.lower()
        if "women" in stl:
            return "F"
        if "men" in stl:
            return "M"
        return "M"  # défaut

    events = _get_active_event_ids(days)
    if not events:
        logger.warning("WTT API: aucun event actif trouvé")
        return []

    results = []
    for ev in events:
        eid = ev["event_id"]
        ename = ev["event_name"]
        ev_end = ev["end"]
        logger.info(f"WTT: analyse event {eid} -{ename}")

        schedule_ts = _get_event_schedule_timestamps(eid)

        official_raw = _wtt_get(
            f"{_WTT_LIVE}/api/cms/GetOfficialResult?EventId={eid}&include_match_card=true&take=200"
        )
        live_raw = _wtt_get(f"{_WTT_LIVE}/api/cms/GetLiveResult?EventId={eid}")

        official = official_raw if isinstance(official_raw, list) else []
        live     = live_raw     if isinstance(live_raw, list)     else []

        # --- Indexer les official results par documentCode normalisé ---
        # Normalise: strip trailing dashes (live vs official ont des longueurs différentes)
        def _norm_doc(d: str) -> str:
            return d.rstrip("-").rstrip()

        doc_info: dict[str, dict] = {}
        for m in official:
            mc  = m.get("match_card") or {}
            doc = _norm_doc(m.get("documentCode", ""))
            sub = m.get("subEventType", "")
            if not doc:
                continue
            comps = mc.get("competitiors", [])
            p1 = comps[0]["competitiorName"] if len(comps) > 0 else ""
            p2 = comps[1]["competitiorName"] if len(comps) > 1 else ""
            ts  = schedule_ts.get(m.get("documentCode", ""), 0)
            doc_info[doc] = {"sub": sub, "p1": p1, "p2": p2, "ts": ts,
                             "round": mc.get("subEventDescription", "")}

        # --- Matchs en cours ---
        for m in live:
            mc  = m.get("match_card") or {}
            doc = m.get("documentCode", "")
            sub = m.get("subEventType", "") or mc.get("subEventDescription", "")
            if not _is_target_gender(sub):
                continue
            stage    = _stage_from_code(doc)
            norm_doc = _norm_doc(doc)
            # Cross-reference official results pour les noms + timestamp
            info     = doc_info.get(norm_doc, {})
            p1 = info.get("p1") or (mc.get("competitiors") or [{}])[0].get("competitiorName", "")
            p2 = info.get("p2") or (mc.get("competitiors") or [{}, {}])[1].get("competitiorName", "") \
                if len(mc.get("competitiors") or []) > 1 else ""
            live_ts  = info.get("ts") or -1
            rname    = info.get("round") or sub or stage
            if not p1 or not p2:
                logger.debug(f"  Live match sans joueurs: {doc}")
                continue
            results.append({
                "event_id":   f"wtt_{eid}_{doc}",
                "tournament": ename,
                "p1_name":    p1,
                "p2_name":    p2,
                "start_time": live_ts,
                "status":     "inprogress",
                "round_name": rname,
                "group_name": "",
                "gender":     _gender_from_sub(sub),
            })

        # --- Inférer le prochain tour depuis les résultats officiels ---
        from collections import defaultdict
        stage_results: dict[tuple, dict[int, str]] = defaultdict(dict)
        seen_sub_types: set[str] = set()
        for m in official:
            mc  = m.get("match_card", {})
            doc = m.get("documentCode", "")
            sub = m.get("subEventType", "")
            seen_sub_types.add(sub)
            if not mc or not doc:
                continue
            if not _is_target_gender(sub):
                continue
            stage = _stage_from_code(doc)
            mn = re.search(r"(\d{4})", doc.replace("000000", ""))
            if not mn:
                continue
            match_num = int(mn.group(1))
            winner = _extract_winner(m)
            if winner:
                stage_results[(sub, stage)][match_num] = winner

        logger.debug(f"  subEventTypes trouvés: {seen_sub_types}")

        # Pour chaque sous-event, trouver le stage courant et construire les demi ou finales
        sub_stages: dict[str, str] = {}
        for (sub, stage), wins in stage_results.items():
            if stage not in _STAGE_ORDER:
                continue
            if sub not in sub_stages or _STAGE_ORDER.index(stage) > _STAGE_ORDER.index(sub_stages[sub]):
                sub_stages[sub] = stage

        for sub, current_stage in sub_stages.items():
            if current_stage not in _STAGE_ORDER[:-1]:
                continue
            next_stage = _STAGE_ORDER[_STAGE_ORDER.index(current_stage) + 1]
            wins = stage_results.get((sub, current_stage), {})
            # Bracket standard : (1 vs 2) → SF1, (3 vs 4) → SF2
            sorted_wins = sorted(wins.items())
            pairs = [(sorted_wins[i], sorted_wins[i+1])
                     for i in range(0, len(sorted_wins)-1, 2)
                     if i+1 < len(sorted_wins)]
            for (n1, p1), (n2, p2) in pairs:
                already_live = any(
                    r["p1_name"] == p1 and r["p2_name"] == p2
                    for r in results
                )
                if not already_live:
                    inferred_doc = f"{sub}_{next_stage}"
                    ts = schedule_ts.get(inferred_doc, 0) or _event_end_timestamp(ev_end)
                    results.append({
                        "event_id":   f"wtt_{eid}_{sub}_{next_stage}",
                        "tournament": ename,
                        "p1_name":    p1,
                        "p2_name":    p2,
                        "start_time": ts,
                        "status":     "notstarted",
                        "round_name": f"{sub.replace('Singles','').strip()} -{next_stage}",
                        "group_name": "",
                        "gender":     _gender_from_sub(sub),
                    })

    logger.info(f"WTT API: {len(results)} matchs à venir ou en cours")
    return results


# ── Sofascore ────────────────────────────────────────────────────────────────

def _sfetch(session, url: str) -> dict:
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            logger.warning(f"Sofascore {r.status_code} -{url} -{r.text[:200]}")
            return {}
        data = r.json()
        if not data.get("events"):
            logger.debug(f"Sofascore 200 mais events vide -{url}")
        return data
    except Exception as e:
        logger.warning(f"Sofascore exception -{url} -{e}")
        return {}


import time as _time_module

# Cache pour éviter de re-tester Sofascore quand il est bloqué (TTL = 30min)
_sofascore_blocked_until: float = 0.0
_SOFASCORE_BLOCK_TTL = 1800  # secondes


def _try_sofascore_session():
    """Tente de créer une session Sofascore. Retourne la session ou None si bloqué."""
    global _sofascore_blocked_until
    if _time_module.time() < _sofascore_blocked_until:
        logger.debug("Sofascore bloqué (cache TTL) -skip")
        return None
    if not _HAS_CURL_CFFI:
        return None
    for target in ("safari17_0", "firefox133", "chrome131", "safari15_5"):
        try:
            s = cffi_requests.Session(impersonate=target)
            s.headers.update(_CFFI_HEADERS)
            test = s.get(f"{API_BASE}/sport/table-tennis/scheduled-events/{date.today()}", timeout=8)
            if test.status_code == 200:
                logger.info(f"Sofascore accessible avec impersonate={target}")
                return s
            logger.debug(f"impersonate={target} → {test.status_code}")
        except Exception:
            continue
    _sofascore_blocked_until = _time_module.time() + _SOFASCORE_BLOCK_TTL
    logger.warning(f"Sofascore bloqué -fallback WTT API (retry dans {_SOFASCORE_BLOCK_TTL//60}min)")
    return None


def fetch_upcoming_matches(session=None, days: int = 7, all_leagues: bool = False) -> tuple[list[dict], str]:
    """Retourne (matchs, source) où source est 'sofascore' ou 'wtt_api'."""
    if not _HAS_CURL_CFFI:
        logger.error("curl-cffi requis. pip install curl-cffi")
        return [], "wtt_api"

    patterns = TARGET_PATTERNS_ALL if all_leagues else TARGET_PATTERNS_INTL

    if session is None:
        session = _try_sofascore_session()
        if session is None:
            return fetch_upcoming_matches_wtt(days, gender="all"), "wtt_api"

    results = []
    seen_ids = set()
    all_tournament_names: set[str] = set()
    all_status_types: set[str] = set()
    total_events = 0
    for d in range(days + 1):
        day = date.today() + timedelta(days=d)
        data = _sfetch(session, f"{API_BASE}/sport/table-tennis/scheduled-events/{day}")
        for ev in data.get("events", []):
            eid = ev.get("id")
            if not eid or eid in seen_ids:
                continue
            total_events += 1
            status = ev.get("status", {}).get("type", "")
            all_status_types.add(status)
            if status not in ("notstarted", "inprogress"):
                continue
            t = ev.get("tournament", {})
            sub_name = t.get("name", "")
            parent_name = t.get("uniqueTournament", {}).get("name", "")
            # Build display name: "Parent, Sub" when both exist and different
            if parent_name and sub_name and sub_name not in parent_name:
                tournament_name = f"{parent_name}, {sub_name}"
            else:
                tournament_name = parent_name or sub_name
            # Match against both names (catches knockout rounds where sub_name = "Round of 16")
            match_target = f"{parent_name} {sub_name}".lower()
            all_tournament_names.add(tournament_name)
            if not any(p.lower() in match_target for p in patterns):
                continue
            
            round_name = ev.get("roundInfo", {}).get("name", "")
            group_name = ev.get("roundInfo", {}).get("caption", "")
            
            # Fallback for World Cup groups in tournament name
            if not group_name and "Group" in tournament_name:
                import re
                m_group = re.search(r"Group (\d+)", tournament_name)
                if m_group:
                    group_name = f"Group {m_group.group(1)}"
            if not round_name and group_name:
                round_name = group_name

            seen_ids.add(eid)
            results.append({
                "event_id": eid,
                "tournament": tournament_name,
                "p1_name": ev.get("homeTeam", {}).get("name", ""),
                "p2_name": ev.get("awayTeam", {}).get("name", ""),
                "start_time": ev.get("startTimestamp", 0),
                "status": status,
                "round_name": round_name,
                "group_name": group_name,
                "gender": "M",
            })
        import time; time.sleep(0.5)

    logger.info(f"Total événements bruts récupérés : {total_events}")
    logger.info(f"Status types rencontrés : {all_status_types}")
    logger.info(f"Tous les tournois détectés sur Sofascore ({days} jours) :")
    for name in sorted(all_tournament_names):
        matched = any(p.lower() in name.lower() for p in patterns)
        logger.info(f"  {'✓' if matched else '✗'} {name}")

    scope = "toutes compétitions" if all_leagues else "WTT/internationaux"
    logger.info(f"{len(results)} matchs {scope} à venir (prochains {days} jours)")
    return results, "sofascore"


def fetch_odds_for_event(session, event_id: int) -> tuple[float, float]:
    """Récupère les cotes actuelles pour un match Sofascore."""
    url = f"{API_BASE}/event/{event_id}/odds/1/all"
    data = _sfetch(session, url)
    markets = data.get("markets", [])
    winner_market = next((m for m in markets if m.get("marketName") == "Full time"), markets[0] if markets else None)
    
    if winner_market:
        choices = winner_market.get("choices", [])
        if len(choices) >= 2:
            o1_frac = choices[0].get("fractionalValue")
            o2_frac = choices[1].get("fractionalValue")
            
            def f_to_d(f):
                try:
                    n, d = f.split('/')
                    return round(float(n)/float(d) + 1.0, 3)
                except: return 0.0
            
            return f_to_d(o1_frac), f_to_d(o2_frac)
    return 0.0, 0.0


# ── Player lookup ─────────────────────────────────────────────────────────────

def _load_player_map() -> pd.DataFrame:
    """Charge {player_id, name, gender, ittf_rank, wtt_rank} depuis la DB."""
    query = text("""
        SELECT p.id, p.name, COALESCE(p.gender, 'M') AS gender,
               COALESCE(ir.rank, 9999) AS ittf_rank,
               COALESCE(wr.rank, 9999) AS wtt_rank
        FROM players p
        LEFT JOIN (
            SELECT player_id, rank,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM ittf_rankings
        ) ir ON ir.player_id = p.id AND ir.rn = 1
        LEFT JOIN (
            SELECT player_id, rank,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY snapshot_date DESC) AS rn
            FROM wtt_rankings
        ) wr ON wr.player_id = p.id AND wr.rn = 1
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def _normalize_name(s: str) -> str:
    """Normalise les caractères Unicode pour la comparaison de noms.
    Remplace tous les tirets/traits d'union Unicode (catégorie Pd) par un tiret ASCII.
    """
    import unicodedata
    s = unicodedata.normalize("NFC", s)
    return "".join(
        "-" if unicodedata.category(ch) == "Pd" else ch
        for ch in s
    ).lower().strip()


def _match_player(name: str, player_map: pd.DataFrame) -> int | None:
    """Retourne player_id depuis le nom (exact ou partiel).
    Gère les noms WTT en majuscules (MATSUSHIMA Sora → Matsushima S.)
    et les noms abrégés (initiale du prénom).
    """
    name_lower = _normalize_name(name)
    db_names = player_map["name"].apply(_normalize_name)

    # 1. Exact match
    exact = player_map[db_names == name_lower]
    if not exact.empty:
        return int(exact.iloc[0]["id"])

    parts = name_lower.split()
    if not parts:
        return None

    # 2. Partial match sur le premier token (nom de famille pour noms asiatiques/WTT)
    surname = parts[0]
    if len(surname) > 2:
        # Exclure les entrées doubles (noms contenant '/')
        by_surname = player_map[
            db_names.str.startswith(surname, na=False) &
            ~player_map["name"].str.contains("/", na=False)
        ]
        if len(by_surname) == 1:
            return int(by_surname.iloc[0]["id"])
        # Si plusieurs, affiner avec l'initiale du prénom
        if len(by_surname) > 1 and len(parts) > 1:
            initial = parts[1][0]
            refined = by_surname[db_names[by_surname.index].str.contains(
                rf"\b{re.escape(initial)}", na=False, regex=True
            )]
            if len(refined) == 1:
                return int(refined.iloc[0]["id"])

    # 3. Partial match sur le dernier token (noms occidentaux)
    last = parts[-1]
    if len(last) > 2:
        by_last = player_map[db_names.str.contains(last, na=False)]
        if len(by_last) == 1:
            return int(by_last.iloc[0]["id"])

    return None


# ── Feature building ──────────────────────────────────────────────────────────

def _build_player_stats(player_id: int) -> dict:
    """Construit les dernières stats connues pour un joueur depuis la DB."""
    query = text("""
        SELECT
            m.played_at,
            CASE WHEN m.player1_id = :pid THEN m.winner = 1
                 ELSE m.winner = 2 END AS won,
            CASE WHEN m.player1_id = :pid THEN m.score_p1
                 ELSE m.score_p2 END AS sets_won,
            CASE WHEN m.player1_id = :pid THEN m.score_p2
                 ELSE m.score_p1 END AS sets_lost
        FROM matches m
        WHERE (m.player1_id = :pid OR m.player2_id = :pid)
          AND m.is_walkover = 0
        ORDER BY m.played_at DESC
        LIMIT 10
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"pid": player_id})

    if df.empty:
        return {"form": 0.5, "avg_sets": 2.0, "rest_hours": 48.0, "fatigue": 0}

    df["played_at"] = pd.to_datetime(df["played_at"])
    last_match = df.iloc[0]["played_at"]
    rest_hours = min((pd.Timestamp.now() - last_match).total_seconds() / 3600, 168)

    # Forme sur les 5 derniers (pondération exponentielle)
    recent = df.head(5)
    weights = [0.85 ** i for i in range(len(recent))]
    wins = [1 if r else 0 for r in recent["won"]]
    form = sum(w * x for w, x in zip(weights, wins)) / sum(weights) if weights else 0.5

    avg_sets = df.head(5)["sets_won"].mean() if not df.empty else 2.0

    return {
        "form": round(form, 3),
        "avg_sets": round(avg_sets, 2),
        "rest_hours": round(rest_hours, 1),
        "fatigue": 1 if rest_hours < 48 else 0,
    }


def _get_elo(player_id: int) -> float:
    query = text("""
        SELECT rating FROM elo_ratings
        WHERE player_id = :pid
        ORDER BY computed_at DESC LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"pid": player_id}).fetchone()
    return float(row[0]) if row else 1500.0


def _get_h2h(p1_id: int, p2_id: int) -> dict:
    query = text("""
        SELECT winner, player1_id
        FROM matches
        WHERE ((player1_id = :p1 AND player2_id = :p2)
            OR (player1_id = :p2 AND player2_id = :p1))
          AND is_walkover = 0
        ORDER BY played_at DESC
        LIMIT 20
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"p1": p1_id, "p2": p2_id})

    if df.empty or len(df) < 1:
        return {"h2h_matches": 0, "h2h_winrate_p1": 0.5, "h2h_recent_winrate_p1": 0.5}

    p1_wins = sum(
        1 for _, r in df.iterrows()
        if (r["player1_id"] == p1_id and r["winner"] == 1)
        or (r["player1_id"] == p2_id and r["winner"] == 2)
    )
    total = len(df)
    recent = df.head(5)
    p1_wins_recent = sum(
        1 for _, r in recent.iterrows()
        if (r["player1_id"] == p1_id and r["winner"] == 1)
        or (r["player1_id"] == p2_id and r["winner"] == 2)
    )
    return {
        "h2h_matches": total,
        "h2h_winrate_p1": round(p1_wins / total, 3),
        "h2h_recent_winrate_p1": round(p1_wins_recent / len(recent), 3),
    }


def _elo_win_prob(elo1: float, elo2: float) -> float:
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))


def build_features_for_match(p1_id: int, p2_id: int,
                              p1_ittf: int, p2_ittf: int,
                              p1_wtt: int = 9999, p2_wtt: int = 9999) -> pd.DataFrame:
    feats = build_single_match_features(p1_id, p2_id, p1_ittf, p2_ittf, p1_wtt, p2_wtt)
    return pd.DataFrame([feats])


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prédictions matchs WTT/intl à venir")
    p.add_argument("--days", type=int, default=7, help="Horizon en jours (défaut: 7)")
    p.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="Confiance minimum à afficher (ex: 0.60)")
    p.add_argument("--all-leagues", action="store_true",
                   help="Inclure aussi Liga Pro, Setka Cup, ETTU, etc.")
    return p.parse_args()


def main():
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")

    # Charger le modèle
    model_path = f"data/{args.model}_model.pkl"
    model = LGBMModel.load(model_path) if args.model == "lgbm" else XGBModel.load(model_path)
    logger.info(f"Modèle {args.model.upper()} chargé")

    # Charger la map joueurs
    player_map = _load_player_map()
    logger.info(f"{len(player_map)} joueurs en DB")

    sfs_session = None
    upcoming, source = fetch_upcoming_matches(session=None, days=args.days, all_leagues=args.all_leagues)
    logger.info(f"Source: {source}")
    if not upcoming:
        logger.info("Aucun match WTT/international à venir trouvé")
        return

    # Prédire
    predictions = []
    not_found = []

    for ev in upcoming:
        p1_id = _match_player(ev["p1_name"], player_map)
        p2_id = _match_player(ev["p2_name"], player_map)

        if p1_id is None or p2_id is None:
            not_found.append(f"{ev['p1_name']} vs {ev['p2_name']}")
            continue

        p1_row = player_map[player_map["id"] == p1_id].iloc[0]
        p2_row = player_map[player_map["id"] == p2_id].iloc[0]

        p1_wtt = int(p1_row["wtt_rank"]) if int(p1_row["wtt_rank"]) < 9999 else 9999
        p2_wtt = int(p2_row["wtt_rank"]) if int(p2_row["wtt_rank"]) < 9999 else 9999
        features = build_features_for_match(
            p1_id, p2_id,
            int(p1_row["ittf_rank"]), int(p2_row["ittf_rank"]),
            p1_wtt, p2_wtt,
        )

        prob_p1 = float(model.predict_proba(features)[0])
        prob_p2 = 1 - prob_p1

        # Détecter le favori
        fav = ev["p1_name"] if prob_p1 >= 0.5 else ev["p2_name"]
        fav_prob = max(prob_p1, prob_p2)

        if fav_prob >= args.min_conf:
            import datetime as _dt
            st_val = ev["start_time"]
            if st_val == -1:
                start = "En cours"
            elif st_val:
                start = _dt.datetime.fromtimestamp(st_val).strftime("%d/%m %H:%M")
            else:
                start = "?"
            # --- NOUVEAU : Enregistrement Paper Betting ---
            o1, o2 = fetch_odds_for_event(sfs_session, ev["event_id"]) if sfs_session else (0.0, 0.0)
            
            # On calcule l'edge contre les cotes réelles si dispos
            edge_1 = prob_p1 - (1/o1) if o1 > 0 else 0
            edge_2 = prob_p2 - (1/o2) if o2 > 0 else 0
            
            # Enregistrement si Edge > 1% (on conserve même les petits edges pour l'audit)
            if (edge_1 > 0.01 or edge_2 > 0.01) and ev["status"] == "notstarted":
                # On cherche s'il n'existe pas déjà un match en DB pour lier l'ID si possible
                # Mais souvent le match n'est pas encore créé par le scraper principal.
                # On enregistre le pari en 'PENDING'.
                # Paramètres Kelly
                BANKROLL = 1000.0
                FRACTIONAL_KELLY = 0.1
                
                bet_p = 1 if edge_1 >= edge_2 else 2
                edge = max(edge_1, edge_2)
                odds = o1 if bet_p == 1 else o2
                prob = prob_p1 if bet_p == 1 else prob_p2
                
                # Kelly Formula: f = (p*o - 1) / (o - 1)
                f_star = (prob * odds - 1) / (odds - 1)
                stake = max(2.0, round(BANKROLL * f_star * FRACTIONAL_KELLY, 2))
                
                # Recherche du match_id dans la table 'matches' par external_id (sfs_{id})
                with get_session() as db_session:
                    exists = db_session.execute(
                        text("SELECT id FROM matches WHERE external_id = :ext"),
                        {"ext": f"sfs_{ev['event_id']}"}
                    ).fetchone()
                    
                    if exists:
                        mid = exists[0]
                        # Vérifier si on n'a pas déjà un pari pour ce match
                        already_bet = db_session.execute(
                            text("SELECT id FROM betting_records WHERE match_id = :mid AND is_paper = 1"),
                            {"mid": mid}
                        ).fetchone()
                        
                        if not already_bet:
                            rec = BettingRecord(
                                match_id=mid, bet_player=bet_p, stake=stake, odds=odds,
                                predicted_prob=round(prob, 3), model_edge=round(edge, 3),
                                result="PENDING", is_paper=True, placed_at=pd.Timestamp.now()
                            )
                            db_session.add(rec)
                            logger.info(f"   [Paper Bet] Enregistré PENDING: {ev['p1_name']} vs {ev['p2_name']} (@{odds}) | Stake: {stake}€ (Kelly)")

            elo_prob_p1 = float(features["elo_win_prob_p1"].iloc[0])
            elo_fav_prob = elo_prob_p1 if prob_p1 >= 0.5 else (1 - elo_prob_p1)
            delta_model_elo = fav_prob - elo_fav_prob
            predictions.append({
                "Date": start,
                "Tournoi": ev["tournament"][:30],
                "J1": ev["p1_name"],
                "WTT#1": p1_wtt if p1_wtt < 9999 else "-",
                "J2": ev["p2_name"],
                "WTT#2": p2_wtt if p2_wtt < 9999 else "-",
                "P(J1)": f"{prob_p1:.1%}",
                "P(J2)": f"{prob_p2:.1%}",
                "Favori": fav,
                "Confiance": f"{fav_prob:.1%}",
                "Δ Modèle/Elo": f"{delta_model_elo:+.1%}",
                "Edge(book)": f"{max(edge_1, edge_2):+.1%}" if (o1 > 0 and o2 > 0) else "-",
            })

    # Afficher
    if not predictions:
        logger.warning("Aucun match avec confiance suffisante")
        return

    df = pd.DataFrame(predictions).sort_values("Date")
    print(f"\n{'='*100}")
    print(f"  PREDICTIONS {args.model.upper()} - matchs WTT/internationaux ({len(df)} matchs)")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}")
    
    # --- SIMULATION DE TOURNOI (NOUVEAU) ---
    logger.info("Simulation des phases finales (World Cup / Brackets)...")
    sim = TournamentSimulator(model, player_map)
    leaders = sim.simulate_world_cup_groups(upcoming)
    projections = sim.project_knockout_stage(leaders)
    
    if projections:
        df_proj = pd.DataFrame(projections)
        df_proj.to_csv("data/tournament_projections.csv", index=False)
        logger.info(f"Projections de tournois sauvegardées ({len(projections)} rounds projetés)")
        print(f"\nProjections (Knockout Stage) :\n{df_proj.to_string(index=False)}")

    # Save to CSV for easy reading
    df.to_csv("data/upcoming_predictions.csv", index=False)
    logger.info(f"Prédictions sauvegardées dans data/upcoming_predictions.csv")

    if not_found:
        logger.warning(f"{len(not_found)} matchs ignorés (joueurs non trouvés en DB) :")
        for m in not_found[:10]:
            logger.warning(f"  - {m}")


if __name__ == "__main__":
    main()
