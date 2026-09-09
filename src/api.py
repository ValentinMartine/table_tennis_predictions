"""API on-demand pour les prédictions de matchs à venir."""
import datetime as dt
import os
from functools import lru_cache

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from scripts.predict_upcoming import (
    _load_player_map,
    _match_player,
    build_features_for_match,
    enrich_with_sofascore_odds,
    fetch_upcoming_matches,
)
from src.models.lgbm_model import LGBMModel
from src.scraping.oddsapi import enrich_with_bookmaker_odds


app = FastAPI(title="Table Tennis Predictions API", version="1.0.0")

origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "https://www.valentinmartine.com,https://valentinmartine.com,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _model() -> LGBMModel:
    return LGBMModel.load("data/lgbm_model.pkl")


@lru_cache(maxsize=1)
def _players():
    return _load_player_map()


def _format_start(value) -> str:
    if value == -1:
        return "in_progress"
    if value:
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).isoformat()
    return ""


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/predict/upcoming")
def predict_upcoming(
    days: int = Query(7, ge=1, le=30),
    min_conf: float = Query(0.0, ge=0.0, le=1.0),
    all_leagues: bool = False,
):
    model = _model()
    player_map = _players()

    upcoming, source = fetch_upcoming_matches(days=days, all_leagues=all_leagues)
    if source == "sofascore":
        upcoming = enrich_with_sofascore_odds(upcoming)
    else:
        upcoming = enrich_with_bookmaker_odds(upcoming, os.getenv("ODDS_API_KEY", ""))

    predictions = []
    not_found = []

    for event in upcoming:
        p1_id = _match_player(event["p1_name"], player_map)
        p2_id = _match_player(event["p2_name"], player_map)
        if p1_id is None or p2_id is None:
            not_found.append({"p1": event["p1_name"], "p2": event["p2_name"]})
            continue

        p1_row = player_map[player_map["id"] == p1_id].iloc[0]
        p2_row = player_map[player_map["id"] == p2_id].iloc[0]
        p1_wtt = int(p1_row["wtt_rank"]) if int(p1_row["wtt_rank"]) < 9999 else 9999
        p2_wtt = int(p2_row["wtt_rank"]) if int(p2_row["wtt_rank"]) < 9999 else 9999
        odds_p1 = event.get("book_odds_p1") or None
        odds_p2 = event.get("book_odds_p2") or None

        features = build_features_for_match(
            p1_id,
            p2_id,
            int(p1_row["ittf_rank"]),
            int(p2_row["ittf_rank"]),
            p1_wtt,
            p2_wtt,
            odds_p1=odds_p1,
            odds_p2=odds_p2,
        )
        prob_p1 = float(model.predict_proba(features)[0])
        prob_p2 = 1.0 - prob_p1
        confidence = max(prob_p1, prob_p2)
        if confidence < min_conf:
            continue

        edge_p1 = prob_p1 - (1 / odds_p1) if odds_p1 and odds_p1 > 1 else None
        edge_p2 = prob_p2 - (1 / odds_p2) if odds_p2 and odds_p2 > 1 else None

        predictions.append({
            "event_id": event.get("event_id"),
            "date": _format_start(event.get("start_time")),
            "tournament": event.get("tournament", ""),
            "round": event.get("round_name", ""),
            "p1": event["p1_name"],
            "p2": event["p2_name"],
            "p1_prob": round(prob_p1, 4),
            "p2_prob": round(prob_p2, 4),
            "favorite": event["p1_name"] if prob_p1 >= prob_p2 else event["p2_name"],
            "confidence": round(confidence, 4),
            "odds_p1": odds_p1,
            "odds_p2": odds_p2,
            "bookmaker": event.get("bookmaker"),
            "edge": round(max(x for x in (edge_p1, edge_p2) if x is not None), 4)
            if edge_p1 is not None or edge_p2 is not None else None,
        })

    predictions.sort(key=lambda row: row["date"] or "9999")
    return {
        "source": source,
        "days": days,
        "min_conf": min_conf,
        "count": len(predictions),
        "not_found_count": len(not_found),
        "predictions": predictions,
    }
