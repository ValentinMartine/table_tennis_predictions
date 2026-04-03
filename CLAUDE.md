# Table Tennis Prediction — Project Rules

## Stack
- Python 3.11, SQLite (SQLAlchemy), Streamlit, LightGBM, XGBoost
- Deployed on Render.com (free tier) via `render.yaml`

## Key paths
- `src/features/` — feature engineering (elo, h2h, form, pipeline)
- `src/models/` — lgbm_model.py, xgb_model.py (FEATURE_COLS defined in lgbm_model.py)
- `src/scraping/` — scrapers (Sofascore, WTT rankings, ITTF)
- `dashboard/app.py` — Streamlit UI
- `dashboard/queries.py` — all DB query functions
- `scripts/` — standalone CLI scripts (train, scrape, fetch rankings, backtest)
- `config/settings.yaml` — all hyperparameters and feature config
- `data/tt_matches.db` — SQLite DB (never read directly)

## Coding rules
- Always respond in English
- No comments unless logic is non-obvious
- No docstrings on unchanged functions
- No extra error handling for impossible cases
- No backwards-compat shims — just change the code

## Model
- 38 features defined in `src/models/lgbm_model.py::FEATURE_COLS`
- XGBModel imports FEATURE_COLS from lgbm_model — only edit once
- Train/val/test split is temporal (no shuffle)
- Calibration: isotonic via CalibratedClassifierCV

## DB schema (relevant tables)
- `matches` — played_at, player1_id, player2_id, winner, score_p1/p2, sets_detail, odds_p1/p2
- `players` — id, name, ittf_id, gender, country, date_of_birth
- `ittf_rankings` — player_id, rank, points, snapshot_date
- `wtt_rankings` — player_id, rank, points_ytd, snapshot_date
- `competitions` — comp_id, priority (1=Champions/Star Contender, 2=Contender, 99=skip)

## Rankings update commands
```bash
python scripts/fetch_wtt_rankings.py      # WTT men's rankings (week snapshot)
python scripts/fetch_ittf_rankings.py     # ITTF men's rankings (same WTT API)
# Women's rankings endpoint not yet found — manual update needed
```
