"""
Dashboard Streamlit — Table Tennis Prediction (Refactored)
Lancement : streamlit run dashboard/app.py
"""
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.queries import get_summary_stats, get_all_model_metrics

# -- Import des onglets modulaires --
from dashboard.tabs.tab_predictions import render_tab_predictions
from dashboard.tabs.tab_betting import render_tab_betting
from dashboard.tabs.tab_players import render_tab_players
from dashboard.tabs.tab_model import render_tab_model
from dashboard.tabs.tab_data import render_tab_data

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TT Predictions", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px; }
    .metric-card { background: #1e2130; border-radius: 10px; padding: 16px 20px; border: 1px solid #2d3250; }
    [data-testid="stToolbar"] { display: none; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

PRIORITY_LABELS = {1: "★ Priorité 1", 2: "Priorité 2", 3: "Priorité 3", 99: "Archivé"}
PRIORITY_COLORS = {1: "#2ecc71", 2: "#4c9be8", 3: "#f39c12", 99: "#7f8c8d"}

WTT_CALENDAR_2025_2026 = [
    {"tournoi": "WTT Star Contender Doha", "debut": "2025-01-06", "fin": "2025-01-11", "type": "Star Contender", "lieu": "Doha"},
    {"tournoi": "WTT Singapore Smash", "debut": "2025-01-30", "fin": "2025-02-09", "type": "Champions", "lieu": "Singapore"},
    {"tournoi": "WTT Champions Chongqing", "debut": "2025-03-11", "fin": "2025-03-16", "type": "Champions", "lieu": "Chongqing"},
    {"tournoi": "WTT Champions Incheon", "debut": "2026-03-27", "fin": "2026-03-31", "type": "Champions", "lieu": "Incheon"},
    {"tournoi": "ITTF World Cup Macao", "debut": "2026-04-15", "fin": "2026-04-21", "type": "World Cup", "lieu": "Macao"},
    {"tournoi": "Saudi Smash", "debut": "2026-05-01", "fin": "2026-05-11", "type": "Grand Smash", "lieu": "Jeddah"},
]

def fmt_date(val) -> str:
    if val is None: return "—"
    try: return pd.to_datetime(val).strftime("%d/%m/%Y")
    except Exception: return str(val)

@st.cache_resource
def _load_model(model_name: str = "lgbm"):
    path = Path(f"data/{model_name}_model.pkl")
    if not path.exists(): return None
    try:
        with open(path, "rb") as f: return pickle.load(f)
    except Exception: return None

st.title("Table Tennis Predictions")
st.markdown("Prédictions de matchs internationaux (WTT, Championnats du Monde, JO).")

stats = get_summary_stats()
all_metrics = get_all_model_metrics()
lgbm_acc = all_metrics.get("LGBM", {}).get("accuracy")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matchs analysés", f"{stats['total_matches']:,}" if stats['total_matches'] else "—")
c2.metric("Joueurs suivis", f"{stats['total_players']:,}" if stats['total_players'] else "—")
c3.metric("Précision LGBM", f"{lgbm_acc:.1%}" if lgbm_acc else "—")
c4.metric("Données jusqu'au", fmt_date(stats['latest_match']) if stats['latest_match'] else "—")

st.divider()

# ── ONGLETS ───────────────────────────────────────────────────────────────────
tab_pred, tab_calc, tab_players, tab_model, tab_donnees = st.tabs([
    "Prédictions", "Paris", "Joueurs", "Modèle", "Données"
])

with tab_pred:
    render_tab_predictions(WTT_CALENDAR_2025_2026, _load_model, fmt_date)
with tab_calc:
    render_tab_betting(_load_model)
with tab_players:
    render_tab_players(fmt_date)
with tab_model:
    render_tab_model()
with tab_donnees:
    render_tab_data(WTT_CALENDAR_2025_2026, PRIORITY_LABELS, PRIORITY_COLORS, fmt_date)
