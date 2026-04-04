"""
Dashboard Streamlit - Table Tennis Prediction
Lancement : streamlit run dashboard/app.py
"""
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.queries import get_summary_stats, get_all_model_metrics
from dashboard.tabs.tab_betting import render_tab_betting
from dashboard.tabs.tab_players import render_tab_players
from dashboard.tabs.tab_model import render_tab_model

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TT Predictions", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px; }
    [data-testid="stToolbar"] { display: none; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


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


# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("Table Tennis Predictions")

stats = get_summary_stats()
all_metrics = get_all_model_metrics()
lgbm_acc = all_metrics.get("lgbm", all_metrics.get("LGBM", {})).get("accuracy")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matchs analysés", f"{stats['total_matches']:,}" if stats['total_matches'] else "—")
c2.metric("Joueurs suivis", f"{stats['total_players']:,}" if stats['total_players'] else "—")
c3.metric("Précision LGBM", f"{lgbm_acc:.1%}" if lgbm_acc else "—")
c4.metric("Données jusqu'au", fmt_date(stats['latest_match']) if stats['latest_match'] else "—")

st.divider()

# ── PRÉDICTIONS (page principale) ────────────────────────────────────────────
st.subheader("Prochains matchs - Prédictions")

col_cfg1, col_cfg2, col_cfg3, col_btn = st.columns([3, 3, 3, 2])
with col_cfg1:
    days_ahead = st.slider("Horizon (jours)", 1, 30, 15, key="pred_days")
with col_cfg2:
    min_conf = st.slider("Confiance min (%)", 50, 90, 50, key="pred_conf") / 100
with col_cfg3:
    model_choice = st.radio("Modèle", ["lgbm", "xgb", "ensemble"], horizontal=True, key="pred_model",
                            captions=["LightGBM ★", "XGBoost", "Ensemble"])
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Actualiser", use_container_width=True):
        st.cache_data.clear()

try:
    from scripts.predict_upcoming import fetch_upcoming_matches, _load_player_map, _match_player, build_features_for_match
    import datetime as _dt

    @st.cache_data(ttl=900, show_spinner=False)
    def _get_matches_list(days):
        result = fetch_upcoming_matches(days=days, all_leagues=False)
        matches = result[0] if isinstance(result, tuple) else result
        source = result[1] if isinstance(result, tuple) else "wtt_api"
        return matches, source

    upmodel = _load_model(model_choice)
    if upmodel is None:
        st.warning(f"Modèle {model_choice.upper()} non disponible.")
    else:
        with st.spinner("Recherche des prochains matchs..."):
            matches, data_source = _get_matches_list(days_ahead)
        source_label = "Sofascore" if data_source == "sofascore" else "WTT API officielle"

        if not matches:
            st.info("Aucun match WTT/international trouvé dans cet horizon.")
        else:
            player_map = _load_player_map()
            predictions = []
            not_found = []

            for ev in matches:
                p1_id = _match_player(ev["p1_name"], player_map)
                p2_id = _match_player(ev["p2_name"], player_map)
                if p1_id is None or p2_id is None:
                    not_found.append(f"{ev['p1_name']} vs {ev['p2_name']}")
                    continue

                p1_row = player_map[player_map["id"] == p1_id].iloc[0]
                p2_row = player_map[player_map["id"] == p2_id].iloc[0]
                features = build_features_for_match(
                    p1_id, p2_id,
                    int(p1_row["ittf_rank"]), int(p2_row["ittf_rank"])
                )

                prob_p1 = float(upmodel.predict_proba(features)[0])
                prob_p2 = 1 - prob_p1
                fav = ev["p1_name"] if prob_p1 >= 0.5 else ev["p2_name"]
                fav_prob = max(prob_p1, prob_p2)

                elo_prob_p1 = float(features["elo_win_prob_p1"].iloc[0])
                elo_fav_prob = elo_prob_p1 if prob_p1 >= 0.5 else (1 - elo_prob_p1)
                edge_vs_elo = round(fav_prob - elo_fav_prob, 4)

                st_val = ev["start_time"]
                if st_val == -1:
                    start = "En cours"
                elif st_val:
                    start = _dt.datetime.fromtimestamp(st_val).strftime("%d/%m %H:%M")
                else:
                    start = "?"
                wtt1 = int(p1_row["wtt_rank"]) if int(p1_row["wtt_rank"]) < 9999 else "—"
                wtt2 = int(p2_row["wtt_rank"]) if int(p2_row["wtt_rank"]) < 9999 else "—"
                g = ev.get("gender", "M")
                genre_badge = "🔵" if g == "M" else "🩷"
                predictions.append({
                    "Date": start,
                    "G": genre_badge,
                    "Tournoi": ev["tournament"][:35],
                    "Joueur 1": ev["p1_name"],
                    "WTT1": wtt1,
                    "prob_p1": prob_p1,
                    "prob_p2": prob_p2,
                    "WTT2": wtt2,
                    "Joueur 2": ev["p2_name"],
                    "Favori": fav,
                    "Confiance": fav_prob,
                    "Edge vs Elo": edge_vs_elo,
                })

            # Partager avec l'onglet Paris
            st.session_state["live_predictions"] = predictions

            filtered = [p for p in predictions if p["Confiance"] >= min_conf]
            st.caption(f"Source : {source_label} · {len(matches)} matchs trouvés · {len(filtered)} avec confiance ≥ {min_conf:.0%} · {len(not_found)} joueurs non reconnus")

            if not filtered:
                st.info("Aucun match ne correspond aux critères. Baisse la confiance ou augmente l'horizon.")
            else:
                def _prob_color(val):
                    try:
                        v = float(str(val).replace("%", "")) / 100
                        if v >= 0.65: return "background-color: rgba(46,204,113,0.30); font-weight: bold"
                        if v >= 0.50: return "background-color: rgba(46,204,113,0.12)"
                        if v <= 0.35: return "background-color: rgba(200,80,80,0.20); color: rgba(200,80,80,0.9)"
                        return "color: rgba(200,80,80,0.7)"
                    except Exception: return ""

                def _conf_color(val):
                    try:
                        v = float(str(val).replace("%", "")) / 100
                        if v >= 0.75: return "background-color: rgba(46,204,113,0.25)"
                        if v >= 0.65: return "background-color: rgba(243,156,18,0.25)"
                        return ""
                    except Exception: return ""

                def _edge_color(val):
                    try:
                        v = float(str(val).replace("%", "").replace("+", "")) / 100
                        if v >= 0.08: return "background-color: rgba(46,204,113,0.35)"
                        if v >= 0.04: return "background-color: rgba(46,204,113,0.15)"
                        if v < 0: return "color: rgba(200,80,80,0.8)"
                        return ""
                    except Exception: return ""

                display = pd.DataFrame(filtered).sort_values("Edge vs Elo", ascending=False)
                display["P(J1)"] = display["prob_p1"].apply(lambda v: f"{v:.1%}")
                display["P(J2)"] = display["prob_p2"].apply(lambda v: f"{v:.1%}")
                display["Confiance"] = display["Confiance"].apply(lambda v: f"{v:.1%}")
                display["Edge vs Elo"] = display["Edge vs Elo"].apply(lambda v: f"+{v:.1%}" if v >= 0 else f"{v:.1%}")
                cols = ["Date", "G", "Tournoi", "Joueur 1", "WTT1", "P(J1)", "Joueur 2", "WTT2", "P(J2)", "Favori", "Confiance", "Edge vs Elo"]
                st.dataframe(
                    display[cols].style
                        .map(_prob_color, subset=["P(J1)", "P(J2)"])
                        .map(_conf_color, subset=["Confiance"])
                        .map(_edge_color, subset=["Edge vs Elo"]),
                    use_container_width=True, hide_index=True, height=500,
                )

            if not_found:
                with st.expander(f"{len(not_found)} matchs ignorés (joueurs non reconnus)"):
                    for m in not_found[:20]:
                        st.text(f"• {m}")

except ImportError as e:
    st.error(f"Module manquant : {e}. Vérifie que `curl-cffi` est installé.")
except Exception as e:
    st.error(f"Erreur : {e}")

st.divider()

# ── ONGLETS ───────────────────────────────────────────────────────────────────
tab_players, tab_betting, tab_model = st.tabs(["Joueurs", "Paris", "Modèle"])

with tab_players:
    render_tab_players(fmt_date)
with tab_betting:
    render_tab_betting(_load_model)
with tab_model:
    render_tab_model()
