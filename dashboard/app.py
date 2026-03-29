"""
Dashboard Streamlit — Table Tennis Prediction
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

from dashboard.queries import (
    get_all_model_metrics,
    get_betting_history,
    get_betting_stats_by_competition,
    get_competition_status,
    get_features_for_prediction,
    get_h2h,
    get_h2h_summary,
    get_ittf_ranking_coverage,
    get_latest_elo,
    get_matches_over_time,
    get_matches_per_competition,
    get_player_elo_history,
    get_player_info,
    get_player_ittf_rank,
    get_player_match_history,
    get_player_names,
    get_player_rolling_winrate,
    get_player_countries,
    get_player_stats,
    get_player_wtt_rank,
    get_recent_bets,
    get_rolling_roi,
    get_summary_stats,
    get_top_players,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TT Predictions",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #2d3250;
    }
</style>
""", unsafe_allow_html=True)

PRIORITY_LABELS = {1: "★ Priorité 1", 2: "Priorité 2", 3: "Priorité 3", 99: "Archivé"}
PRIORITY_COLORS = {1: "#2ecc71", 2: "#4c9be8", 3: "#f39c12", 99: "#7f8c8d"}

WTT_CALENDAR_2025_2026 = [
    {"tournoi": "WTT Star Contender Doha", "debut": "2025-01-06", "fin": "2025-01-11", "type": "Star Contender", "lieu": "Doha, QAT", "vainqueur_h": "Tomokazu Harimoto (JPN)", "vainqueur_f": "Kuai Man (CHN)"},
    {"tournoi": "WTT Singapore Smash", "debut": "2025-01-30", "fin": "2025-02-09", "type": "Champions", "lieu": "Singapore", "vainqueur_h": "Lin Shidong (CHN)", "vainqueur_f": "Sun Yingsha (CHN)"},
    {"tournoi": "WTT Champions Chongqing", "debut": "2025-03-11", "fin": "2025-03-16", "type": "Champions", "lieu": "Chongqing, CHN", "vainqueur_h": "Wang Chuqin (CHN)", "vainqueur_f": "Sun Yingsha (CHN)"},
    {"tournoi": "WTT Contender Tunis", "debut": "2025-04-22", "fin": "2025-04-27", "type": "Contender", "lieu": "Tunis, TUN", "vainqueur_h": "Felix Lebrun (FRA)", "vainqueur_f": "Miwa Harimoto (JPN)"},
    {"tournoi": "WTT Star Contender Ljubljana", "debut": "2025-06-17", "fin": "2025-06-22", "type": "Star Contender", "lieu": "Ljubljana, SLO", "vainqueur_h": "Hugo Calderano (BRA)", "vainqueur_f": "Miyu Nagasaki (JPN)"},
    {"tournoi": "WTT Contender Zagreb", "debut": "2025-06-24", "fin": "2025-06-29", "type": "Contender", "lieu": "Zagreb, CRO", "vainqueur_h": "Tomokazu Harimoto (JPN)", "vainqueur_f": "Satsuki Odo (JPN)"},
    {"tournoi": "WTT Contender Lagos", "debut": "2025-07-22", "fin": "2025-07-26", "type": "Contender", "lieu": "Lagos, NGR", "vainqueur_h": "Anders Lind (DEN)", "vainqueur_f": "Hana Goda (JPN)"},
    {"tournoi": "WTT Champions Yokohama", "debut": "2025-08-07", "fin": "2025-08-11", "type": "Champions", "lieu": "Yokohama, JPN", "vainqueur_h": "Tomokazu Harimoto (JPN)", "vainqueur_f": "Chen Xingtong (CHN)"},
    {"tournoi": "WTT Contender Almaty", "debut": "2025-09-02", "fin": "2025-09-07", "type": "Contender", "lieu": "Almaty, KAZ", "vainqueur_h": "Shunsuke Togami (JPN)", "vainqueur_f": "Honoka Hashimoto (JPN)"},
    {"tournoi": "WTT Champions Montpellier", "debut": "2025-10-28", "fin": "2025-11-02", "type": "Champions", "lieu": "Montpellier, FRA", "vainqueur_h": "Truls Möregård (SWE)", "vainqueur_f": "Wang Yidi (CHN)"},
    {"tournoi": "WTT Champions Frankfurt", "debut": "2025-11-04", "fin": "2025-11-09", "type": "Champions", "lieu": "Frankfurt, GER", "vainqueur_h": "Sora Matsushima (JPN)", "vainqueur_f": "Hina Hayata (JPN)"},
    {"tournoi": "WTT Star Contender Muscat", "debut": "2025-11-17", "fin": "2025-11-22", "type": "Star Contender", "lieu": "Muscat, OMA", "vainqueur_h": "Felix Lebrun (FRA)", "vainqueur_f": "Miyuu Kihara (JPN)"},
    {"tournoi": "WTT Finals Hong Kong", "debut": "2025-12-10", "fin": "2025-12-14", "type": "Cup Finals", "lieu": "Hong Kong, HKG", "vainqueur_h": "Tomokazu Harimoto (JPN)", "vainqueur_f": "Wang Manyu (CHN)"},
    {"tournoi": "WTT Champions Doha", "debut": "2026-01-07", "fin": "2026-01-11", "type": "Champions", "lieu": "Doha, QAT", "vainqueur_h": "Lin Yun-ju (TPE)", "vainqueur_f": "Zhu Yuling (MAC)"},
    {"tournoi": "WTT Singapore Smash", "debut": "2026-02-19", "fin": "2026-03-01", "type": "Champions", "lieu": "Singapore", "vainqueur_h": "Wang Chuqin (CHN)", "vainqueur_f": "Sun Yingsha (CHN)"},
    {"tournoi": "WTT Champions Chongqing", "debut": "2026-03-10", "fin": "2026-03-15", "type": "Champions", "lieu": "Chongqing, CHN", "vainqueur_h": "Felix Lebrun (FRA)", "vainqueur_f": "Miwa Harimoto (JPN)"},
]


def fmt_date(val) -> str:
    if val is None:
        return "—"
    try:
        return pd.to_datetime(val).strftime("%d/%m/%Y")
    except Exception:
        return str(val)


# ── CHARGEMENT DU MODÈLE (cached) ────────────────────────────────────────────

@st.cache_resource
def _load_model(model_name: str = "lgbm"):
    path = Path(f"data/{model_name}_model.pkl")
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ── HEADER ────────────────────────────────────────────────────────────────────

st.title("🏓 Table Tennis Predictions")
st.markdown(
    "Prédictions de matchs internationaux (WTT, Championnats du Monde, JO) "
    "basées sur un modèle **LightGBM** entraîné sur plus de **40 000 matchs** — "
    "Elo, forme récente, H2H, classements ITTF."
)

stats = get_summary_stats()
date_range = ""
if stats["earliest_match"] and stats["latest_match"]:
    date_range = f"{fmt_date(stats['earliest_match'])} – {fmt_date(stats['latest_match'])}"

all_metrics = get_all_model_metrics()
lgbm_acc = None
if "LGBM" in all_metrics:
    lgbm_acc = all_metrics["LGBM"].get("accuracy")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matchs analysés", f"{stats['total_matches']:,}" if stats['total_matches'] else "—")
c2.metric("Joueurs suivis", f"{stats['total_players']:,}" if stats['total_players'] else "—")
c3.metric("Précision du modèle", f"{lgbm_acc:.1%}" if lgbm_acc else "—")
c4.metric("Données jusqu'au", fmt_date(stats['latest_match']) if stats['latest_match'] else "—")

st.divider()

# ── ONGLETS ───────────────────────────────────────────────────────────────────

tab_pred, tab_calc, tab_players, tab_model, tab_donnees = st.tabs([
    "🎯 Prédictions", "🎰 Paris", "👤 Joueurs", "🤖 Modèle", "📊 Données"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTIONS (matchs WTT/internationaux à venir)
# ══════════════════════════════════════════════════════════════════════════════

with tab_pred:
    st.subheader("Prochains matchs WTT / Internationaux")
    st.markdown(
        "Prédictions pour les tournois WTT, Championnats du Monde et Jeux Olympiques "
        "à venir. Les données sont récupérées en temps réel depuis Sofascore."
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        days_ahead = st.slider("Horizon (jours)", min_value=1, max_value=14, value=7, key="pred_days")
    with col_cfg2:
        min_conf = st.slider("Confiance minimum (%)", min_value=50, max_value=90, value=60, key="pred_conf") / 100
    with col_cfg3:
        model_choice = st.radio("Modèle", ["lgbm", "xgb"], horizontal=True, key="pred_model",
                                captions=["LightGBM (recommandé)", "XGBoost"])

    if st.button("Charger les matchs à venir", type="primary", key="pred_fetch", use_container_width=False):
        try:
            from scripts.predict_upcoming import (
                fetch_upcoming_matches,
                _load_player_map,
                _match_player,
                build_features_for_match,
            )
            import datetime as _dt

            upmodel = _load_model(model_choice)
            if upmodel is None:
                st.warning(f"Modèle {model_choice.upper()} non disponible.")
                st.stop()

            with st.spinner("Connexion à Sofascore…"):
                matches = fetch_upcoming_matches(days=days_ahead, all_leagues=False)

            if not matches:
                st.info(
                    "Aucun match WTT/international trouvé dans cet horizon. "
                    "Les tournois WTT sont généralement publiés 1 à 2 semaines à l'avance.",
                    icon="📅"
                )
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

                    if fav_prob >= min_conf:
                        start = _dt.datetime.fromtimestamp(ev["start_time"]).strftime("%d/%m %H:%M") \
                                if ev["start_time"] else "?"
                        predictions.append({
                            "Date": start,
                            "Tournoi": ev["tournament"][:35],
                            "Joueur 1": ev["p1_name"],
                            "P(J1)": f"{prob_p1:.1%}",
                            "Joueur 2": ev["p2_name"],
                            "P(J2)": f"{prob_p2:.1%}",
                            "Favori": fav,
                            "Confiance": fav_prob,
                        })

                if not predictions:
                    st.info(
                        f"Aucun match avec confiance ≥ {min_conf:.0%} parmi "
                        f"{len(matches)} matchs trouvés.",
                        icon="ℹ️"
                    )
                else:
                    st.success(f"**{len(predictions)} matchs** — modèle {model_choice.upper()}, confiance ≥ {min_conf:.0%}")

                    def _conf_color(val):
                        try:
                            v = float(val)
                            if v >= 0.75:
                                return "background-color: rgba(46,204,113,0.25)"
                            if v >= 0.65:
                                return "background-color: rgba(243,156,18,0.25)"
                            return ""
                        except Exception:
                            return ""

                    display = pd.DataFrame(predictions).sort_values("Date")
                    display["Confiance"] = display["Confiance"].apply(lambda v: f"{v:.1%}")
                    st.dataframe(
                        display.style.applymap(_conf_color, subset=["Confiance"]),
                        use_container_width=True, hide_index=True, height=500,
                    )

                if not_found:
                    with st.expander(f"{len(not_found)} joueurs non trouvés en base"):
                        for m in not_found[:20]:
                            st.text(f"• {m}")

        except ImportError as e:
            st.error(f"Module manquant : {e}. Vérifie que `curl-cffi` est installé.")
        except Exception as e:
            st.error(f"Erreur lors de la récupération : {e}")
    else:
        # Afficher le calendrier WTT comme info d'attente
        st.info("Clique sur **Charger les matchs à venir** pour lancer les prédictions en temps réel.", icon="👆")
        st.markdown("#### Prochains tournois WTT")
        df_cal = pd.DataFrame(WTT_CALENDAR_2025_2026)
        df_cal["debut"] = pd.to_datetime(df_cal["debut"])
        df_cal["fin"] = pd.to_datetime(df_cal["fin"])
        today = pd.Timestamp.today().normalize()
        df_upcoming_cal = df_cal[df_cal["fin"] >= today].sort_values("debut").head(6)
        TYPE_COLORS = {
            "Champions": "#f1c40f", "Star Contender": "#e67e22",
            "Contender": "#3498db", "Cup Finals": "#9b59b6", "ITTF": "#2ecc71",
        }
        for _, row in df_upcoming_cal.iterrows():
            is_ongoing = row["debut"] <= today <= row["fin"]
            status = "🔴 En cours" if is_ongoing else "⏳ À venir"
            with st.container(border=True):
                c1c, c2c, c3c, c4c = st.columns([4, 2, 2, 1])
                c1c.markdown(f"**{row['tournoi']}**  \n📍 {row['lieu']}")
                c2c.markdown(f"Type  \n**{row['type']}**")
                c3c.markdown(f"{row['debut'].strftime('%d/%m')} → {row['fin'].strftime('%d/%m/%Y')}")
                c4c.markdown(status)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CALCULATEUR D'EDGE
# ══════════════════════════════════════════════════════════════════════════════

with tab_calc:
    st.subheader("Paris")
    st.markdown(
        "L'**edge** mesure l'avantage par rapport aux cotes d'un bookmaker : "
        "si le modèle estime 65% de chances et que la cote implique 55%, l'edge est **+10%**. "
        "Un edge positif indique une opportunité de valeur."
    )

    model_lgbm = _load_model("lgbm")
    model_xgb = _load_model("xgb")
    has_model = model_lgbm is not None

    if not has_model:
        st.warning("Modèle non disponible. Les probabilités Elo seront utilisées à la place.", icon="⚠️")

    all_names_edge = get_player_names(limit=500)

    if not all_names_edge:
        st.info("Aucun joueur en base de données.", icon="📭")
    else:
        ec1, ec2 = st.columns(2)
        with ec1:
            edge_p1 = st.selectbox("Joueur 1", [""] + all_names_edge, key="edge_p1")
        with ec2:
            # Exclure Joueur 1 de la liste Joueur 2
            p2_options = [n for n in all_names_edge if n != edge_p1] if edge_p1 else all_names_edge
            edge_p2 = st.selectbox("Joueur 2", [""] + p2_options, key="edge_p2")

        # ── Infos clés en priorité dès que les deux joueurs sont sélectionnés ──
        if edge_p1 and edge_p2:
            feats_preview = get_features_for_prediction(edge_p1, edge_p2)
            wtt_rank_p1, wtt_pts_p1 = get_player_wtt_rank(edge_p1)
            wtt_rank_p2, wtt_pts_p2 = get_player_wtt_rank(edge_p2)
            h2h_data = get_h2h_summary(edge_p1, edge_p2)

            st.divider()
            st.markdown("#### Comparaison des joueurs")

            info_col1, info_col2, info_col3 = st.columns([5, 2, 5])

            with info_col1:
                with st.container(border=True):
                    elo_p1_val = feats_preview["_elo_p1"]
                    ittf_p1 = feats_preview["ittf_rank_p1"]
                    form_p1 = feats_preview["form_p1"]
                    st.markdown(f"**{edge_p1}**")
                    st.markdown(f"Elo : **{elo_p1_val:.0f}**")
                    st.markdown(f"WTT rank : **{'#' + str(wtt_rank_p1) if wtt_rank_p1 < 9999 else 'N/C'}**"
                                + (f"  ·  {wtt_pts_p1:.0f} pts" if wtt_pts_p1 else ""))
                    st.markdown(f"ITTF rank : **{'#' + str(ittf_p1) if ittf_p1 < 9999 else 'N/C'}**")
                    st.markdown(f"Forme (5J) : **{form_p1:.0%}**")

            with info_col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                # H2H centré
                if h2h_data["matches"] > 0:
                    st.markdown(
                        f"<div style='text-align:center;font-size:13px;color:#aaa'>H2H</div>"
                        f"<div style='text-align:center;font-size:22px;font-weight:bold'>"
                        f"{h2h_data['p1_wins']} – {h2h_data['p2_wins']}</div>"
                        f"<div style='text-align:center;font-size:11px;color:#aaa'>{h2h_data['matches']} matchs</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div style='text-align:center;font-size:13px;color:#aaa'>H2H<br>Aucun</div>",
                        unsafe_allow_html=True,
                    )

            with info_col3:
                with st.container(border=True):
                    elo_p2_val = feats_preview["_elo_p2"]
                    ittf_p2 = feats_preview["ittf_rank_p2"]
                    form_p2 = feats_preview["form_p2"]
                    st.markdown(f"**{edge_p2}**")
                    st.markdown(f"Elo : **{elo_p2_val:.0f}**")
                    st.markdown(f"WTT rank : **{'#' + str(wtt_rank_p2) if wtt_rank_p2 < 9999 else 'N/C'}**"
                                + (f"  ·  {wtt_pts_p2:.0f} pts" if wtt_pts_p2 else ""))
                    st.markdown(f"ITTF rank : **{'#' + str(ittf_p2) if ittf_p2 < 9999 else 'N/C'}**")
                    st.markdown(f"Forme (5J) : **{form_p2:.0%}**")

            st.divider()

        # ── Cotes + Bankroll ──────────────────────────────────────────────────
        if edge_p1 and edge_p2:
            oc1, oc2 = st.columns(2)
            with oc1:
                odds_p1 = st.number_input("Cote bookmaker (Joueur 1)", min_value=1.01, max_value=20.0,
                                          value=1.80, step=0.05, key="odds_p1")
            with oc2:
                odds_p2 = st.number_input("Cote bookmaker (Joueur 2)", min_value=1.01, max_value=20.0,
                                          value=2.10, step=0.05, key="odds_p2")

            bankroll_input = st.number_input(
                "Bankroll (€)", min_value=10.0, max_value=100000.0,
                value=1000.0, step=50.0, key="bankroll_calc"
            )

            feats = feats_preview  # already fetched above
            elo_prob = feats["elo_win_prob_p1"]
            implied_p1 = (1 / odds_p1) / (1 / odds_p1 + 1 / odds_p2)

            model_prob = None
            if has_model:
                try:
                    feat_row = {k: v for k, v in feats.items() if not k.startswith("_")}
                    feat_row["implied_prob_p1"] = implied_p1
                    feat_row["has_odds"] = 1
                    sample = pd.DataFrame([feat_row])
                    model_prob = float(model_lgbm.predict_proba(sample)[0])
                except Exception as e:
                    st.warning(f"Erreur de prédiction : {e}")

            use_prob = model_prob if model_prob is not None else elo_prob
            edge_val = use_prob - (1 / odds_p1)
            edge_p2_val = (1 - use_prob) - (1 / odds_p2)

            from src.backtesting.kelly import compute_stake
            kelly_stake_p1 = 0.0
            kelly_stake_p2 = 0.0
            if edge_val > 0:
                kelly_stake_p1 = compute_stake(bankroll_input, use_prob, odds_p1, 0.25, 0.02)
            if edge_p2_val > 0:
                kelly_stake_p2 = compute_stake(bankroll_input, 1 - use_prob, odds_p2, 0.25, 0.02)

            # Résumé visuel
            r1, r2, r3, r4 = st.columns(4)
            r1.metric(
                f"Prob. modèle ({edge_p1.split()[-1]})",
                f"{use_prob:.1%}",
                delta=f"Elo : {elo_prob:.1%}",
                delta_color="off",
            )
            r2.metric("Prob. cotes implicite", f"{implied_p1:.1%}")
            edge_color_icon = "✅" if edge_val > 0.03 else ("⚠️" if edge_val > 0 else "❌")
            r3.metric(
                f"Edge {edge_p1.split()[-1]}",
                f"{edge_val:+.1%}",
                delta=edge_color_icon,
                delta_color="normal" if edge_val > 0 else "inverse",
            )
            r4.metric(
                "Mise Kelly (25%)",
                f"{kelly_stake_p1:.2f} €" if kelly_stake_p1 > 0 else "Pas de value",
            )

            st.divider()

            # Tableau de synthèse par joueur
            col_j1, col_j2 = st.columns(2)
            with col_j1:
                with st.container(border=True):
                    e_icon = "🟢" if edge_val >= 0.05 else ("🟡" if edge_val > 0 else "🔴")
                    st.markdown(f"### {e_icon} {edge_p1}")
                    st.markdown(f"- Probabilité modèle : **{use_prob:.1%}**")
                    st.markdown(f"- Cote : **{odds_p1}** (implicite : {implied_p1:.1%})")
                    st.markdown(f"- Edge : **{edge_val:+.1%}**")
                    if kelly_stake_p1 > 0:
                        st.success(f"Mise recommandée : **{kelly_stake_p1:.2f} €**")
                    else:
                        st.error("Pas de value — ne pas parier")

            with col_j2:
                with st.container(border=True):
                    e2_icon = "🟢" if edge_p2_val >= 0.05 else ("🟡" if edge_p2_val > 0 else "🔴")
                    st.markdown(f"### {e2_icon} {edge_p2}")
                    st.markdown(f"- Probabilité modèle : **{1 - use_prob:.1%}**")
                    st.markdown(f"- Cote : **{odds_p2}** (implicite : {1 - implied_p1:.1%})")
                    st.markdown(f"- Edge : **{edge_p2_val:+.1%}**")
                    if kelly_stake_p2 > 0:
                        st.success(f"Mise recommandée : **{kelly_stake_p2:.2f} €**")
                    else:
                        st.error("Pas de value — ne pas parier")

            # Features détaillées
            with st.expander("Détail des features utilisées"):
                detail_data = {
                    "Feature": ["Elo J1", "Elo J2", "Différence Elo", "Forme J1 (5 matchs)", "Forme J2 (5 matchs)", "H2H (matchs)", "Rang ITTF J1", "Rang ITTF J2"],
                    "Valeur": [
                        f"{feats['_elo_p1']:.0f}", f"{feats['_elo_p2']:.0f}",
                        f"{feats['elo_diff']:+.0f}", f"{feats['form_p1']:.1%}", f"{feats['form_p2']:.1%}",
                        feats['h2h_matches'],
                        str(feats['ittf_rank_p1']) if feats['ittf_rank_p1'] < 9999 else "N/C",
                        str(feats['ittf_rank_p2']) if feats['ittf_rank_p2'] < 9999 else "N/C",
                    ],
                }
                st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
        else:
            st.info("Sélectionne deux joueurs et entre leurs cotes pour calculer l'edge.", icon="👆")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — JOUEURS
# ══════════════════════════════════════════════════════════════════════════════

with tab_players:

    player_sub = st.tabs(["🏆 Top joueurs", "👤 Profil", "⚔️ Head-to-Head"])

    # ── Top joueurs ───────────────────────────────────────────────────────────
    with player_sub[0]:
        st.subheader("Classement des joueurs")
        st.caption("Classé par nombre de matchs joués dans les compétitions actives.")

        all_countries = get_player_countries()
        f1, f2, f3 = st.columns(3)
        with f1:
            gender_opt = st.selectbox("Sexe", ["Tous", "Hommes (M)", "Femmes (F)"], key="top_gender")
        with f2:
            min_m = st.slider("Matchs minimum", 5, 100, 10, step=5, key="top_min_matches")
        with f3:
            top_n = st.slider("Nombre de joueurs", 10, 100, 30, step=5, key="top_n")

        selected_countries = st.multiselect(
            "Filtrer par pays", all_countries, default=[], key="top_countries",
            placeholder="Tous les pays",
        )

        gender_filter = None
        if gender_opt == "Hommes (M)":
            gender_filter = "M"
        elif gender_opt == "Femmes (F)":
            gender_filter = "F"

        df_players = get_top_players(
            limit=top_n, min_matches=min_m, gender=gender_filter,
            countries=selected_countries if selected_countries else None,
            priority_max=98,
        )

        if df_players.empty:
            st.info("Aucun joueur trouvé avec ces critères.", icon="📭")
        else:
            if "date_of_birth" in df_players.columns:
                df_players["date_of_birth"] = pd.to_datetime(df_players["date_of_birth"], errors="coerce")
                today_ts = pd.Timestamp.today()
                df_players["age"] = df_players["date_of_birth"].apply(
                    lambda d: int((today_ts - d).days / 365.25) if pd.notna(d) else None
                )
            else:
                df_players["age"] = None

            color_col = "gender" if df_players["gender"].notna().any() else "country"
            df_plot = df_players.sort_values("win_rate_pct", ascending=True)
            fig3 = px.bar(
                df_plot, x="win_rate_pct", y="name",
                color=color_col, orientation="h",
                hover_data={"age": True, "gender": True, "country": True, "matches_played": True},
                color_discrete_map={"M": "#4c9be8", "F": "#e84c9b"},
                labels={
                    "win_rate_pct": "Win rate (%)",
                    "name": "Joueur", "gender": "Sexe",
                    "age": "Âge", "matches_played": "Matchs joués",
                },
                height=max(420, len(df_plot) * 22),
            )
            fig3.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig3, use_container_width=True)

            with st.expander("Tableau détaillé"):
                cols_show = ["name", "country", "gender", "age", "matches_played", "win_rate_pct"]
                cols_show = [c for c in cols_show if c in df_players.columns]
                st.dataframe(
                    df_players[cols_show].rename(columns={
                        "name": "Joueur", "country": "Pays", "gender": "Sexe",
                        "age": "Âge", "matches_played": "Matchs", "win_rate_pct": "Win rate (%)",
                    }),
                    use_container_width=True, hide_index=True,
                )

    # ── Profil joueur ─────────────────────────────────────────────────────────
    with player_sub[1]:
        st.subheader("Profil d'un joueur")

        all_names = get_player_names(limit=500)
        if not all_names:
            st.info("Aucun joueur en base de données.", icon="📭")
        else:
            selected_player = st.selectbox(
                "Rechercher un joueur", [""] + all_names, key="profile_player"
            )
            if selected_player:
                pstats = get_player_stats(selected_player)
                info = get_player_info(selected_player)
                elo_current = get_latest_elo(selected_player)
                ittf_rank = get_player_ittf_rank(selected_player)

                if pstats:
                    s1, s2, s3, s4, s5, s6 = st.columns(6)
                    s1.metric("Matchs joués", pstats["total_matches"])
                    s2.metric("Victoires", pstats["wins"])
                    s3.metric("Défaites", pstats["losses"])
                    s4.metric("Win rate", f"{pstats['win_rate']}%")
                    s5.metric("Rating Elo", f"{elo_current:.0f}" if elo_current != 1500.0 else "N/A")
                    s6.metric("Rang ITTF", str(ittf_rank) if ittf_rank != 9999 else "Non classé")

                    st.caption(
                        f"Pays : {info.get('country') or '?'} · "
                        f"Âge : {info.get('age') or '?'} ans · "
                        f"Période : {fmt_date(pstats['first_match'])} → {fmt_date(pstats['last_match'])}"
                    )

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    df_elo = get_player_elo_history(selected_player)
                    if not df_elo.empty:
                        st.markdown("#### Évolution Elo")
                        fig_elo = go.Figure()
                        fig_elo.add_trace(go.Scatter(
                            x=df_elo["played_at"], y=df_elo["elo_rating"],
                            mode="lines", line=dict(color="#4c9be8", width=2),
                            fill="tozeroy", fillcolor="rgba(76,155,232,0.1)",
                        ))
                        fig_elo.add_hline(y=1500, line_dash="dash", line_color="gray",
                                          opacity=0.4, annotation_text="1500 (initial)")
                        fig_elo.update_layout(
                            xaxis_title="", yaxis_title="Rating Elo",
                            height=300, showlegend=False, margin=dict(t=10),
                        )
                        st.plotly_chart(fig_elo, use_container_width=True)
                    else:
                        st.markdown("#### Win rate glissant (20 matchs)")
                        df_wr = get_player_rolling_winrate(selected_player, window=20)
                        if not df_wr.empty and df_wr["rolling_winrate"].notna().any():
                            fig_wr2 = go.Figure()
                            fig_wr2.add_trace(go.Scatter(
                                x=df_wr["played_at"], y=df_wr["rolling_winrate"],
                                mode="lines", line=dict(color="#2ecc71", width=2),
                                fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
                            ))
                            fig_wr2.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.4)
                            fig_wr2.update_layout(
                                xaxis_title="", yaxis_title="Win rate (%)",
                                height=300, showlegend=False, margin=dict(t=10),
                            )
                            st.plotly_chart(fig_wr2, use_container_width=True)
                        else:
                            st.info("Pas assez de données pour le graphique.")

                with chart_col2:
                    st.markdown("#### Win rate par compétition")
                    df_hist_all = get_player_match_history(selected_player, limit=500)
                    if not df_hist_all.empty:
                        df_hist_all["won"] = df_hist_all.apply(
                            lambda r: int(
                                (r["player1"] == selected_player and r["winner"] == 1) or
                                (r["player2"] == selected_player and r["winner"] == 2)
                            ), axis=1
                        )
                        comp_wr = (
                            df_hist_all.groupby("competition")
                            .agg(matches=("won", "count"), wins=("won", "sum"))
                            .reset_index()
                        )
                        comp_wr = comp_wr[comp_wr["matches"] >= 3]
                        comp_wr["win_rate"] = (comp_wr["wins"] / comp_wr["matches"] * 100).round(1)
                        comp_wr = comp_wr.sort_values("win_rate", ascending=True)
                        if not comp_wr.empty:
                            fig_comp_wr = px.bar(
                                comp_wr, x="win_rate", y="competition", orientation="h",
                                color="win_rate",
                                color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                                range_color=[30, 75],
                                labels={"win_rate": "Win rate (%)", "competition": ""},
                                height=300, text="win_rate",
                            )
                            fig_comp_wr.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.4)
                            fig_comp_wr.update_traces(texttemplate="%{text}%", textposition="outside")
                            fig_comp_wr.update_layout(coloraxis_showscale=False, margin=dict(t=10))
                            st.plotly_chart(fig_comp_wr, use_container_width=True)

                df_hist = get_player_match_history(selected_player, limit=50)
                if not df_hist.empty:
                    st.markdown("#### Derniers matchs")
                    df_hist["played_at"] = pd.to_datetime(df_hist["played_at"], errors="coerce")
                    df_hist["Score"] = df_hist.apply(lambda r: f"{r['score_p1']}-{r['score_p2']}", axis=1)
                    df_hist["Résultat"] = df_hist.apply(
                        lambda r: "✅ Gagné" if (
                            (r["player1"] == selected_player and r["winner"] == 1) or
                            (r["player2"] == selected_player and r["winner"] == 2)
                        ) else "❌ Perdu", axis=1
                    )
                    df_hist["Date"] = df_hist["played_at"].apply(fmt_date)
                    show_cols = ["Date", "competition", "player1", "player2", "Score", "Résultat", "round_name"]
                    show_cols = [c for c in show_cols if c in df_hist.columns]

                    def style_result_col(val):
                        return "color: #2ecc71" if "Gagné" in str(val) else "color: #e74c3c"

                    st.dataframe(
                        df_hist[show_cols].rename(columns={
                            "competition": "Compétition", "player1": "J1",
                            "player2": "J2", "round_name": "Tour",
                        }).style.applymap(style_result_col, subset=["Résultat"]),
                        use_container_width=True, hide_index=True, height=400,
                    )

    # ── Head-to-Head ──────────────────────────────────────────────────────────
    with player_sub[2]:
        st.subheader("Head-to-Head")
        st.caption("Historique des confrontations directes entre deux joueurs.")

        all_names_h2h = get_player_names(limit=500)
        if not all_names_h2h:
            st.info("Aucun joueur en base de données.", icon="📭")
        else:
            col_p1, col_vs, col_p2 = st.columns([5, 1, 5])
            with col_p1:
                p1 = st.selectbox("Joueur 1", [""] + all_names_h2h, key="h2h_p1")
            with col_vs:
                st.markdown("<br><div style='text-align:center;font-size:20px'>VS</div>", unsafe_allow_html=True)
            with col_p2:
                p2 = st.selectbox("Joueur 2", [""] + all_names_h2h, key="h2h_p2")

            if p1 and p2 and p1 != p2:
                summary = get_h2h_summary(p1, p2)

                if summary["matches"] == 0:
                    st.info(f"Aucun match direct entre **{p1}** et **{p2}** en base.")
                else:
                    h1, h2, h3 = st.columns([2, 1, 2])
                    with h1:
                        st.markdown(
                            f"<div style='text-align:center;font-size:32px;font-weight:bold'>"
                            f"{summary['p1_wins']}</div>"
                            f"<div style='text-align:center;color:#aaa'>{p1}</div>",
                            unsafe_allow_html=True,
                        )
                    with h2:
                        st.markdown(
                            f"<div style='text-align:center;font-size:20px;color:#888;padding-top:8px'>"
                            f"{summary['matches']} matchs</div>",
                            unsafe_allow_html=True,
                        )
                    with h3:
                        st.markdown(
                            f"<div style='text-align:center;font-size:32px;font-weight:bold'>"
                            f"{summary['p2_wins']}</div>"
                            f"<div style='text-align:center;color:#aaa'>{p2}</div>",
                            unsafe_allow_html=True,
                        )

                    df_h2h = get_h2h(p1, p2)
                    if not df_h2h.empty:
                        st.markdown("#### Historique des rencontres")
                        df_h2h["played_at"] = pd.to_datetime(df_h2h["played_at"], errors="coerce")
                        df_h2h["Score"] = df_h2h.apply(lambda r: f"{r['score_p1']}-{r['score_p2']}", axis=1)
                        df_h2h["Vainqueur"] = df_h2h.apply(
                            lambda r: r["player1"] if r["winner"] == 1 else r["player2"], axis=1
                        )
                        df_h2h["Date"] = df_h2h["played_at"].apply(fmt_date)
                        st.dataframe(
                            df_h2h[["Date", "competition", "player1", "player2", "Score", "Vainqueur", "round_name"]].rename(
                                columns={"competition": "Compétition", "player1": "J1",
                                         "player2": "J2", "round_name": "Tour"}
                            ),
                            use_container_width=True, hide_index=True, height=350,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODÈLE
# ══════════════════════════════════════════════════════════════════════════════

with tab_model:

    model_sub = st.tabs(["📊 Performance", "🔍 Comparaison modèles"])

    with model_sub[0]:
        st.subheader("Performance du modèle LGBM")

        model_path = Path("data/lgbm_model.pkl")
        metrics_path = Path("data/lgbm_metrics.json")
        shap_path = Path("data/lgbm_shap_importance.csv")
        calib_path = Path("data/lgbm_calibration.csv")

        if not model_path.exists():
            st.info("Modèle non disponible.", icon="📭")
        else:
            import os
            mtime = os.path.getmtime(model_path)
            dt = datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M")
            st.success(f"Modèle chargé — entraîné le {dt}")

            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text())
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                m2.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
                m3.metric("Log Loss", f"{metrics.get('log_loss', 0):.4f}")
                m4.metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.4f}")
                m5.metric("Brier Score", f"{metrics.get('brier_score', 0):.4f}")

                st.caption(
                    f"Entraîné sur {metrics.get('train_size', '?')} matchs · "
                    f"Testé sur {metrics.get('test_size', '?')} matchs · "
                    f"Split temporel (test ≥ 2025)"
                )

            col_shap, col_calib = st.columns(2)

            with col_shap:
                st.markdown("#### Importance des features (SHAP)")
                if shap_path.exists():
                    df_shap = pd.read_csv(shap_path)
                    df_shap = df_shap.sort_values("mean_abs_shap", ascending=True).tail(15)
                    fig_shap = px.bar(
                        df_shap, x="mean_abs_shap", y="feature", orientation="h",
                        color="mean_abs_shap",
                        color_continuous_scale=["#4c9be8", "#2ecc71"],
                        labels={"mean_abs_shap": "Importance SHAP", "feature": ""},
                        height=450,
                    )
                    fig_shap.update_layout(coloraxis_showscale=False, margin=dict(t=10))
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("Analyse SHAP non disponible.", icon="📭")

            with col_calib:
                st.markdown("#### Calibration des probabilités")
                st.caption("Une bonne calibration signifie que 60% prédit ≈ 60% réel.")
                if calib_path.exists():
                    df_calib = pd.read_csv(calib_path)
                    fig_calib = go.Figure()
                    fig_calib.add_trace(go.Scatter(
                        x=df_calib["prob_pred"], y=df_calib["prob_true"],
                        mode="lines+markers", name="Modèle",
                        line=dict(color="#4c9be8", width=2),
                    ))
                    fig_calib.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Calibration parfaite",
                        line=dict(color="gray", dash="dash"),
                    ))
                    fig_calib.update_layout(
                        xaxis_title="Probabilité prédite",
                        yaxis_title="Fréquence réelle",
                        height=450, legend=dict(x=0.02, y=0.98),
                    )
                    st.plotly_chart(fig_calib, use_container_width=True)
                else:
                    st.info("Données de calibration non disponibles.", icon="📭")

    with model_sub[1]:
        st.subheader("Comparaison des modèles")
        st.caption("LGBM, XGBoost et baseline Elo sur le jeu de test (matchs ≥ 2025).")

        all_metrics = get_all_model_metrics()

        if not all_metrics:
            st.info("Aucun modèle entraîné.", icon="📭")
        else:
            METRIC_DISPLAY = {
                "accuracy": "Accuracy", "f1_macro": "F1 (macro)",
                "precision": "Precision", "recall": "Recall",
                "mcc": "MCC", "roc_auc": "ROC AUC",
                "log_loss": "Log Loss", "brier_score": "Brier Score",
            }
            LOWER_BETTER = {"log_loss", "brier_score"}

            rows = []
            for model_name, m in all_metrics.items():
                row = {"Modèle": model_name}
                for key, label in METRIC_DISPLAY.items():
                    val = m.get(key)
                    row[label] = round(val, 4) if val is not None else None
                rows.append(row)

            df_compare = pd.DataFrame(rows).set_index("Modèle")
            st.dataframe(df_compare, use_container_width=True)

            st.markdown("#### Comparaison visuelle")
            metrics_to_plot = list(METRIC_DISPLAY.values())
            fig_comp = go.Figure()
            model_colors = {"LGBM": "#4c9be8", "XGB": "#e8834c", "Elo baseline": "#95a5a6"}

            for model_name in df_compare.index:
                color = model_colors.get(model_name, "#7f8c8d")
                vals = []
                for col in metrics_to_plot:
                    v = df_compare.loc[model_name, col]
                    if v is None:
                        vals.append(None)
                    elif col in ["Log Loss", "Brier Score"]:
                        vals.append(round(1 - v, 4))
                    else:
                        vals.append(v)

                fig_comp.add_trace(go.Bar(
                    name=model_name, x=metrics_to_plot, y=vals,
                    marker_color=color,
                ))

            fig_comp.update_layout(
                barmode="group", height=400,
                yaxis_title="Score (Log Loss et Brier inversés)",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.caption("Log Loss et Brier Score affichés en 1-valeur (plus haut = meilleur).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DONNÉES (Calendrier + Historique + Backtest + Suivi)
# ══════════════════════════════════════════════════════════════════════════════

with tab_donnees:

    explore_sub = st.tabs(["📅 Calendrier WTT", "📊 Historique", "📈 Backtest", "🎯 Suivi des paris"])

    # ── Calendrier ────────────────────────────────────────────────────────────
    with explore_sub[0]:
        st.subheader("Calendrier WTT 2025-2026")

        df_cal = pd.DataFrame(WTT_CALENDAR_2025_2026)
        df_cal["debut"] = pd.to_datetime(df_cal["debut"])
        df_cal["fin"] = pd.to_datetime(df_cal["fin"])
        today = pd.Timestamp.today().normalize()

        type_filter = st.multiselect(
            "Filtrer par type", options=df_cal["type"].unique().tolist(),
            default=df_cal["type"].unique().tolist(),
        )
        show_past = st.toggle("Afficher les tournois passés", value=False)

        TYPE_COLORS_CAL = {
            "Champions": "#f1c40f", "Star Contender": "#e67e22",
            "Contender": "#3498db", "Cup Finals": "#9b59b6", "ITTF": "#2ecc71",
        }

        df_filtered = df_cal[df_cal["type"].isin(type_filter)].copy()
        df_filtered["year"] = df_filtered["debut"].dt.year

        df_upcoming = df_filtered[df_filtered["fin"] >= today].sort_values("debut")
        df_past = df_filtered[df_filtered["fin"] < today].sort_values("debut", ascending=False)

        def _render_tournament(row):
            is_ongoing = row["debut"] <= today <= row["fin"]
            is_upcoming = row["debut"] > today
            is_past = row["fin"] < today
            status_icon = "🔴 En cours" if is_ongoing else ("⏳ À venir" if is_upcoming else "✅ Passé")
            with st.container(border=True):
                c1e, c2e, c3e, c4e = st.columns([4, 2, 2, 1])
                c1e.markdown(f"**{row['tournoi']}**  \n📍 {row['lieu']}")
                c2e.markdown(f"Type  \n**{row['type']}**")
                c3e.markdown(f"{row['debut'].strftime('%d/%m')} → {row['fin'].strftime('%d/%m/%Y')}")
                c4e.markdown(status_icon)
                if is_past:
                    vh = row.get("vainqueur_h", "")
                    vf = row.get("vainqueur_f", "")
                    if vh or vf:
                        parts = []
                        if vh:
                            parts.append(f"🏆 H : {vh}")
                        if vf:
                            parts.append(f"🏆 F : {vf}")
                        c1e.markdown("  \n".join(parts))

        # À venir / en cours
        if not df_upcoming.empty:
            st.markdown("#### ⏳ À venir")
            for _, row in df_upcoming.iterrows():
                _render_tournament(row)

        # Passés, groupés par année (plus récent en premier)
        if show_past and not df_past.empty:
            st.markdown("#### ✅ Passés")
            for year in sorted(df_past["year"].unique(), reverse=True):
                st.markdown(f"**{year}**")
                for _, row in df_past[df_past["year"] == year].iterrows():
                    _render_tournament(row)
        elif not show_past and df_upcoming.empty:
            st.info("Aucun tournoi à venir. Active **Afficher les tournois passés** pour voir l'historique.", icon="📅")

    # ── Données historiques ───────────────────────────────────────────────────
    with explore_sub[1]:
        st.subheader("Historique des données")

        df_status = get_competition_status()
        if not df_status.empty:
            now = datetime.utcnow()
            df_status["last_match"] = pd.to_datetime(df_status["last_match"], errors="coerce")
            df_status["first_match"] = pd.to_datetime(df_status["first_match"], errors="coerce")
            df_status["jours_depuis"] = df_status["last_match"].apply(
                lambda d: int((now - d).days) if pd.notna(d) else None
            )
            df_status["statut"] = df_status["priority"].map(PRIORITY_LABELS).fillna("—")

            df_active = df_status[df_status["priority"] < 99].copy()
            df_archived = df_status[df_status["priority"] == 99].copy()

            if not df_active.empty:
                st.markdown("**Compétitions actives**")
                display_active = df_active[["competition", "statut", "total_matches", "first_match", "last_match", "jours_depuis"]].copy()
                display_active.columns = ["Compétition", "Priorité", "Matchs", "Premier match", "Dernier match", "Jours depuis"]
                display_active["Premier match"] = display_active["Premier match"].apply(fmt_date)
                display_active["Dernier match"] = display_active["Dernier match"].apply(fmt_date)
                display_active["Jours depuis"] = display_active["Jours depuis"].apply(
                    lambda x: f"{x}j" if x is not None else "—"
                )
                st.dataframe(display_active, use_container_width=True, hide_index=True)

            with st.expander(f"Compétitions archivées ({len(df_archived)})"):
                if not df_archived.empty:
                    display_arch = df_archived[["competition", "total_matches", "first_match", "last_match"]].copy()
                    display_arch.columns = ["Compétition", "Matchs", "Premier match", "Dernier match"]
                    display_arch["Premier match"] = display_arch["Premier match"].apply(fmt_date)
                    display_arch["Dernier match"] = display_arch["Dernier match"].apply(fmt_date)
                    st.dataframe(display_arch, use_container_width=True, hide_index=True)

        st.divider()

        df_time = get_matches_over_time(include_archived=False)
        if not df_time.empty:
            df_time["month"] = pd.to_datetime(df_time["month"])
            fig = px.bar(
                df_time, x="month", y="matches", color="competition",
                title="Matchs par mois et compétition",
                labels={"month": "", "matches": "Matchs", "competition": "Compétition"},
                height=360,
            )
            fig.update_layout(legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Matchs par compétition")
            df_comp = get_matches_per_competition(include_archived=False)
            if not df_comp.empty:
                color_map = {v: PRIORITY_COLORS[k] for k, v in PRIORITY_LABELS.items()}
                df_comp["statut"] = df_comp["priority"].map(PRIORITY_LABELS).fillna("—")
                fig2 = px.bar(
                    df_comp.head(15), x="matches", y="competition",
                    orientation="h", color="statut",
                    color_discrete_map=color_map,
                    labels={"matches": "Matchs", "competition": "", "statut": "Priorité"},
                    height=420,
                )
                fig2.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            st.markdown("#### Couverture rankings ITTF")
            df_rank = get_ittf_ranking_coverage()
            if not df_rank.empty:
                fig4 = px.bar(
                    df_rank, x="year", y="players_ranked",
                    labels={"year": "Année", "players_ranked": "Joueurs classés"},
                    height=280, color_discrete_sequence=["#4c9be8"],
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Aucune donnée de ranking ITTF disponible.", icon="📭")


    # ── Backtest ──────────────────────────────────────────────────────────────
    with explore_sub[2]:
        st.subheader("Backtest — simulation de paris")
        st.markdown(
            "Simulation de paris sur les données historiques avec la stratégie Kelly fractionnaire. "
            "Permet d'évaluer la rentabilité théorique du modèle."
        )

        df_bets = get_betting_history(paper_only=True)
        if df_bets.empty:
            st.info(
                "Aucun historique de paris disponible. "
                "Le backtest n'a pas encore été exécuté sur ces données.",
                icon="📭"
            )
        else:
            df_bets["placed_at"] = pd.to_datetime(df_bets["placed_at"], errors="coerce")
            df_bets = df_bets.sort_values("placed_at")

            initial = 1000.0
            df_bets["cumulative_pl"] = df_bets["profit_loss"].cumsum()
            df_bets["bankroll"] = initial + df_bets["cumulative_pl"]
            df_bets["roi_pct"] = df_bets["cumulative_pl"] / initial * 100

            n = len(df_bets)
            final_roi = df_bets["roi_pct"].iloc[-1] if n > 0 else 0
            win_rate = (df_bets["result"] == "win").mean() * 100
            avg_edge = df_bets["model_edge"].mean() * 100
            peak = df_bets["bankroll"].cummax()
            drawdown = ((df_bets["bankroll"] - peak) / peak * 100).min()
            returns = df_bets["profit_loss"] / df_bets["stake"]
            sharpe = (returns.mean() / returns.std() * (n ** 0.5)) if returns.std() > 0 else 0

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Paris", f"{n:,}")
            k2.metric("ROI total", f"{final_roi:.1f}%", delta="✓ >3%" if final_roi > 3 else "✗ <3%")
            k3.metric("Win rate", f"{win_rate:.1f}%")
            k4.metric("Sharpe", f"{sharpe:.2f}", delta="✓ >1.2" if sharpe > 1.2 else "✗ <1.2")
            k5.metric("Drawdown max", f"{drawdown:.1f}%",
                      delta="✓ >-15%" if drawdown > -15 else "✗ <-15%",
                      delta_color="normal" if drawdown > -15 else "inverse")
            k6.metric("Edge moyen", f"{avg_edge:.1f}%")

            col_bl, col_edge = st.columns(2)

            with col_bl:
                st.markdown("#### Évolution du bankroll")
                fig_bl = go.Figure()
                fig_bl.add_trace(go.Scatter(
                    x=df_bets["placed_at"], y=df_bets["bankroll"],
                    fill="tozeroy", fillcolor="rgba(76,155,232,0.15)",
                    line=dict(color="#4c9be8", width=2), name="Bankroll",
                ))
                fig_bl.add_hline(y=initial, line_dash="dash",
                                 line_color="gray", opacity=0.6, annotation_text="Initial")
                fig_bl.update_layout(
                    xaxis_title="", yaxis_title="Bankroll (€)",
                    height=320, showlegend=False,
                )
                st.plotly_chart(fig_bl, use_container_width=True)

            with col_edge:
                st.markdown("#### Distribution des edges")
                fig_edge = px.histogram(
                    df_bets, x="model_edge", nbins=40, color="result",
                    color_discrete_map={"win": "#2ecc71", "loss": "#e74c3c", "pending": "#95a5a6"},
                    labels={"model_edge": "Edge modèle", "result": "Résultat"},
                    height=320,
                )
                fig_edge.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
                st.plotly_chart(fig_edge, use_container_width=True)

            st.markdown("#### ROI cumulatif (%)")
            fig_roi = px.line(
                df_bets, x="placed_at", y="roi_pct",
                labels={"placed_at": "", "roi_pct": "ROI (%)"},
                height=250,
                color_discrete_sequence=["#2ecc71" if final_roi > 0 else "#e74c3c"],
            )
            fig_roi.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_roi.add_hline(y=3, line_dash="dot", line_color="#2ecc71",
                              opacity=0.4, annotation_text="Objectif 3%")
            st.plotly_chart(fig_roi, use_container_width=True)

            df_comp_stats = get_betting_stats_by_competition()
            if not df_comp_stats.empty:
                st.markdown("#### Performance par compétition")

                def color_roi(val):
                    try:
                        v = float(val)
                        color = "#2ecc71" if v > 3 else "#e74c3c" if v < 0 else "#f39c12"
                        return f"color: {color}"
                    except Exception:
                        return ""

                st.dataframe(
                    df_comp_stats.style.applymap(color_roi, subset=["roi_pct"]),
                    use_container_width=True, hide_index=True,
                )

    # ── Monitoring ────────────────────────────────────────────────────────────
    with explore_sub[3]:
        st.subheader("Suivi des paris")

        mode = st.radio("Mode", ["Paper trading", "Paris réels"], horizontal=True)
        is_paper = mode == "Paper trading"

        df_mon = get_rolling_roi()
        df_recent = get_recent_bets(days=30, paper=is_paper)

        if df_mon.empty:
            st.info("Aucun pari enregistré.", icon="📭")
        else:
            df_mon_filtered = df_mon[df_mon["is_paper"] == (1 if is_paper else 0)]
            if df_mon_filtered.empty:
                st.info(f"Aucun pari en mode {'paper' if is_paper else 'réel'}.", icon="📭")
            else:
                n_total = len(df_mon_filtered)
                cum_roi = df_mon_filtered["cumulative_roi"].iloc[-1]
                peak_val = df_mon_filtered["cumulative_pl"].cummax().iloc[-1]
                current_pl = df_mon_filtered["cumulative_pl"].iloc[-1]

                STOP_LOSS = -10.0
                stop_triggered = cum_roi < STOP_LOSS

                if stop_triggered:
                    st.error(f"⛔ Stop-loss déclenché — ROI cumulatif : {cum_roi:.1f}%", icon="🚨")
                else:
                    st.success(f"ROI cumulatif : {cum_roi:.1f}% sur {n_total} paris", icon="✅")

                mon_c1, mon_c2, mon_c3 = st.columns(3)
                mon_c1.metric("P&L total", f"{current_pl:.2f} €")
                mon_c2.metric("Peak", f"{peak_val:.2f} €")
                mon_c3.metric("ROI cumulatif", f"{cum_roi:.1f}%")

                fig_mon = go.Figure()
                fig_mon.add_trace(go.Scatter(
                    x=df_mon_filtered["placed_at"] if "placed_at" in df_mon_filtered.columns else df_mon_filtered.index,
                    y=df_mon_filtered["cumulative_roi"],
                    mode="lines", line=dict(color="#4c9be8", width=2),
                    fill="tozeroy", fillcolor="rgba(76,155,232,0.1)",
                ))
                fig_mon.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_mon.add_hline(y=STOP_LOSS, line_dash="dot", line_color="#e74c3c",
                                  opacity=0.5, annotation_text=f"Stop-loss {STOP_LOSS}%")
                fig_mon.update_layout(
                    xaxis_title="", yaxis_title="ROI cumulatif (%)",
                    height=300, showlegend=False,
                )
                st.plotly_chart(fig_mon, use_container_width=True)

        st.markdown("#### Derniers paris (30 jours)")
        if df_recent.empty:
            st.info("Aucun pari sur les 30 derniers jours.", icon="📭")
        else:
            df_recent["placed_at"] = pd.to_datetime(df_recent["placed_at"])
            df_recent["edge_%"] = (df_recent["model_edge"] * 100).round(1)
            df_recent["prob_%"] = (df_recent["predicted_prob"] * 100).round(1)
            df_recent["P&L"] = df_recent["profit_loss"].round(2)

            def style_result(val):
                if val == "win":
                    return "background-color: rgba(46,204,113,0.2)"
                if val == "loss":
                    return "background-color: rgba(231,76,60,0.2)"
                return ""

            cols_show = ["placed_at", "competition", "player1", "player2",
                         "odds", "prob_%", "edge_%", "stake", "P&L", "result"]
            cols_show = [c for c in cols_show if c in df_recent.columns]
            st.dataframe(
                df_recent[cols_show].style.applymap(style_result, subset=["result"]),
                use_container_width=True, hide_index=True, height=400,
            )
