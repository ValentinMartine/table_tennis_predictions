import streamlit as st
import pandas as pd
from datetime import datetime
from dashboard.queries import *
from scripts.predict_upcoming import fetch_upcoming_matches, _load_player_map, _match_player, build_features_for_match
import datetime as _dt

@st.cache_data(ttl=900, show_spinner=False)
def get_cached_matches(days):
    return fetch_upcoming_matches(days=days, all_leagues=False)

def render_tab_predictions(WTT_CALENDAR_2025_2026, _load_model, fmt_date):
    st.subheader("Prochains matchs WTT / Internationaux")
    st.markdown(
        "Prédictions pour les tournois WTT, Championnats du Monde et Jeux Olympiques "
        "à venir. Les données sont récupérées en temps réel depuis Sofascore."
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        days_ahead = st.slider("Horizon (jours)", min_value=1, max_value=30, value=15, key="pred_days")
    with col_cfg2:
        min_conf = st.slider("Confiance minimum (%)", min_value=50, max_value=90, value=50, key="pred_conf") / 100
    with col_cfg3:
        model_choice = st.radio("Modèle", ["lgbm", "xgb"], horizontal=True, key="pred_model",
                                captions=["LightGBM (recommandé)", "XGBoost"])

    col1, col2 = st.columns([8, 2])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Actualiser les matchs", use_container_width=True):
            get_cached_matches.clear()

    try:
        upmodel = _load_model(model_choice)
        if upmodel is None:
            st.warning(f"Modèle {model_choice.upper()} non disponible.")
            st.stop()

        with st.spinner("Recherche des matchs WTT sur Sofascore..."):
            matches = get_cached_matches(days_ahead)

        if not matches:
            st.info(
                "Aucun match WTT/international trouvé dans cet horizon. "
                "Les tournois WTT sont généralement publiés 1 à 2 semaines à l'avance."
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

                elo_prob_p1 = float(features["elo_win_prob_p1"].iloc[0])
                elo_fav_prob = elo_prob_p1 if prob_p1 >= 0.5 else (1 - elo_prob_p1)
                edge_vs_elo = round(fav_prob - elo_fav_prob, 4)

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
                        "Edge vs Elo": edge_vs_elo,
                    })

            st.success(f"{len(matches)} matchs internationaux trouvés sur Sofascore pour les {days_ahead} prochains jours.")

            if not predictions:
                st.info(
                    f"Parmi eux, aucun match ne correspond à tes critères (joueurs reconnus + confiance ≥ {min_conf:.0%}). "
                    f"Baisse la confiance ou augmente l'horizon !"
                )
            else:
                st.markdown(f"#### Prédictions du modèle {model_choice.upper()}")

                def _conf_color(val):
                    try:
                        v = float(str(val).replace("%", "").replace("+", "")) / 100
                        if v >= 0.75:
                            return "background-color: rgba(46,204,113,0.25)"
                        if v >= 0.65:
                            return "background-color: rgba(243,156,18,0.25)"
                        return ""
                    except Exception:
                        return ""

                def _edge_color(val):
                    try:
                        v = float(str(val).replace("%", "").replace("+", "")) / 100
                        if v >= 0.08:
                            return "background-color: rgba(46,204,113,0.35)"
                        if v >= 0.04:
                            return "background-color: rgba(46,204,113,0.15)"
                        if v < 0:
                            return "color: rgba(200,80,80,0.8)"
                        return ""
                    except Exception:
                        return ""

                display = pd.DataFrame(predictions).sort_values("Edge vs Elo", ascending=False)
                display["Confiance"] = display["Confiance"].apply(lambda v: f"{v:.1%}")
                display["Edge vs Elo"] = display["Edge vs Elo"].apply(lambda v: f"+{v:.1%}" if v >= 0 else f"{v:.1%}")
                st.dataframe(
                    display.style
                        .applymap(_conf_color, subset=["Confiance"])
                        .applymap(_edge_color, subset=["Edge vs Elo"]),
                    use_container_width=True, hide_index=True, height=500,
                )

            if not_found:
                with st.expander(f"⚠️ {len(not_found)} matchs ignorés (joueurs non reconnus en base)"):
                    st.markdown("Les noms sur Sofascore s'écrivent parfois différemment que dans la base ITTF.")
                    for m in not_found[:20]:
                        st.text(f"• {m}")

    except ImportError as e:
        st.error(f"Module manquant : {e}. Vérifie que `curl-cffi` est installé.")
    except Exception as e:
        st.error(f"Erreur lors de la récupération : {e}")
