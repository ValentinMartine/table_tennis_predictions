import sys
import streamlit as st
import pandas as pd
from dashboard.queries import *
from src.backtesting.kelly import compute_stake

def render_tab_betting(_load_model):
    st.subheader("Paris")
    st.markdown(
        "L'**edge** mesure l'avantage par rapport aux cotes d'un bookmaker : "
        "si le modèle estime 65% de chances et que la cote implique 55%, l'edge est **+10%**. "
        "Un edge positif indique une opportunité de valeur."
    )

    model_lgbm = _load_model("lgbm")
    has_model = model_lgbm is not None

    if not has_model:
        st.warning("Modèle non disponible. Les probabilités Elo seront utilisées à la place.")

    all_names_edge = get_player_names(limit=500)

    if not all_names_edge:
        st.info("Aucun joueur en base de données.")
    else:
        ec1, ec2 = st.columns(2)
        with ec1:
            edge_p1 = st.selectbox("Joueur 1", [""] + all_names_edge, key="edge_p1")
        with ec2:
            p2_options = [n for n in all_names_edge if n != edge_p1] if edge_p1 else all_names_edge
            edge_p2 = st.selectbox("Joueur 2", [""] + p2_options, key="edge_p2")

        if edge_p1 and edge_p2:
            feats_preview = get_features_for_prediction(edge_p1, edge_p2)
            wtt_pts_p1 = feats_preview.get("_ittf_pts_p1")
            wtt_pts_p2 = feats_preview.get("_ittf_pts_p2")
            h2h_data = get_h2h_summary(edge_p1, edge_p2)

            st.divider()
            st.markdown("#### Comparaison des joueurs")

            info_col1, info_col2, info_col3 = st.columns([5, 2, 5])

            with info_col1:
                with st.container(border=True):
                    elo_p1_val = feats_preview["_elo_p1"]
                    ittf_p1 = feats_preview["ittf_rank_p1"]
                    st.markdown(f"**{edge_p1}**")
                    st.markdown(f"Elo : **{elo_p1_val:.0f}**")
                    st.markdown(f"Points WTT : **{int(wtt_pts_p1)}**" if wtt_pts_p1 else "Points WTT : **0**")
                    st.markdown(f"ITTF rank : **{'#' + str(ittf_p1) if ittf_p1 < 9999 else 'N/C'}**")
                    f5, f10, f20 = feats_preview["_form_5_p1"], feats_preview["_form_10_p1"], feats_preview["_form_20_p1"]
                    st.markdown(f"Forme (5|10|20) : **{f5:.0%}** | **{f10:.0%}** | **{f20:.0%}**")

            with info_col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if h2h_data["matches"] > 0:
                    st.markdown(
                        f"<div style='text-align:center;font-size:13px;color:#aaa'>H2H</div>"
                        f"<div style='text-align:center;font-size:22px;font-weight:bold'>"
                        f"{h2h_data['p1_wins']} – {h2h_data['p2_wins']}</div>"
                        f"<div style='text-align:center;font-size:11px;color:#aaa'>{h2h_data['matches']} matchs</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div style='text-align:center;font-size:13px;color:#aaa'>H2H<br>Aucun</div>", unsafe_allow_html=True)

            with info_col3:
                with st.container(border=True):
                    elo_p2_val = feats_preview["_elo_p2"]
                    ittf_p2 = feats_preview["ittf_rank_p2"]
                    st.markdown(f"**{edge_p2}**")
                    st.markdown(f"Elo : **{elo_p2_val:.0f}**")
                    st.markdown(f"Points WTT : **{int(wtt_pts_p2)}**" if wtt_pts_p2 else "Points WTT : **0**")
                    st.markdown(f"ITTF rank : **{'#' + str(ittf_p2) if ittf_p2 < 9999 else 'N/C'}**")
                    f5_2, f10_2, f20_2 = feats_preview["_form_5_p2"], feats_preview["_form_10_p2"], feats_preview["_form_20_p2"]
                    st.markdown(f"Forme (5|10|20) : **{f5_2:.0%}** | **{f10_2:.0%}** | **{f20_2:.0%}**")

            st.divider()

        if edge_p1 and edge_p2:
            oc1, oc2 = st.columns(2)
            with oc1:
                odds_p1 = st.number_input("Cote bookmaker (Joueur 1)", min_value=1.01, max_value=20.0, value=1.80, step=0.05, key="odds_p1")
            with oc2:
                odds_p2 = st.number_input("Cote bookmaker (Joueur 2)", min_value=1.01, max_value=20.0, value=2.10, step=0.05, key="odds_p2")

            bankroll_input = st.number_input("Bankroll (€)", min_value=10.0, max_value=100000.0, value=1000.0, step=50.0, key="bankroll_calc")

            feats = feats_preview
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

            kelly_stake_p1 = compute_stake(bankroll_input, use_prob, odds_p1, 0.25, 0.02) if edge_val > 0 else 0.0

            r1, r2, r3, r4 = st.columns(4)
            r1.metric(f"Prob. modèle ({edge_p1.split()[-1]})", f"{use_prob:.1%}", delta=f"Elo : {elo_prob:.1%}", delta_color="off")
            r2.metric("Prob. cotes", f"{implied_p1:.1%}")
            r3.metric(f"Edge {edge_p1.split()[-1]}", f"{edge_val:+.1%}", delta="Value" if edge_val > 0 else "Pas de value")
            r4.metric("Mise Kelly", f"{kelly_stake_p1:.2f} €" if kelly_stake_p1 > 0 else "Pas de value")

        else:
            st.info("Sélectionne deux joueurs et entre leurs cotes pour calculer l'edge.")

    st.divider()
    st.subheader("Paper Trading (Forward Testing)")
    st.markdown("Suivi des prédictions en cours sur les matchs à venir (enregistrés via `scripts/predict_upcoming.py`).")
    
    pending_df = get_pending_bets()
    if not pending_df.empty:
        st.dataframe(
            pending_df.style.format({
                "odds": "{:.2f}",
                "predicted_prob": "{:.1%}",
                "model_edge": "{:+.1%}"
            }),
            use_container_width=True,
            hide_index=True
        )
        if st.button("Actualiser les résultats (Reconcile)"):
            import subprocess
            subprocess.run([sys.executable, "scripts/reconcile_bets.py"])
            st.rerun()
    else:
        st.info("Aucun pari en attente. Lancez `scripts/predict_upcoming.py` pour en générer.")
