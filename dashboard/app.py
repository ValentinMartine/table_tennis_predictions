"""
Dashboard Streamlit — Table Tennis Prediction
Lancement : streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.queries import (
    get_betting_history,
    get_betting_stats_by_competition,
    get_ittf_ranking_coverage,
    get_matches_over_time,
    get_matches_per_competition,
    get_player_countries,
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
    .metric-card {
        background: #1e1e2e; border-radius: 10px;
        padding: 16px 20px; margin: 4px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

EMPTY_MSG = "Aucune donnée. Lance `python scripts/bootstrap_data.py` pour initialiser."


def empty(msg: str = EMPTY_MSG):
    st.info(msg, icon="📭")


# ── HEADER ────────────────────────────────────────────────────────────────────

st.title("🏓 Table Tennis Predictions")

stats = get_summary_stats()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Matchs en base", f"{stats['total_matches']:,}")
c2.metric("Joueurs", f"{stats['total_players']:,}")
c3.metric("Compétitions", f"{stats['total_competitions']:,}")
c4.metric("Avec cotes", f"{stats['matches_with_odds']:,}")
odds_pct = (
    round(stats["matches_with_odds"] / stats["total_matches"] * 100)
    if stats["total_matches"] > 0 else 0
)
c5.metric("Couverture cotes", f"{odds_pct}%")

st.divider()

# ── ONGLETS ───────────────────────────────────────────────────────────────────

tab_data, tab_model, tab_backtest, tab_live, tab_monitoring = st.tabs([
    "📊 Data", "🤖 Modèle", "📈 Backtest", "⚡ Live", "🎯 Monitoring"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════

with tab_data:
    st.subheader("Vue d'ensemble des données")

    # Couverture temporelle
    df_time = get_matches_over_time()
    if df_time.empty:
        empty()
    else:
        df_time["month"] = pd.to_datetime(df_time["month"])
        fig = px.bar(
            df_time, x="month", y="matches", color="competition",
            title="Matchs par mois et compétition",
            labels={"month": "", "matches": "Matchs", "competition": "Compétition"},
            height=350,
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    # Matchs par compétition
    with col_left:
        st.markdown("#### Matchs par compétition")
        df_comp = get_matches_per_competition()
        if df_comp.empty:
            empty()
        else:
            fig2 = px.bar(
                df_comp.head(15), x="matches", y="competition",
                orientation="h", color="type",
                labels={"matches": "Matchs", "competition": "", "type": "Type"},
                height=400,
                color_discrete_map={"league": "#4c9be8", "international": "#e8834c"},
            )
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

    # Top joueurs
    with col_right:
        st.markdown("#### Top joueurs")

        all_countries = get_player_countries()
        f1, f2, f3 = st.columns(3)
        with f1:
            gender_opt = st.selectbox(
                "Sexe", ["Tous", "Hommes (M)", "Femmes (F)"], key="top_gender"
            )
        with f2:
            min_m = st.slider("Matchs min", 5, 100, 10, step=5, key="top_min_matches")
        with f3:
            top_n = st.slider("Nombre de joueurs", 10, 100, 20, step=5, key="top_n")

        selected_countries = st.multiselect(
            "Pays", all_countries, default=[], key="top_countries",
            placeholder="Tous les pays",
        )

        gender_filter = None
        if gender_opt == "Hommes (M)":
            gender_filter = "M"
        elif gender_opt == "Femmes (F)":
            gender_filter = "F"

        df_players = get_top_players(
            limit=top_n,
            min_matches=min_m,
            gender=gender_filter,
            countries=selected_countries if selected_countries else None,
        )

        if df_players.empty:
            empty()
        else:
            # Calcul âge
            if "date_of_birth" in df_players.columns:
                df_players["date_of_birth"] = pd.to_datetime(
                    df_players["date_of_birth"], errors="coerce"
                )
                today = pd.Timestamp.today()
                df_players["age"] = df_players["date_of_birth"].apply(
                    lambda d: int((today - d).days / 365.25) if pd.notna(d) else None
                )
            else:
                df_players["age"] = None

            color_col = "gender" if df_players["gender"].notna().any() else "country"
            hover_data = {"age": True, "gender": True, "country": True}

            fig3 = px.scatter(
                df_players, x="matches_played", y="win_rate_pct",
                text="name", color=color_col,
                hover_data=hover_data,
                color_discrete_map={"M": "#4c9be8", "F": "#e84c9b"},
                labels={
                    "matches_played": "Matchs joués",
                    "win_rate_pct": "Win rate (%)",
                    "gender": "Sexe",
                    "age": "Âge",
                },
                height=380,
            )
            fig3.update_traces(textposition="top center", textfont_size=9)
            fig3.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig3, use_container_width=True)

            # Tableau détaillé
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

    # Rankings ITTF historiques
    st.markdown("#### Couverture rankings ITTF (snapshots hebdomadaires)")
    df_rank = get_ittf_ranking_coverage()
    if df_rank.empty:
        empty("Aucun ranking ITTF. Lance `python scripts/bootstrap_data.py`.")
    else:
        fig4 = px.bar(
            df_rank, x="year", y="players_ranked",
            labels={"year": "Année", "players_ranked": "Joueurs classés"},
            height=250,
            color_discrete_sequence=["#4c9be8"],
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Table détaillée
    if not df_comp.empty:
        st.markdown("#### Détail par compétition")
        df_display = df_comp.copy()
        df_display["odds_%"] = (df_display["with_odds"] / df_display["matches"] * 100).round(1)
        st.dataframe(
            df_display[["competition", "type", "matches", "with_odds", "odds_%", "first_match", "last_match"]],
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODÈLE
# ══════════════════════════════════════════════════════════════════════════════

with tab_model:
    st.subheader("Performance du modèle")

    # Chargement des métriques si disponibles
    metrics_path = Path("data/model_metrics.json")
    shap_path = Path("data/shap_importance.csv")
    calib_path = Path("data/calibration_data.csv")

    if not metrics_path.exists():
        empty(
            "Aucune métrique disponible. "
            "Lance `python scripts/train_model.py --model lgbm --shap` pour entraîner."
        )
    else:
        import json
        metrics = json.loads(metrics_path.read_text())

        # Métriques principales
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}",
                  delta=f"{(metrics.get('accuracy', 0) - metrics.get('elo_accuracy', 0.5)):.1%} vs Elo")
        m2.metric("Log Loss", f"{metrics.get('log_loss', 0):.4f}",
                  delta=f"{metrics.get('log_loss', 0) - metrics.get('elo_log_loss', 0):.4f} vs Elo",
                  delta_color="inverse")
        m3.metric("Brier Score", f"{metrics.get('brier_score', 0):.4f}", delta_color="inverse")
        m4.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
        m5.metric("Échantillons test", f"{metrics.get('n_samples', 0):,}")

        col_shap, col_calib = st.columns(2)

        # SHAP feature importance
        with col_shap:
            st.markdown("#### Importance des features (SHAP)")
            if shap_path.exists():
                df_shap = pd.read_csv(shap_path)
                fig_shap = px.bar(
                    df_shap.head(15), x="mean_abs_shap", y="feature",
                    orientation="h",
                    labels={"mean_abs_shap": "SHAP moyen |valeur|", "feature": ""},
                    color="mean_abs_shap",
                    color_continuous_scale="Blues",
                    height=450,
                )
                fig_shap.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                empty("SHAP non calculé. Relancer avec `--shap`.")

        # Courbe de calibration
        with col_calib:
            st.markdown("#### Calibration des probabilités")
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
                    height=450,
                    legend=dict(x=0.02, y=0.98),
                )
                st.plotly_chart(fig_calib, use_container_width=True)
            else:
                empty("Données de calibration non disponibles.")

    # CV scores si dispo
    cv_path = Path("data/cv_scores.json")
    if cv_path.exists():
        import json
        cv = json.loads(cv_path.read_text())
        st.markdown("#### Validation croisée (5-fold temporel)")
        st.json(cv)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.subheader("Résultats du backtesting")

    df_bets = get_betting_history(paper_only=True)

    if df_bets.empty:
        empty(
            "Aucun pari simulé. "
            "Lance `python scripts/backtest.py --plot` pour générer le backtest."
        )
    else:
        df_bets["placed_at"] = pd.to_datetime(df_bets["placed_at"])
        df_bets = df_bets.sort_values("placed_at")
        df_bets["profit_loss"] = pd.to_numeric(df_bets["profit_loss"], errors="coerce").fillna(0)
        df_bets["stake"] = pd.to_numeric(df_bets["stake"], errors="coerce").fillna(0)
        df_bets["model_edge"] = pd.to_numeric(df_bets["model_edge"], errors="coerce").fillna(0)

        # Bankroll (reconstruction)
        initial = 1000.0
        df_bets["cumulative_pl"] = df_bets["profit_loss"].cumsum()
        df_bets["bankroll"] = initial + df_bets["cumulative_pl"]
        df_bets["roi_pct"] = df_bets["cumulative_pl"] / initial * 100

        # KPIs
        n = len(df_bets)
        final_roi = df_bets["roi_pct"].iloc[-1] if n > 0 else 0
        win_rate = (df_bets["result"] == "win").mean() * 100
        avg_odds = df_bets["odds"].mean()
        avg_edge = df_bets["model_edge"].mean() * 100

        # Drawdown
        peak = df_bets["bankroll"].cummax()
        drawdown = ((df_bets["bankroll"] - peak) / peak * 100).min()

        # Sharpe
        returns = df_bets["profit_loss"] / df_bets["stake"]
        sharpe = (returns.mean() / returns.std() * (n ** 0.5)) if returns.std() > 0 else 0

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Paris", f"{n:,}")
        k2.metric("ROI total", f"{final_roi:.1f}%",
                  delta="✓ >3%" if final_roi > 3 else "✗ <3%")
        k3.metric("Win rate", f"{win_rate:.1f}%")
        k4.metric("Sharpe", f"{sharpe:.2f}",
                  delta="✓ >1.2" if sharpe > 1.2 else "✗ <1.2")
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
                line=dict(color="#4c9be8", width=2),
                name="Bankroll",
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
                df_bets, x="model_edge",
                nbins=40, color="result",
                color_discrete_map={"win": "#2ecc71", "loss": "#e74c3c", "pending": "#95a5a6"},
                labels={"model_edge": "Edge modèle", "result": "Résultat"},
                height=320,
            )
            fig_edge.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
            st.plotly_chart(fig_edge, use_container_width=True)

        # ROI cumulatif
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

        # Stats par compétition
        st.markdown("#### Performance par compétition")
        df_comp_stats = get_betting_stats_by_competition()
        if not df_comp_stats.empty:
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE
# ══════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.subheader("Prédictions live")

    col_refresh, col_status = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.rerun()
    with col_status:
        model_path = Path("data/lgbm_model.pkl")
        if model_path.exists():
            import os
            mtime = os.path.getmtime(model_path)
            from datetime import datetime
            dt = datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M")
            st.success(f"Modèle chargé — dernière mise à jour : {dt}")
        else:
            st.warning("Modèle non entraîné. Lance `python scripts/train_model.py`.")

    st.divider()

    # Prochaines alertes depuis les paris paper récents
    df_live = get_recent_bets(days=1, paper=True)
    if df_live.empty:
        empty(
            "Aucune alerte en cours. "
            "Le script `scripts/live_predict.py` doit tourner en continu ou via cron."
        )
    else:
        df_live["placed_at"] = pd.to_datetime(df_live["placed_at"])
        df_live["edge_%"] = (df_live["model_edge"] * 100).round(1)
        df_live["prob_%"] = (df_live["predicted_prob"] * 100).round(1)

        for _, row in df_live.iterrows():
            edge_color = "🟢" if row["edge_%"] >= 8 else "🟡" if row["edge_%"] >= 5 else "🔵"
            result_badge = (
                "✅ Gagné" if row["result"] == "win"
                else "❌ Perdu" if row["result"] == "loss"
                else "⏳ En attente"
            )
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
                c1.markdown(f"**{row['player1']} vs {row['player2']}**  \n{row['competition']}")
                c2.markdown(f"Cote  \n**{row['odds']:.2f}**")
                c3.markdown(f"Proba  \n**{row['prob_%']}%**")
                c4.markdown(f"{edge_color} Edge  \n**+{row['edge_%']}%**")
                c5.markdown(f"{result_badge}  \n{row['placed_at'].strftime('%H:%M')}")

    # Instructions cron
    with st.expander("⚙️ Configuration du cron"):
        st.code("""
# Windows Task Scheduler ou dans un terminal :
# Toutes les 15 minutes :
python scripts/live_predict.py

# Ou via Railway / Render (cron job) :
# */15 * * * * cd /app && python scripts/live_predict.py
        """, language="bash")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════

with tab_monitoring:
    st.subheader("Suivi des paris")

    mode = st.radio("Mode", ["Paper trading", "Paris réels"], horizontal=True)
    is_paper = mode == "Paper trading"

    df_mon = get_rolling_roi()
    df_recent = get_recent_bets(days=30, paper=is_paper)

    if df_mon.empty:
        empty("Aucun pari enregistré.")
    else:
        df_mon_filtered = df_mon[df_mon["is_paper"] == (1 if is_paper else 0)]
        if df_mon_filtered.empty:
            empty(f"Aucun pari en mode {'paper' if is_paper else 'réel'}.")
        else:
            # KPIs récents
            n_total = len(df_mon_filtered)
            cum_roi = df_mon_filtered["cumulative_roi"].iloc[-1]
            peak_val = df_mon_filtered["cumulative_pl"].cummax().iloc[-1]
            current_pl = df_mon_filtered["cumulative_pl"].iloc[-1]
            dd = (current_pl - peak_val) / (100 + peak_val) * 100 if peak_val > 0 else 0

            STOP_LOSS = -10.0
            stop_triggered = cum_roi < STOP_LOSS

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Paris total", f"{n_total:,}")
            s2.metric("ROI cumulatif", f"{cum_roi:.1f}%")
            s3.metric("P&L total", f"{current_pl:.1f}€")
            if stop_triggered:
                s4.error(f"🛑 STOP-LOSS déclenché ({cum_roi:.1f}%)")
            else:
                s4.metric(
                    "Stop-loss",
                    f"{STOP_LOSS}% seuil",
                    delta=f"{cum_roi - STOP_LOSS:.1f}% de marge",
                )

            col_roi, col_win = st.columns(2)

            with col_roi:
                st.markdown("#### ROI glissant (50 paris)")
                if "rolling_roi" in df_mon_filtered.columns:
                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(
                        x=df_mon_filtered["placed_at"] if "placed_at" in df_mon_filtered.columns
                          else list(range(len(df_mon_filtered))),
                        y=df_mon_filtered["rolling_roi"],
                        mode="lines",
                        line=dict(color="#4c9be8", width=2),
                        name="ROI glissant",
                    ))
                    fig_roll.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_roll.add_hline(
                        y=STOP_LOSS, line_dash="dot", line_color="#e74c3c",
                        opacity=0.6, annotation_text="Stop-loss",
                    )
                    fig_roll.update_layout(
                        xaxis_title="", yaxis_title="ROI (%)", height=300, showlegend=False,
                    )
                    st.plotly_chart(fig_roll, use_container_width=True)

            with col_win:
                st.markdown("#### Win rate par tranche de cotes")
                if not df_recent.empty:
                    df_recent["odds_bucket"] = pd.cut(
                        df_recent["odds"],
                        bins=[1.0, 1.5, 1.7, 1.9, 2.1, 2.5, 3.5, 10.0],
                        labels=["<1.5", "1.5-1.7", "1.7-1.9", "1.9-2.1", "2.1-2.5", "2.5-3.5", ">3.5"],
                    )
                    wr = (
                        df_recent[df_recent["result"].isin(["win", "loss"])]
                        .groupby("odds_bucket", observed=True)
                        .apply(lambda x: (x["result"] == "win").mean() * 100)
                        .reset_index(name="win_rate_%")
                    )
                    if not wr.empty:
                        fig_wr = px.bar(
                            wr, x="odds_bucket", y="win_rate_%",
                            labels={"odds_bucket": "Tranche de cotes", "win_rate_%": "Win rate (%)"},
                            color="win_rate_%",
                            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                            range_color=[40, 65],
                            height=300,
                        )
                        fig_wr.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
                        fig_wr.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_wr, use_container_width=True)
                    else:
                        empty("Pas assez de résultats.")
                else:
                    empty("Aucun pari récent (30 jours).")

        # Historique récent
        st.markdown("#### Historique des 30 derniers jours")
        if df_recent.empty:
            empty()
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
