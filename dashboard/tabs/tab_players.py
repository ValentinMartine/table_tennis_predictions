import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.queries import *

def render_tab_players(fmt_date):
    player_sub = st.tabs(["Top joueurs", "Profil", "Head-to-Head"])

    with player_sub[0]:
        st.subheader("Classement des joueurs")
        st.caption("Classé par nombre de matchs joués dans les compétitions actives.")

        all_countries = get_player_countries()
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            gender_opt = st.selectbox("Sexe", ["Tous", "Hommes (M)", "Femmes (F)"], key="top_gender")
        with f2:
            min_m = st.slider("Matchs minimum", 5, 100, 10, step=5, key="top_min_matches")
        with f3:
            top_n = st.slider("Nombre de joueurs", 10, 100, 30, step=5, key="top_n")
        with f4:
            _today = pd.Timestamp.today().date().isoformat()
            _period_map = {
                "2025–2026": ("2025-01-01", _today),
                "2024": ("2024-01-01", "2024-12-31"),
                "2023": ("2023-01-01", "2023-12-31"),
                "Tout": (None, None),
            }
            period_opt = st.selectbox("Période", list(_period_map.keys()), index=0, key="top_period")
            date_from, date_to = _period_map[period_opt]

        selected_countries = st.multiselect("Filtrer par pays", all_countries, default=[], key="top_countries", placeholder="Tous les pays")

        gender_filter = "M" if gender_opt == "Hommes (M)" else ("F" if gender_opt == "Femmes (F)" else None)

        df_players = get_top_players(
            limit=top_n, min_matches=min_m, gender=gender_filter,
            countries=selected_countries if selected_countries else None,
            priority_max=98, date_from=date_from, date_to=date_to,
        )

        if df_players.empty:
            st.info("Aucun joueur trouvé avec ces critères.")
        else:
            if "date_of_birth" in df_players.columns:
                df_players["date_of_birth"] = pd.to_datetime(df_players["date_of_birth"], errors="coerce")
                today_ts = pd.Timestamp.today()
                df_players["age"] = df_players["date_of_birth"].apply(lambda d: int((today_ts - d).days / 365.25) if pd.notna(d) else None)
            else:
                df_players["age"] = None

            df_plot = df_players.sort_values("win_rate_pct", ascending=True).copy()
            df_plot["label"] = df_plot.apply(lambda r: f"{r['win_rate_pct']}%  ({int(r['matches_played'])} matchs)", axis=1)
            
            def _flag(code):
                if not code or len(str(code)) != 2: return ""
                return chr(0x1F1E6 + ord(str(code).upper()[0]) - 65) + chr(0x1F1E6 + ord(str(code).upper()[1]) - 65)

            df_plot["display_name"] = df_plot.apply(
                lambda r: f"#{int(r['ittf_rank'])}  {_flag(r.get('country'))}  {r['name']}" if pd.notna(r.get("ittf_rank")) else f"—   {_flag(r.get('country'))}  {r['name']}",
                axis=1
            )
            color_map = {"M": "#4c9be8", "F": "#e84c9b"}
            df_plot["bar_color"] = df_plot["gender"].map(color_map).fillna("#aaaaaa")

            fig3 = go.Figure(go.Bar(
                x=df_plot["win_rate_pct"], y=df_plot["display_name"], orientation="h",
                marker_color=df_plot["bar_color"], text=df_plot["label"], textposition="outside",
                customdata=df_plot[["gender", "country", "age", "matches_played"]].values,
                hovertemplate="<b>%{y}</b><br>Win rate : %{x}%<br>Sexe : %{customdata[0]}<br>Pays : %{customdata[1]}<br>Âge : %{customdata[2]}<br>Matchs : %{customdata[3]}<extra></extra>",
            ))
            fig3.update_layout(xaxis_title="Win rate (%)", xaxis_range=[0, 130], yaxis_title="", height=max(500, len(df_plot) * 32), margin={"r": 10, "l": 10})
            fig3.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig3, use_container_width=True)

            with st.expander("Tableau détaillé"):
                cols_show = [c for c in ["name", "country", "gender", "age", "matches_played", "win_rate_pct"] if c in df_players.columns]
                st.dataframe(df_players[cols_show], use_container_width=True, hide_index=True)

    with player_sub[1]:
        st.subheader("Profil d'un joueur")
        all_names = get_player_names(limit=500)
        if not all_names:
            st.info("Aucun joueur en base de données.")
        else:
            selected_player = st.selectbox("Rechercher un joueur", [""] + all_names, key="profile_player")
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

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    df_elo = get_player_elo_history(selected_player)
                    if not df_elo.empty:
                        st.markdown("#### Évolution Elo")
                        fig_elo = go.Figure()
                        fig_elo.add_trace(go.Scatter(x=df_elo["played_at"], y=df_elo["elo_rating"], mode="lines", fill="tozeroy"))
                        fig_elo.add_hline(y=1500, line_dash="dash", opacity=0.4)
                        st.plotly_chart(fig_elo, use_container_width=True)

                with chart_col2:
                    df_hist_all = get_player_match_history(selected_player, limit=500)
                    if not df_hist_all.empty:
                        st.markdown("#### Stats supplémentaires")

                df_hist = get_player_match_history(selected_player, limit=50)
                if not df_hist.empty:
                    df_hist["played_at"] = pd.to_datetime(df_hist["played_at"], errors="coerce")
                    st.dataframe(df_hist.head(20), use_container_width=True, hide_index=True)

    with player_sub[2]:
        st.subheader("Head-to-Head")
        all_names_h2h = get_player_names(limit=500)
        p1 = st.selectbox("Joueur 1 ", [""] + all_names_h2h, key="h2p1")
        p2 = st.selectbox("Joueur 2 ", [""] + all_names_h2h, key="h2p2")
        if p1 and p2 and p1 != p2:
            summary = get_h2h_summary(p1, p2)
            st.write(f"Vitoires {p1} : {summary['p1_wins']} / Victoires {p2} : {summary['p2_wins']}")
