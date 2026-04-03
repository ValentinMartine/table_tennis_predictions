import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dashboard.queries import *

def render_tab_data(WTT_CALENDAR_2025_2026, PRIORITY_LABELS, PRIORITY_COLORS, fmt_date):
    explore_sub = st.tabs(["Calendrier", "Historique", "Backtest", "Suivi des paris"])

    with explore_sub[0]:
        
        # Récupération dynamique depuis la DB pour les tournois passés/importés
        df_status = get_competition_status()
        records = []
        if not df_status.empty:
            df_active = df_status[(df_status["priority"] < 99) & (df_status["first_match"].notnull())].copy()
            for _, r in df_active.iterrows():
                records.append({
                    "Tournoi": r["competition"],
                    "Début": pd.to_datetime(r["first_match"]).date(),
                    "Fin": pd.to_datetime(r["last_match"]).date(),
                    "Type": r["type"],
                    "Lieu": "—"
                })
        
        # Ajout du calendrier prévisionnel hardcodé
        for c in WTT_CALENDAR_2025_2026:
            year_str = str(c["debut"])[:4]
            records.append({
                "Tournoi": f"{c['tournoi']} {year_str}",
                "Début": pd.to_datetime(c["debut"]).date(),
                "Fin": pd.to_datetime(c["fin"]).date(),
                "Type": c.get("type", "WTT"),
                "Lieu": c.get("lieu", "—")
            })
            
        df_cal = pd.DataFrame(records)
        if not df_cal.empty:
            # Enlever les doublons (la DB a priorité)
            df_cal = df_cal.drop_duplicates(subset=["Tournoi"], keep="first")
            
            today = pd.Timestamp.today().date()
            df_cal["Statut"] = df_cal.apply(
                lambda r: "En cours" if (pd.notnull(r["Début"]) and pd.notnull(r["Fin"]) and r["Début"] <= today <= r["Fin"]) 
                          else ("À venir" if pd.notnull(r["Début"]) and r["Début"] > today else "Passé"),
                axis=1
            )
            
            # Trier par date avant le formattage en string pour que ce soit un tri chronologique
            df_cal = df_cal.sort_values("Début", ascending=False)
            
            # Formater les dates pour l'affichage (évite les bugs Streamlit de timezone)
            df_cal["Début"] = pd.to_datetime(df_cal["Début"]).dt.strftime('%d/%m/%Y')
            df_cal["Fin"] = pd.to_datetime(df_cal["Fin"]).dt.strftime('%d/%m/%Y')
            
            st.dataframe(df_cal, use_container_width=True, hide_index=True)

            st.markdown("### Détails d'un tournoi")
            selected_tourney = st.selectbox("Sélectionne un tournoi (Édition) pour voir ses matchs passés", df_cal["Tournoi"].tolist())
            if selected_tourney:
                # On essaie d'extraire le nom et l'année de manière plus robuste
                parts = selected_tourney.strip().rsplit(' ', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    t_name, t_year = parts[0].strip(), parts[1].strip()
                    df_matches = get_competition_matches(t_name, t_year)
                    if not df_matches.empty:
                        df_matches["Date"] = pd.to_datetime(df_matches["played_at"]).dt.strftime('%d/%m/%Y %H:%M')
                        display_m = df_matches[["Date", "p1_name", "p2_name", "score_p1", "score_p2", "status"]]
                        display_m.columns = ["Date", "Joueur 1", "Joueur 2", "Score J1", "Score J2", "Statut"]
                        st.dataframe(display_m, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucun match joué ou enregistré en base de données pour ce tournoi.")
                else:
                    st.info("Les matchs de ce tournoi prévisionnel ne sont pas encore disponibles dans notre base.")

        else:
            st.info("Aucun tournoi disponible.")

    with explore_sub[1]:
        st.subheader("Historique des données")
        df_status = get_competition_status()
        if not df_status.empty:
            df_status["last_match"] = pd.to_datetime(df_status["last_match"], errors="coerce")
            df_active = df_status[df_status["priority"] < 99]
            if not df_active.empty:
                st.dataframe(df_active[["competition", "total_matches", "last_match"]], use_container_width=True, hide_index=True)

        df_time = get_matches_over_time(include_archived=False)
        if not df_time.empty:
            fig = px.bar(df_time, x="month", y="matches", color="competition", title="Matchs par mois")
            st.plotly_chart(fig, use_container_width=True)

    with explore_sub[2]:
        st.subheader("Backtest — simulation de paris")
        df_bets = get_betting_history(paper_only=True)
        if df_bets.empty:
            st.info("Aucun historique de paris disponible.")
        else:
            df_bets["placed_at"] = pd.to_datetime(df_bets["placed_at"], errors="coerce").sort_values()
            df_bets["cumulative_pl"] = df_bets["profit_loss"].cumsum()
            st.line_chart(df_bets.set_index("placed_at")["cumulative_pl"])

    with explore_sub[3]:
        st.subheader("Suivi des paris")
        df_mon = get_rolling_roi()
        if not df_mon.empty:
            st.line_chart(df_mon.set_index("placed_at")["cumulative_roi"] if "placed_at" in df_mon.columns else df_mon["cumulative_roi"])
