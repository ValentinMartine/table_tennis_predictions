import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dashboard.queries import get_all_model_metrics

def render_tab_model():
    model_sub = st.tabs(["Performance", "Comparaison modèles"])

    with model_sub[0]:
        st.subheader("Performance du modèle LGBM")

        model_path = Path("data/lgbm_model.pkl")
        metrics_path = Path("data/lgbm_metrics.json")
        shap_path = Path("data/lgbm_shap_importance.csv")
        calib_path = Path("data/lgbm_calibration.csv")

        if not model_path.exists():
            st.info("Modèle non disponible.")
        else:
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

            col_shap, col_calib = st.columns(2)
            with col_shap:
                st.markdown("#### Importance des features (SHAP)")
                if shap_path.exists():
                    df_shap = pd.read_csv(shap_path)
                    df_shap = df_shap.sort_values("mean_abs_shap", ascending=True).tail(15)
                    fig_shap = px.bar(
                        df_shap, x="mean_abs_shap", y="feature", orientation="h",
                        color="mean_abs_shap", color_continuous_scale=["#4c9be8", "#2ecc71"], height=450,
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("Analyse SHAP non disponible.")

            with col_calib:
                st.markdown("#### Calibration des probabilités")
                if calib_path.exists():
                    df_calib = pd.read_csv(calib_path)
                    fig_calib = go.Figure()
                    fig_calib.add_trace(go.Scatter(x=df_calib["prob_pred"], y=df_calib["prob_true"], mode="lines+markers", name="Modèle"))
                    fig_calib.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite", line=dict(dash="dash")))
                    st.plotly_chart(fig_calib, use_container_width=True)

    with model_sub[1]:
        st.subheader("Comparaison des modèles")
        all_metrics = get_all_model_metrics()

        if not all_metrics:
            st.info("Aucun modèle entraîné.")
        else:
            METRIC_DISPLAY = {"accuracy": "Accuracy", "f1_macro": "F1", "roc_auc": "ROC AUC", "log_loss": "Log Loss", "brier_score": "Brier Score"}
            rows = []
            for model_name, m in all_metrics.items():
                row = {"Modèle": model_name}
                for key, label in METRIC_DISPLAY.items():
                    val = m.get(key)
                    row[label] = round(val, 4) if val is not None else None
                rows.append(row)
            df_compare = pd.DataFrame(rows).set_index("Modèle")
            st.dataframe(df_compare, use_container_width=True)
