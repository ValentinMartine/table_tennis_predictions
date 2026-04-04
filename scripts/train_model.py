"""
Script : entraînement et évaluation des modèles.

Usage :
    python scripts/train_model.py
    python scripts/train_model.py --model lgbm
    python scripts/train_model.py --model xgb --shap
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.calibration import calibration_curve

from src.features.pipeline import build_features
from src.models.baseline import elo_baseline_predictions, evaluate_predictions
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.models.ensemble_model import EnsembleModel


def save_dashboard_artifacts(stats: dict, elo_stats: dict, preds: np.ndarray,
                              y_true: pd.Series, model, model_name: str,
                              df_test: pd.DataFrame, do_shap: bool) -> None:
    """Sauvegarde les métriques et données pour le dashboard."""
    Path("data").mkdir(exist_ok=True)

    # Métriques globales (dernier modèle entraîné)
    metrics = {**stats, "elo_accuracy": elo_stats["accuracy"],
               "elo_log_loss": elo_stats["log_loss"]}
    Path("data/model_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Métriques par modèle (pour comparaison multi-modèles)
    Path(f"data/{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Métriques baseline Elo (partagées entre modèles)
    Path("data/elo_baseline_metrics.json").write_text(json.dumps(elo_stats, indent=2))

    # Courbe de calibration
    prob_true, prob_pred = calibration_curve(y_true, preds, n_bins=10)
    pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true}).to_csv(
        f"data/{model_name}_calibration.csv", index=False)
    # Alias pour rétrocompatibilité dashboard
    pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true}).to_csv(
        "data/calibration_data.csv", index=False)

    # SHAP (LGBM et XGB)
    if do_shap and isinstance(model, (LGBMModel, XGBModel)):
        importance = model.shap_analysis(df_test)
        importance.to_csv(f"data/{model_name}_shap_importance.csv", index=False)
        importance.to_csv("data/shap_importance.csv", index=False)
        logger.info("\nTop 10 features (SHAP):")
        logger.info(importance.head(10).to_string(index=False))

    logger.info("Artefacts dashboard sauvegardés dans data/")


def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne les modèles TT")
    parser.add_argument("--model", choices=["lgbm", "xgb", "all", "ensemble"], default="lgbm")
    parser.add_argument("--shap", action="store_true", help="Affiche l'analyse SHAP")
    parser.add_argument("--calib-on-val", action="store_true",
                        help="Calibre les probabilités sur le val set (cv=prefit)")
    parser.add_argument("--config", default="config/settings.yaml")
    return parser.parse_args()


def temporal_split(df: pd.DataFrame, config: dict):
    """
    Split 3-way temporel :
      Train : played_at < val_cutoff_date   (ex: < 2024-01-01)
      Val   : val_cutoff_date <= played_at < train_cutoff_date  (ex: 2024)
      Test  : played_at >= train_cutoff_date  (ex: >= 2025-01-01)
    """
    val_cutoff   = pd.Timestamp(config["models"]["val_cutoff_date"])
    test_cutoff  = pd.Timestamp(config["models"]["train_cutoff_date"])

    df_train = df[df["played_at"] < val_cutoff].copy()
    df_val   = df[(df["played_at"] >= val_cutoff) & (df["played_at"] < test_cutoff)].copy()
    df_test  = df[df["played_at"] >= test_cutoff].copy()

    logger.info(
        f"Split temporel - "
        f"Train : {len(df_train)} (< {val_cutoff.date()}) | "
        f"Val : {len(df_val)} ({val_cutoff.date()} – {test_cutoff.date()}) | "
        f"Test : {len(df_test)} (>= {test_cutoff.date()})"
    )
    return df_train, df_val, df_test


def save_elo_history(df: pd.DataFrame) -> None:
    """Sauvegarde l'historique Elo par joueur (pour le dashboard)."""
    from src.database.db import engine as _engine
    try:
        with _engine.connect() as conn:
            players = pd.read_sql("SELECT id, name FROM players", conn)
    except Exception as e:
        logger.warning(f"Impossible de charger les joueurs pour elo_history : {e}")
        return

    p1 = df[["player1_id", "played_at", "elo_p1"]].rename(
        columns={"player1_id": "player_id", "elo_p1": "elo_rating"}
    )
    p2 = df[["player2_id", "played_at", "elo_p2"]].rename(
        columns={"player2_id": "player_id", "elo_p2": "elo_rating"}
    )
    elo_long = pd.concat([p1, p2]).sort_values(["player_id", "played_at"])
    elo_long["month"] = pd.to_datetime(elo_long["played_at"]).dt.to_period("M")
    # Garder le dernier Elo du mois par joueur (réduit la taille)
    elo_monthly = (
        elo_long.groupby(["player_id", "month"])
        .last()
        .reset_index()
    )
    elo_monthly["played_at"] = elo_monthly["played_at"].astype(str)
    elo_monthly = elo_monthly.merge(players, left_on="player_id", right_on="id", how="left")
    elo_monthly[["player_id", "name", "played_at", "elo_rating"]].to_csv(
        "data/elo_history.csv", index=False
    )
    logger.info(f"Historique Elo sauvegardé : {len(elo_monthly)} lignes, {elo_monthly['player_id'].nunique()} joueurs")


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("Construction des features...")
    df = build_features(args.config)
    if df.empty:
        logger.error("Dataset vide - lancez d'abord run_scraper.py")
        return

    odds_coverage = df["has_odds"].mean() if "has_odds" in df.columns else 0.0
    odds_count = int(df["has_odds"].sum()) if "has_odds" in df.columns else 0
    logger.info(f"Couverture odds bookmaker : {odds_count}/{len(df)} matchs ({odds_coverage:.1%})")
    if odds_coverage < 0.05:
        logger.warning("< 5% des matchs ont des cotes — implied_prob_p1 aura peu d'impact. "
                       "Lancez store_pre_match_odds.py régulièrement pour accumuler des données.")

    logger.info("Sauvegarde de l'historique Elo...")
    save_elo_history(df)

    df_train, df_val, df_test = temporal_split(df, config)

    # Baseline Elo sur val et test
    elo_preds_val  = elo_baseline_predictions(df_val)
    elo_preds_test = elo_baseline_predictions(df_test)
    elo_stats_val  = evaluate_predictions(df_val["target"],  elo_preds_val,  "Elo Baseline (val)")
    elo_stats_test = evaluate_predictions(df_test["target"], elo_preds_test, "Elo Baseline (test)")
    logger.info(f"Baseline Elo val  : {elo_stats_val}")
    logger.info(f"Baseline Elo test : {elo_stats_test}")

    models_to_train = ["lgbm", "xgb"] if args.model == "all" else [args.model]

    for model_name in models_to_train:
        logger.info(f"\n=== {model_name.upper()} ===")
        
        # Chargement automatique des paramètres optimisés si dispos
        params = None
        params_path = Path(f"data/best_params_{model_name}.json")
        if params_path.exists():
            logger.info(f"Chargement des paramètres optimisés depuis {params_path}")
            params = json.loads(params_path.read_text())
            
        if model_name == "ensemble":
            model = EnsembleModel()
        elif model_name == "lgbm":
            model = LGBMModel(args.config, params=params)
        else:
            model = XGBModel(args.config, params=params)
            
        model.fit(df_train, df_val=df_val if args.calib_on_val else None)

        preds_val  = model.predict_proba(df_val)
        preds_test = model.predict_proba(df_test)

        stats_val  = evaluate_predictions(df_val["target"],  preds_val,  f"{model_name} (val)")
        stats_test = evaluate_predictions(df_test["target"], preds_test, f"{model_name} (test)")
        logger.info(f"Val  : {stats_val}")
        logger.info(f"Test : {stats_test}")
        logger.info(f"Amélioration log_loss vs Elo - val: {elo_stats_val['log_loss'] - stats_val['log_loss']:.4f} | test: {elo_stats_test['log_loss'] - stats_test['log_loss']:.4f}")

        model_path = f"data/{model_name}_model.pkl"
        model.save(model_path)

        # Dashboard : métriques test (référence principale) + val pour comparaison
        metrics_combined = {
            **{f"test_{k}": v for k, v in stats_test.items()},
            **{f"val_{k}": v for k, v in stats_val.items()},
            # Alias rétrocompatibilité (dashboard attend ces clés sans préfixe)
            **stats_test,
            "elo_accuracy": elo_stats_test["accuracy"],
            "elo_log_loss": elo_stats_test["log_loss"],
        }
        save_dashboard_artifacts(
            metrics_combined, elo_stats_test, preds_test, df_test["target"],
            model, model_name, df_test, args.shap,
        )


if __name__ == "__main__":
    main()
