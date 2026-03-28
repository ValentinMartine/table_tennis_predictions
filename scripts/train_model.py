"""
Script : entraînement et évaluation des modèles.

Usage :
    python scripts/train_model.py
    python scripts/train_model.py --model lgbm
    python scripts/train_model.py --model xgb --shap
"""
import argparse
import json
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


def save_dashboard_artifacts(stats: dict, elo_stats: dict, preds: np.ndarray,
                              y_true: pd.Series, model, model_name: str,
                              df_test: pd.DataFrame, do_shap: bool) -> None:
    """Sauvegarde les métriques et données pour le dashboard."""
    Path("data").mkdir(exist_ok=True)

    # Métriques
    metrics = {**stats, "elo_accuracy": elo_stats["accuracy"],
               "elo_log_loss": elo_stats["log_loss"]}
    Path("data/model_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Courbe de calibration
    prob_true, prob_pred = calibration_curve(y_true, preds, n_bins=10)
    pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true}).to_csv(
        "data/calibration_data.csv", index=False)

    # SHAP
    if do_shap and model_name == "lgbm" and isinstance(model, LGBMModel):
        importance = model.shap_analysis(df_test)
        importance.to_csv("data/shap_importance.csv", index=False)
        logger.info("\nTop 10 features (SHAP):")
        logger.info(importance.head(10).to_string(index=False))

    logger.info("Artefacts dashboard sauvegardés dans data/")


def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne les modèles TT")
    parser.add_argument("--model", choices=["lgbm", "xgb", "all"], default="lgbm")
    parser.add_argument("--shap", action="store_true", help="Affiche l'analyse SHAP")
    parser.add_argument("--config", default="config/settings.yaml")
    return parser.parse_args()


def temporal_split(df: pd.DataFrame, config: dict):
    cutoff = pd.Timestamp(config["models"]["train_cutoff_date"])
    df_train = df[df["played_at"] < cutoff].copy()
    df_test = df[df["played_at"] >= cutoff].copy()
    logger.info(f"Train : {len(df_train)} | Test : {len(df_test)}")
    return df_train, df_test


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("Construction des features...")
    df = build_features(args.config)
    if df.empty:
        logger.error("Dataset vide — lancez d'abord run_scraper.py")
        return

    df_train, df_test = temporal_split(df, config)

    # Baseline Elo
    elo_preds = elo_baseline_predictions(df_test)
    elo_stats = evaluate_predictions(df_test["target"], elo_preds, "Elo Baseline")
    logger.info(f"Baseline Elo : {elo_stats}")

    models_to_train = ["lgbm", "xgb"] if args.model == "all" else [args.model]

    for model_name in models_to_train:
        logger.info(f"\n=== {model_name.upper()} ===")
        model = LGBMModel(args.config) if model_name == "lgbm" else XGBModel(args.config)
        model.fit(df_train)

        preds = model.predict_proba(df_test)
        stats = evaluate_predictions(df_test["target"], preds, model_name)
        logger.info(stats)

        # Comparaison avec baseline
        improvement = elo_stats["log_loss"] - stats["log_loss"]
        logger.info(f"Amélioration log_loss vs Elo : {improvement:.4f}")

        model_path = f"data/{model_name}_model.pkl"
        model.save(model_path)

        save_dashboard_artifacts(
            stats, elo_stats, preds, df_test["target"],
            model, model_name, df_test, args.shap,
        )


if __name__ == "__main__":
    main()
