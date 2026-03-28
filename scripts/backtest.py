"""
Script : backtesting out-of-sample avec simulation de paris.

Usage :
    python scripts/backtest.py
    python scripts/backtest.py --model lgbm --bankroll 500
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from loguru import logger

from src.backtesting.simulator import BettingSimulator
from src.features.pipeline import build_features
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--config", default="config/settings.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    df = build_features(args.config)
    if df.empty:
        logger.error("Dataset vide")
        return

    cutoff = pd.Timestamp(config["models"]["train_cutoff_date"])
    df_train = df[df["played_at"] < cutoff]
    df_test = df[df["played_at"] >= cutoff]

    # Chargement ou entraînement du modèle
    model_path = f"data/{args.model}_model.pkl"
    try:
        model = LGBMModel.load(model_path) if args.model == "lgbm" else XGBModel.load(model_path)
        logger.info(f"Modèle chargé depuis {model_path}")
    except FileNotFoundError:
        logger.info("Modèle non trouvé, entraînement...")
        model = LGBMModel(args.config) if args.model == "lgbm" else XGBModel(args.config)
        model.fit(df_train)

    df_test = df_test.copy()
    df_test["pred_prob_p1"] = model.predict_proba(df_test)

    # Simulation
    simulator = BettingSimulator(args.config)
    df_bets, stats = simulator.run(df_test, initial_bankroll=args.bankroll)

    if df_bets.empty:
        logger.warning("Aucun pari qualifié")
        return

    logger.info("\n=== RÉSULTATS BACKTESTING ===")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Critères de succès
    logger.info("\n=== CRITÈRES DE SUCCÈS ===")
    logger.info(f"  ROI >3%    : {'✓' if stats['roi_pct'] > 3 else '✗'} ({stats['roi_pct']}%)")
    logger.info(f"  Sharpe >1.2: {'✓' if stats['sharpe_ratio'] > 1.2 else '✗'} ({stats['sharpe_ratio']})")
    logger.info(f"  Drawdown <15%: {'✓' if stats['max_drawdown_pct'] > -15 else '✗'} ({stats['max_drawdown_pct']}%)")

    if args.plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(df_bets["bankroll"].values)
        plt.title("Évolution du bankroll")
        plt.xlabel("Nb paris")
        plt.ylabel("Bankroll (€)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        cumulative_roi = (df_bets["bankroll"] - args.bankroll) / args.bankroll * 100
        plt.plot(cumulative_roi.values)
        plt.axhline(0, color="red", linestyle="--", alpha=0.5)
        plt.title("ROI cumulatif (%)")
        plt.xlabel("Nb paris")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("data/backtest_results.png", dpi=150)
        logger.info("Graphique sauvegardé : data/backtest_results.png")


if __name__ == "__main__":
    main()
