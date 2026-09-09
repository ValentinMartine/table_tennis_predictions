"""
Analyse la politique de mise sélective.

Produit deux artefacts web :
- data/betting_thresholds.json : compromis seuil de confiance / ROI / nombre de paris.
- data/walk_forward_backtest.json : choix du seuil sur le passé, test sur la période suivante.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from scripts.train_model import temporal_split
from src.backtesting.simulator import BettingSimulator
from src.features.pipeline import build_features
from src.models.lgbm_model import LGBMModel


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _odds_subset(df: pd.DataFrame) -> pd.DataFrame:
    odds_p1 = pd.to_numeric(df["odds_p1"], errors="coerce")
    odds_p2 = pd.to_numeric(df["odds_p2"], errors="coerce")
    mask = odds_p1.notna() & odds_p2.notna() & (odds_p1 > 1) & (odds_p2 > 1)
    return df.loc[mask].sort_values("played_at").copy()


def _simulate(df: pd.DataFrame, confidence: float, config_path: str, bankroll: float = 1000.0):
    sim = BettingSimulator(config_path)
    sim.min_confidence = confidence
    bets, stats = sim.run(df, initial_bankroll=bankroll)
    return bets, stats


def _threshold_curve(df: pd.DataFrame, config_path: str, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        bets, stats = _simulate(df, threshold, config_path)
        rows.append({
            "threshold": round(threshold, 2),
            "n_bets": int(stats.get("n_bets", 0)),
            "roi_pct": stats.get("roi_pct"),
            "final_bankroll": stats.get("final_bankroll"),
            "max_drawdown_pct": stats.get("max_drawdown_pct"),
            "sharpe_ratio": stats.get("sharpe_ratio"),
            "win_rate_pct": stats.get("win_rate_pct"),
        })
    return rows


def _best_threshold(tune_df: pd.DataFrame, config_path: str, thresholds: list[float], min_bets: int):
    candidates = []
    for threshold in thresholds:
        _, stats = _simulate(tune_df, threshold, config_path)
        if stats.get("n_bets", 0) >= min_bets:
            candidates.append((stats["roi_pct"], threshold, stats))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])


def _walk_forward(
    df: pd.DataFrame,
    config_path: str,
    thresholds: list[float],
    min_tune_bets: int,
    test_freq: str,
) -> dict:
    if df.empty:
        return {"folds": [], "summary": {}}

    df = df.sort_values("played_at").copy()
    periods = sorted(df["played_at"].dt.to_period(test_freq).unique())
    folds = []
    bankroll = 1000.0

    for period in periods[1:]:
        test_start = period.start_time
        test_end = period.end_time
        tune_df = df[df["played_at"] < test_start]
        test_df = df[(df["played_at"] >= test_start) & (df["played_at"] <= test_end)]
        if test_df.empty:
            continue

        best = _best_threshold(tune_df, config_path, thresholds, min_tune_bets)
        if best is None:
            continue

        _, threshold, tune_stats = best
        bets, test_stats = _simulate(test_df, threshold, config_path, bankroll)
        if not test_stats:
            folds.append({
                "period": str(period),
                "threshold": round(threshold, 2),
                "tune_n_bets": int(tune_stats["n_bets"]),
                "tune_roi_pct": tune_stats["roi_pct"],
                "test_n_bets": 0,
                "test_roi_pct": None,
            })
            continue

        bankroll = float(bets["bankroll"].iloc[-1])
        folds.append({
            "period": str(period),
            "test_start": test_start.date().isoformat(),
            "test_end": test_end.date().isoformat(),
            "threshold": round(threshold, 2),
            "tune_n_bets": int(tune_stats["n_bets"]),
            "tune_roi_pct": tune_stats["roi_pct"],
            "test_n_bets": int(test_stats["n_bets"]),
            "test_roi_pct": test_stats["roi_pct"],
            "test_final_bankroll": test_stats["final_bankroll"],
            "rolling_bankroll": round(bankroll, 2),
        })

    total_bets = sum(f.get("test_n_bets", 0) for f in folds)
    summary = {
        "folds": len(folds),
        "n_bets": int(total_bets),
        "final_bankroll": round(bankroll, 2),
        "roi_pct": round((bankroll - 1000.0) / 1000.0 * 100, 2),
        "avg_threshold": round(float(np.mean([f["threshold"] for f in folds])), 3) if folds else None,
        "test_freq": test_freq,
    }
    return {"folds": folds, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--model-path", default="data/lgbm_model.pkl")
    parser.add_argument("--min-tune-bets", type=int, default=20)
    parser.add_argument("--test-freq", default="Q", help="Période pandas: M, Q, 2Q...")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    thresholds = [round(x, 2) for x in np.arange(0.50, 0.81, 0.05)]
    df = build_features(args.config)
    _, df_val, df_test = temporal_split(df, config)

    model = LGBMModel.load(args.model_path)
    eval_df = pd.concat([df_val, df_test]).sort_values("played_at").copy()
    eval_df["pred_prob_p1"] = model.predict_proba(eval_df)
    odds_eval = _odds_subset(eval_df)
    odds_test = _odds_subset(df_test.copy())
    if not odds_test.empty:
        odds_test["pred_prob_p1"] = model.predict_proba(odds_test)

    coverage = {
        "val_with_odds": int(_odds_subset(df_val).shape[0]),
        "test_with_odds": int(_odds_subset(df_test).shape[0]),
        "eval_with_odds": int(odds_eval.shape[0]),
        "eval_start": odds_eval["played_at"].min().date().isoformat() if not odds_eval.empty else None,
        "eval_end": odds_eval["played_at"].max().date().isoformat() if not odds_eval.empty else None,
    }

    threshold_payload = {
        "thresholds": _threshold_curve(odds_test, args.config, thresholds),
        "coverage": coverage,
        "basis": "test_real_odds",
    }
    walk_payload = _walk_forward(
        odds_eval,
        args.config,
        thresholds,
        min_tune_bets=args.min_tune_bets,
        test_freq=args.test_freq,
    )

    (DATA / "betting_thresholds.json").write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DATA / "walk_forward_backtest.json").write_text(
        json.dumps(walk_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Seuils écrits : {DATA / 'betting_thresholds.json'}")
    logger.info(f"Walk-forward écrit : {DATA / 'walk_forward_backtest.json'}")
    logger.info(f"Couverture cotes : {coverage}")
    logger.info(f"Walk-forward summary : {walk_payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
