"""
Modèles de baseline : Elo pur et Bradley-Terry.

Ces modèles ne nécessitent pas d'entraînement ML et servent de
référence pour évaluer les gains apportés par les modèles ML.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def elo_baseline_predictions(df: pd.DataFrame) -> pd.Series:
    """Utilise elo_win_prob_p1 comme prédiction directe."""
    return df["elo_win_prob_p1"].clip(0.01, 0.99)


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    name: str = "model",
) -> dict:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob).clip(1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "model": name,
        "accuracy":       round(float((y_pred == y_true).mean()), 4),
        "f1":             round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "f1_macro":       round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision":      round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":         round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "mcc":            round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "log_loss":       round(float(log_loss(y_true, y_prob)), 4),
        "brier_score":    round(float(brier_score_loss(y_true, y_prob)), 4),
        "roc_auc":        round(float(roc_auc_score(y_true, y_prob)), 4),
        "n_samples":      int(len(y_true)),
    }
