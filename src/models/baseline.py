"""
Modèles de baseline : Elo pur et Bradley-Terry.

Ces modèles ne nécessitent pas d'entraînement ML et servent de
référence pour évaluer les gains apportés par les modèles ML.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


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

    accuracy = ((y_prob >= 0.5).astype(int) == y_true).mean()
    return {
        "model": name,
        "accuracy": round(accuracy, 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "n_samples": len(y_true),
    }
