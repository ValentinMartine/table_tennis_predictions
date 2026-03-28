"""
Utilitaires pour la gestion des cotes en backtesting.
"""
import numpy as np
import pandas as pd


def fill_synthetic_odds(df: pd.DataFrame, vig: float = 0.05) -> pd.DataFrame:
    """
    Remplace les cotes NULL par des cotes synthétiques calculées depuis
    la probabilité Elo + un overround (vig).

    Formule :
        odds_p1 = (1 / elo_win_prob_p1) * (1 + vig)
        odds_p2 = (1 / (1 - elo_win_prob_p1)) * (1 + vig)

    Args:
        df  : DataFrame contenant 'odds_p1', 'odds_p2', 'elo_win_prob_p1'
        vig : overround simulé (défaut 5%)

    Returns:
        DataFrame avec les colonnes odds_p1/odds_p2 remplies.
    """
    df = df.copy()

    if "elo_win_prob_p1" not in df.columns:
        return df

    mask = df["odds_p1"].isna() | df["odds_p2"].isna()
    if not mask.any():
        return df

    elo = df.loc[mask, "elo_win_prob_p1"].clip(0.01, 0.99)
    df.loc[mask, "odds_p1"] = ((1 / elo) * (1 + vig)).round(3)
    df.loc[mask, "odds_p2"] = ((1 / (1 - elo)) * (1 + vig)).round(3)
    if "odds_source" in df.columns:
        df["odds_source"] = df["odds_source"].fillna("synthetic_elo")
    else:
        df["odds_source"] = "synthetic_elo"
        df.loc[~mask, "odds_source"] = "real"

    return df
