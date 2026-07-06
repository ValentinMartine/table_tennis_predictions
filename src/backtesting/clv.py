"""
Closing Line Value (CLV) — mesure de la qualité prédictive d'une stratégie de pari.

La cote de clôture (dernier prix avant le match) est l'estimation la plus efficiente
de la vraie probabilité. Battre régulièrement la clôture est, dans la littérature du
pari sportif, le meilleur prédicteur de rentabilité long terme — bien avant le taux
de réussite (qui est bruité à court terme).

Deux mesures complémentaires :
- CLV en cote  : bet_odds / closing_odds - 1  (>0 = on a obtenu un meilleur prix
  que la clôture).
- CLV en proba : proba implicite dévigorée à la clôture - proba implicite au pari
  (>0 = le marché a bougé vers notre sélection après notre pari).
"""
import numpy as np
import pandas as pd


def no_vig_prob(odds_side: float, odds_other: float) -> float:
    """Probabilité implicite dévigorée (marge du book retirée) pour un côté."""
    if odds_side <= 1 or odds_other <= 1:
        return float("nan")
    r1, r2 = 1.0 / odds_side, 1.0 / odds_other
    return r1 / (r1 + r2)


def clv_odds(bet_odds: float, closing_odds: float) -> float:
    """CLV en cote : >0 si on a parié à une cote plus haute que la clôture."""
    if bet_odds <= 1 or closing_odds <= 1:
        return float("nan")
    return bet_odds / closing_odds - 1.0


def clv_prob(
    bet_side_odds: float, bet_other_odds: float,
    close_side_odds: float, close_other_odds: float,
) -> float:
    """
    CLV en probabilité dévigorée : (proba clôture du côté parié) - (proba au pari).
    Positif = le marché a convergé vers notre sélection après le pari.
    """
    p_bet = no_vig_prob(bet_side_odds, bet_other_odds)
    p_close = no_vig_prob(close_side_odds, close_other_odds)
    return p_close - p_bet


def clv_report(bets: pd.DataFrame) -> dict:
    """
    Agrège la CLV sur un ensemble de paris.

    Colonnes attendues : bet_odds, closing_odds. Optionnelles : profit_loss.
    Ignore les paris sans cote de clôture.
    """
    df = bets.copy()
    df = df[df.get("closing_odds").notna()] if "closing_odds" in df else df.iloc[0:0]
    if df.empty:
        return {"n_bets": 0, "n_with_closing": 0}

    clv = df.apply(lambda r: clv_odds(r["bet_odds"], r["closing_odds"]), axis=1)
    clv = clv.replace([np.inf, -np.inf], np.nan).dropna()
    if clv.empty:
        return {"n_bets": int(len(bets)), "n_with_closing": 0}

    report = {
        "n_bets": int(len(bets)),
        "n_with_closing": int(len(clv)),
        "mean_clv": round(float(clv.mean()), 4),
        "median_clv": round(float(clv.median()), 4),
        "pct_positive_clv": round(float((clv > 0).mean()), 4),
        "clv_std": round(float(clv.std(ddof=0)), 4),
    }
    # Signal fort : corrélation entre CLV et profit réalisé (doit être positive)
    if "profit_loss" in df.columns and df["profit_loss"].notna().any():
        aligned = pd.DataFrame({"clv": clv, "pnl": df.loc[clv.index, "profit_loss"]}).dropna()
        if len(aligned) >= 3 and aligned["clv"].std() > 0 and aligned["pnl"].std() > 0:
            report["corr_clv_pnl"] = round(float(aligned["clv"].corr(aligned["pnl"])), 4)
    return report
