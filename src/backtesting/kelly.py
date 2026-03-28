"""
Kelly criterion et gestion du bankroll.
"""


def kelly_fraction(prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Calcule la fraction du bankroll à miser.

    Args:
        prob   : probabilité estimée de victoire
        odds   : cote décimale (ex: 1.90)
        fraction : diviseur du Kelly (0.25 = quart Kelly recommandé)

    Returns:
        Fraction du bankroll [0, max_stake]
    """
    b = odds - 1  # profit net si victoire
    q = 1 - prob
    kelly = (b * prob - q) / b
    return max(0.0, kelly * fraction)


def compute_stake(
    bankroll: float,
    prob: float,
    odds: float,
    kelly_fraction_val: float = 0.25,
    max_stake_pct: float = 0.02,
) -> float:
    """
    Calcule le montant à miser en euros.

    Args:
        bankroll       : bankroll actuel
        prob           : probabilité estimée
        odds           : cote décimale
        kelly_fraction_val : fraction Kelly
        max_stake_pct  : plafond en % du bankroll

    Returns:
        Montant en euros (0 si edge négatif)
    """
    f = kelly_fraction(prob, odds, fraction=kelly_fraction_val)
    stake = bankroll * f
    max_stake = bankroll * max_stake_pct
    return min(stake, max_stake)


def model_edge(prob: float, odds: float) -> float:
    """Edge = probabilité estimée - probabilité implicite des cotes."""
    implied = 1 / odds
    return prob - implied
