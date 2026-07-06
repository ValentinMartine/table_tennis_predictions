"""
Diagnostics de calibration : une proba de 0,70 doit correspondre à 70 % de
victoires observées. Essentiel pour le value betting (un edge mal calibré fausse
le critère de Kelly).

Fournit :
- reliability bins (proba prédite vs fréquence observée, par bin),
- ECE / MCE (Expected / Maximum Calibration Error),
- décomposition de Brier (Murphy) : Brier = Reliability - Resolution + Uncertainty.
"""
import numpy as np


def calibration_report(y_true, y_prob, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float).clip(0.0, 1.0)
    n = len(y_true)
    if n == 0:
        return {"n_samples": 0}

    base_rate = float(y_true.mean())
    brier = float(np.mean((y_prob - y_true) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index dans [0, n_bins-1]
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    bins = []
    reliability = 0.0   # Σ n_k (f_k - o_k)^2 / N
    resolution = 0.0    # Σ n_k (o_k - base)^2 / N
    ece = 0.0
    mce = 0.0
    for k in range(n_bins):
        mask = idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        f_k = float(y_prob[mask].mean())   # confiance moyenne
        o_k = float(y_true[mask].mean())   # fréquence observée
        gap = abs(f_k - o_k)
        reliability += n_k * (f_k - o_k) ** 2
        resolution += n_k * (o_k - base_rate) ** 2
        ece += n_k * gap
        mce = max(mce, gap)
        bins.append({
            "bin": k,
            "p_pred": round(f_k, 4),
            "p_true": round(o_k, 4),
            "count": n_k,
            "gap": round(gap, 4),
        })

    reliability /= n
    resolution /= n
    uncertainty = base_rate * (1.0 - base_rate)
    ece /= n

    return {
        "n_samples": n,
        "base_rate": round(base_rate, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_reliability": round(reliability, 5),   # plus bas = mieux
        "brier_resolution": round(resolution, 5),      # plus haut = mieux
        "brier_uncertainty": round(uncertainty, 5),
        "bins": bins,
    }
