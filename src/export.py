"""Export des résultats vers un artefact JSON web-ready (data/predictions.json).

Agrège les métriques du modèle, la courbe de calibration, l'importance des
variables (SHAP), le backtest de bankroll et les prédictions à venir, pour la
page de présentation web (valentinmartine.com).

Usage :
    python -m src.export
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _load_json(name: str) -> dict:
    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _pct(value) -> float | None:
    """Convertit '75.4%' ou '0.75' en fraction 0..1."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().replace(",", ".")
    if s in ("", "-", "?"):
        return None
    try:
        v = float(s.replace("%", ""))
        return round(v / 100.0, 4) if "%" in s or v > 1 else round(v, 4)
    except ValueError:
        return None


def _int_or_none(value) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        i = int(float(value))
        return i if i < 9999 else None
    except (ValueError, TypeError):
        return None


def build() -> Path:
    m = _load_json("model_metrics.json")
    elo = _load_json("elo_baseline_metrics.json")

    metrics = {
        "model": "LightGBM",
        "accuracy": m.get("test_accuracy"),
        "roc_auc": m.get("test_roc_auc"),
        "log_loss": m.get("test_log_loss"),
        "brier": m.get("test_brier_score"),
        "n_test": m.get("test_n_samples"),
        "elo_accuracy": m.get("elo_accuracy") or elo.get("test_accuracy"),
        "elo_log_loss": m.get("elo_log_loss") or elo.get("test_log_loss"),
    }

    calibration = []
    cal = DATA / "lgbm_calibration.csv"
    if cal.exists():
        cdf = pd.read_csv(cal)
        calibration = [
            {"pred": round(float(r["prob_pred"]), 4), "observed": round(float(r["prob_true"]), 4)}
            for _, r in cdf.iterrows()
        ]

    feature_importance = []
    shap = DATA / "lgbm_shap_importance.csv"
    if shap.exists():
        sdf = pd.read_csv(shap).head(10)
        feature_importance = [
            {"feature": str(r["feature"]), "value": round(float(r["mean_abs_shap"]), 4)}
            for _, r in sdf.iterrows()
        ]

    upcoming = []
    up = DATA / "upcoming_predictions.csv"
    if up.exists():
        udf = pd.read_csv(up)
        for _, r in udf.iterrows():
            upcoming.append({
                "date": str(r.get("Date", "")),
                "tournament": str(r.get("Tournoi", "")),
                "p1": str(r.get("J1", "")),
                "p2": str(r.get("J2", "")),
                "p1_prob": _pct(r.get("P(J1)")),
                "favorite": str(r.get("Favori", "")),
                "confidence": _pct(r.get("Confiance")),
                "edge": _pct(r.get("Edge")),
                "wtt_p1": _int_or_none(r.get("WTT#J1")),
                "wtt_p2": _int_or_none(r.get("WTT#J2")),
            })

    backtest = _load_json("backtest_bankroll.json") or None
    betting_thresholds = _load_json("betting_thresholds.json") or None
    walk_forward = _load_json("walk_forward_backtest.json") or None

    # Qualité de calibration (ECE/MCE + décomposition de Brier), si disponible
    cr = _load_json("lgbm_calibration_report.json")
    calibration_quality = {
        "ece": cr.get("ece"),
        "mce": cr.get("mce"),
        "brier_reliability": cr.get("brier_reliability"),
        "brier_resolution": cr.get("brier_resolution"),
    } if cr else None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "Ensemble LightGBM + XGBoost et baseline Elo, calibration isotonique",
        "metrics": metrics,
        "calibration": calibration,
        "calibration_quality": calibration_quality,
        "feature_importance": feature_importance,
        "backtest": backtest,
        "betting_thresholds": betting_thresholds,
        "walk_forward": walk_forward,
        "upcoming": upcoming,
    }

    out = DATA / "predictions.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Écrit : {out} ({len(upcoming)} matchs à venir, {len(calibration)} points de calibration)")
    return out


if __name__ == "__main__":
    build()
