"""
Eval-gate : n'autorise la promotion d'un nouveau modèle que s'il ne régresse pas
par rapport au modèle en production. Empêche un réentraînement autonome de dégrader
silencieusement les prédictions.

Compare les métriques test du candidat aux métriques de référence (production) sur
une métrique primaire (log_loss par défaut, plus bas = mieux), avec tolérance.

Usage :
    python scripts/eval_gate.py --candidate data/lgbm_metrics.json \
                                --baseline  data/production_metrics.json
Codes de sortie : 0 = promotion autorisée, 1 = régression (rejet), 2 = erreur.
"""
import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Métriques où "plus bas = mieux"
LOWER_IS_BETTER = {"log_loss", "brier_score", "test_log_loss", "test_brier_score"}


def _load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _get(metrics: dict, key: str) -> float | None:
    for k in (key, f"test_{key}", key.replace("test_", "")):
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="data/lgbm_metrics.json")
    ap.add_argument("--baseline", default="data/production_metrics.json")
    ap.add_argument("--metric", default="log_loss",
                    help="Métrique primaire de décision (log_loss, brier_score, accuracy…)")
    ap.add_argument("--tolerance", type=float, default=0.005,
                    help="Régression tolérée sur la métrique primaire")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    cand = _load(args.candidate)
    if cand is None:
        logger.error(f"Métriques candidat introuvables : {args.candidate}")
        return 2

    base = _load(args.baseline)
    if base is None:
        # Pas de référence (premier run) → on promeut par défaut
        logger.info("Aucune référence production → promotion autorisée (premier modèle)")
        return 0

    c = _get(cand, args.metric)
    b = _get(base, args.metric)
    if c is None or b is None:
        logger.error(f"Métrique '{args.metric}' absente (candidat={c}, baseline={b})")
        return 2

    lower_better = args.metric in LOWER_IS_BETTER or args.metric.endswith(("log_loss", "brier_score"))
    if lower_better:
        regressed = c > b + args.tolerance
        delta = c - b
    else:
        regressed = c < b - args.tolerance
        delta = c - b

    logger.info(f"{args.metric} : candidat={c:.4f} | production={b:.4f} | Δ={delta:+.4f} "
                f"(tolérance={args.tolerance}, {'plus bas' if lower_better else 'plus haut'}=mieux)")

    if regressed:
        logger.error("RÉGRESSION détectée → promotion REFUSÉE")
        return 1
    logger.info("Pas de régression → promotion AUTORISÉE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
