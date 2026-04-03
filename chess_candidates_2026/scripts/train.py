import sqlite3
import pandas as pd
import numpy as np
import yaml
import sys
from itertools import product
from pathlib import Path
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "chess_src"))

from chess_src.features.pipeline import ChessFeaturePipeline
from chess_src.models.lgbm_model import ChessLGBMModel, FEATURE_COLS

RR_KEYWORDS = [
    "Norway Chess",
    "Sinquefield Cup",
    "Tata Steel Masters",
    "Tata Steel 2024",
    "Candidates",
    "FIDE Candidates",
    "American Cup Champ",
    "Lindores Abbey",
    "Saint Louis Masters",
    "FIDE Grand Prix",
    "Carlsen vs.",
    "Ding vs.",
    "Hainan Danzhou",
    "Leon GM",
    "Sigeman",
    "Superbet",
    "Shamkir",
    "TechM GCL",
    "Grand Chess Tour",
    "Tal Memorial",
]


def is_round_robin(tournament: str) -> bool:
    return any(kw.lower() in tournament.lower() for kw in RR_KEYWORDS)


def evaluate(model, df_features, split_name: str) -> tuple[float, float]:
    df_eval = df_features.dropna(subset=["result"])
    probs = model.predict_proba(df_eval)
    y_true = df_eval["result"].map({0.0: 0, 0.5: 1, 1.0: 2}).astype(int)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, probs)
    logger.info(
        f"[{split_name}] Accuracy={acc:.3f}  LogLoss={ll:.3f}  DrawRate={(y_pred == 1).mean():.1%}"
    )
    return acc, ll


def load_and_filter(db_path, config_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    ratings = pd.read_sql_query(
        "SELECT id, rating_initial FROM players WHERE rating_initial IS NOT NULL", conn
    )
    conn.close()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    ratings_map = ratings.set_index("id")["rating_initial"].to_dict()
    df["_wr"] = df["white_id"].map(ratings_map)
    df["_br"] = df["black_id"].map(ratings_map)
    df["_avg"] = (df["_wr"] + df["_br"]) / 2
    is_candidates = df["tournament"] == "Candidates 2026"
    df = df[(df["_avg"] >= 2600) | is_candidates].drop(columns=["_wr", "_br", "_avg"])

    rr_mask = df["tournament"].apply(is_round_robin) | is_candidates
    df_rr_extra = pd.concat([df[rr_mask]] * 2, ignore_index=True)
    df = pd.concat([df, df_rr_extra], ignore_index=True)

    return df, config


def build_features(df, config, db_path):
    pipeline = ChessFeaturePipeline(config, db_path=str(db_path))
    df_f = pipeline.process(df)
    df_f = df_f.dropna(subset=["result"])
    return df_f.sort_values("played_at").reset_index(drop=True)


def train():
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    db_path = PROJECT_ROOT / "data" / "chess_matches.db"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config.get("players"):
        logger.error("FATAL: No candidates found in config.")
        return

    logger.info("Loading and filtering data...")
    df, config = load_and_filter(db_path, config_path)
    logger.info(f"Dataset: {len(df)} rows after filters")

    logger.info("Computing features (this takes ~1 min)...")
    df_features = build_features(df, config, db_path)
    logger.info(
        f"Feature rows: {len(df_features)}  |  Elo Diff Std={df_features['elo_diff'].std():.1f}"
    )

    if df_features["elo_diff"].std() < 50:
        logger.error("FATAL: Elo variation too low.")
        return

    # Time-split: 70% train / 15% cal / 15% test
    n = len(df_features)
    i_cal = int(n * 0.70)
    i_test = int(n * 0.85)
    df_train = df_features.iloc[:i_cal]
    df_cal = df_features.iloc[i_cal:i_test]
    df_test = df_features.iloc[i_test:]
    logger.info(
        f"Split — Train:{len(df_train)}  Cal:{len(df_cal)}  Test:{len(df_test)}"
    )

    # ── Step 1: Feature importance on a quick baseline model ──────────────────
    logger.info("Step 1: Feature importance analysis...")
    base_model = ChessLGBMModel(config_path=str(config_path))
    base_model.fit(df_train)

    importances = pd.Series(
        base_model.model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    logger.info("Feature importances (gain):")
    for feat, imp in importances.items():
        logger.info(f"  {feat:<35} {imp:>6.0f}")

    # Keep features with importance > 5% of max (drop pure noise)
    threshold = importances.max() * 0.05
    kept_features = importances[importances >= threshold].index.tolist()
    dropped = [f for f in FEATURE_COLS if f not in kept_features]
    logger.info(
        f"Keeping {len(kept_features)}/{len(FEATURE_COLS)} features (dropped: {dropped})"
    )

    # ── Step 2: Grid search on key hyperparams ────────────────────────────────
    logger.info("Step 2: Grid search (max_depth x num_leaves x reg_lambda)...")

    grid = {
        "max_depth": [3, 4, 5],
        "num_leaves": [8, 15, 31],
        "reg_lambda": [0.5, 1.0, 2.0],
    }

    best_ll = float("inf")
    best_params = {}

    for depth, leaves, reg in product(
        grid["max_depth"], grid["num_leaves"], grid["reg_lambda"]
    ):
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": depth,
            "num_leaves": leaves,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": reg,
        }
        m = ChessLGBMModel(params=params)
        m.feature_cols = kept_features
        m.model.set_params(**{k: v for k, v in params.items()})
        m.fit(df_train)
        _, ll = evaluate(m, df_cal, f"d={depth} l={leaves} rl={reg}")
        if ll < best_ll:
            best_ll = ll
            best_params = params

    logger.info(f"Best params: {best_params}  (LogLoss={best_ll:.4f})")

    # ── Step 3: Final model with best params + pruned features ────────────────
    logger.info("Step 3: Training final model...")
    final_model = ChessLGBMModel(params=best_params)
    final_model.feature_cols = kept_features
    final_model.model.set_params(**{k: v for k, v in best_params.items()})
    final_model.fit(df_train)

    logger.info("--- Backtest (uncalibrated) ---")
    evaluate(final_model, df_train, "Train")
    evaluate(final_model, df_test, "Test")

    logger.info("Calibrating with Platt scaling (sigmoid)...")
    final_model.calibrate(df_cal)

    logger.info("--- Backtest (calibrated) ---")
    evaluate(final_model, df_train, "Train")
    _, test_ll = evaluate(final_model, df_test, "Test")

    dist = df_test["result"].value_counts(normalize=True)
    logger.info(
        f"Test distribution — Black:{dist.get(0.0, 0):.1%}  Draw:{dist.get(0.5, 0):.1%}  White:{dist.get(1.0, 0):.1%}"
    )

    # Persist pruned feature list back to lgbm_model.py for dashboard consistency
    _update_feature_cols(kept_features)

    model_path = PROJECT_ROOT / "data" / "chess_lgbm.pkl"
    final_model.save(str(model_path))
    logger.info(f"Model saved → {model_path}")


def _update_feature_cols(kept: list[str]):
    """Rewrite FEATURE_COLS in lgbm_model.py to match the pruned feature set."""
    model_file = PROJECT_ROOT / "chess_src" / "models" / "lgbm_model.py"
    src = model_file.read_text(encoding="utf-8")
    new_list = "[\n" + "".join(f'    "{f}",\n' for f in kept) + "]"
    import re

    new_src = re.sub(
        r"FEATURE_COLS\s*=\s*\[.*?\]",
        f"FEATURE_COLS = {new_list}",
        src,
        flags=re.DOTALL,
    )
    model_file.write_text(new_src, encoding="utf-8")
    logger.info(f"FEATURE_COLS updated in lgbm_model.py ({len(kept)} features)")


if __name__ == "__main__":
    train()
