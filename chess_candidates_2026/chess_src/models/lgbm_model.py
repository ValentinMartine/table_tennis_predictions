import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

# Feature columns used for training
FEATURE_COLS = [
    "black_elo_intra_delta",
    "elo_diff",
    "form_black",
    "round_norm",
    "white_elo_intra_delta",
    "form_white",
    "form_diff",
    "black_color_streak",
    "h2h_recent_points_white",
    "white_color_streak",
    "white_last2_score",
    "black_gap_to_leader",
    "h2h_matches",
    "tournament_points_diff",
    "black_last2_score",
    "white_gap_to_leader",
    "elo_prob_white",
    "white_tournament_points",
    "black_tournament_points",
    "white_color_balance",
    "h2h_points_white",
    "black_color_balance",
]


class ChessLGBMModel:
    def __init__(self, config_path: str = "config/settings.yaml", params: dict = None):
        if params is None:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            params = config.get("models", {}).get("lgbm", {})

        self.model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            num_leaves=params.get("num_leaves", 15),
            min_child_samples=params.get("min_child_samples", 5),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=params.get("reg_alpha", 0.5),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=42,
            verbose=-1,
        )
        self.feature_cols = FEATURE_COLS
        self._is_fitted = False
        self._calibrated = None

    def _prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = df[self.feature_cols].copy()
        # Map 0.0 -> 0 (Black), 0.5 -> 1 (Draw), 1.0 -> 2 (White)
        y = df["result"].map({0.0: 0, 0.5: 1, 1.0: 2}).astype(int)
        return X, y

    def fit(self, df_train: pd.DataFrame) -> None:
        X, y = self._prepare_data(df_train)
        self.model.fit(X, y)
        self._is_fitted = True

    def calibrate(self, df_cal: pd.DataFrame) -> None:
        """Fit isotonic calibration on a held-out calibration set."""
        X_cal, y_cal = self._prepare_data(df_cal)
        self._calibrated = CalibratedClassifierCV(
            self.model, method="sigmoid", cv="prefit"
        )
        self._calibrated.fit(X_cal, y_cal)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns [Prob(Black), Prob(Draw), Prob(White)]"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        X = df[self.feature_cols]
        if self._calibrated is not None:
            return self._calibrated.predict_proba(X)
        return self.model.predict_proba(X)

    def save(self, path: str = "data/chess_lgbm.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "data/chess_lgbm.pkl") -> "ChessLGBMModel":
        with open(path, "rb") as f:
            return pickle.load(f)
