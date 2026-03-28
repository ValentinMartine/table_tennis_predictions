"""
Modèle XGBoost — comparaison avec LightGBM.
Structure identique pour faciliter les comparaisons.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from .lgbm_model import FEATURE_COLS


class XGBModel:
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        xgb_cfg = config.get("models", {}).get("xgb", {})

        self.base_model = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 500),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            max_depth=xgb_cfg.get("max_depth", 6),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.model = CalibratedClassifierCV(self.base_model, method="isotonic", cv=5)
        self.feature_cols = FEATURE_COLS
        self._is_fitted = False

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].copy()
        for col in set(self.feature_cols) - set(available):
            X[col] = 0.0
        return X[self.feature_cols]

    def fit(self, df_train: pd.DataFrame) -> None:
        X = self._get_features(df_train)
        y = df_train["target"].values
        logger.info(f"Entraînement XGBoost sur {len(X)} exemples")
        self.model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Modèle non entraîné")
        return self.model.predict_proba(self._get_features(df))[:, 1]

    def save(self, path: str = "data/xgb_model.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "data/xgb_model.pkl") -> "XGBModel":
        with open(path, "rb") as f:
            return pickle.load(f)
