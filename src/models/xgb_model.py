"""
Modèle XGBoost — comparaison avec LightGBM.
Structure identique pour faciliter les comparaisons.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from .lgbm_model import FEATURE_COLS


class XGBModel:
    def __init__(self, config_path: str = "config/settings.yaml", params: dict = None):
        if params is None:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            params = config.get("models", {}).get("xgb", {})

        self.base_model = XGBClassifier(
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 0.0),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.model = CalibratedClassifierCV(self.base_model, method="isotonic", cv=5)
        self.feature_cols = [
            # Rankings
            "ittf_rank_p1", "ittf_rank_p2", "rank_diff",
            "log_ittf_rank_p1", "log_ittf_rank_p2", "log_rank_diff",
            "wtt_rank_p1", "wtt_rank_p2", "wtt_rank_diff",
            "log_wtt_rank_p1", "log_wtt_rank_p2",
            # Trajectoire de classement
            "rank_velocity_p1", "rank_velocity_p2", "rank_velocity_diff",
            "rank_stability_p1", "rank_stability_p2",
            # Âge
            "age_p1", "age_p2", "age_diff",
            # Style
            "is_p1_lefty", "is_p2_lefty", "is_opposite_hand",
            # Expérience et Inconnu
            "is_p1_unknown", "is_p2_unknown", "matches_played_p1", "matches_played_p2",
            # Cotes bookmaker
            "has_odds", "implied_prob_p1",
        ]
        self._is_fitted = False

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].copy()
        for col in set(self.feature_cols) - set(available):
            X[col] = 0.0
        return X[self.feature_cols]

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> None:
        X = self._get_features(df_train)
        y = df_train["target"].values
        weights = df_train["sample_weight"].values if "sample_weight" in df_train.columns else None

        logger.info(f"Entraînement XGBoost sur {len(X)} exemples")
        if weights is not None:
            logger.info("Utilisation de poids d'entraînement (Time-Decay)")

        if df_val is not None:
            from sklearn.frozen import FrozenEstimator
            self.base_model.fit(X, y, sample_weight=weights)
            X_val = self._get_features(df_val)
            y_val = df_val["target"].values
            self.model = CalibratedClassifierCV(FrozenEstimator(self.base_model), method="isotonic")
            self.model.fit(X_val, y_val)
            logger.info(f"Calibration isotonique sur val set ({len(X_val)} exemples)")
        else:
            self.model.fit(X, y, sample_weight=weights)
        self._is_fitted = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Modèle non entraîné")
        return self.model.predict_proba(self._get_features(df))[:, 1]

    def shap_analysis(self, df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
        X = self._get_features(df).head(n_samples)
        fitted_estimator = self.model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(fitted_estimator)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        return importance

    def save(self, path: str = "data/xgb_model.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "data/xgb_model.pkl") -> "XGBModel":
        with open(path, "rb") as f:
            return pickle.load(f)
