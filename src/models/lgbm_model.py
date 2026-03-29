"""
Modèle LightGBM avec calibration isotonique.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

FEATURE_COLS = [
    # Elo global
    "elo_diff", "elo_win_prob_p1",
    # Elo international (WTT Champions/Star Contender/Cup Finals, Worlds, JO)
    "elo_intl_diff", "elo_intl_win_prob_p1",
    # H2H (min_matches=1, fenêtre 5 ans)
    "h2h_matches", "h2h_winrate_p1", "h2h_recent_winrate_p1",
    # Forme récente (fenêtre 10 matchs)
    "form_p1", "form_p2", "form_diff",
    # Sets : volume + dominance
    "avg_sets_p1", "avg_sets_p2",
    "avg_set_margin_p1", "avg_set_margin_p2", "set_margin_diff",
    "close_sets_rate_p1", "close_sets_rate_p2",
    # Fatigue & repos
    "rest_hours_p1", "rest_hours_p2", "fatigue_p1", "fatigue_p2",
    # Rankings statiques ITTF & WTT
    "ittf_rank_p1", "ittf_rank_p2", "rank_diff",
    "wtt_rank_p1", "wtt_rank_p2", "wtt_rank_diff",
    # Trajectoire de classement (dynamique)
    "rank_velocity_p1", "rank_velocity_p2", "rank_velocity_diff",
    "rank_stability_p1", "rank_stability_p2",
    # Âge
    "age_p1", "age_p2", "age_diff",
    # Cotes bookmaker
    "has_odds", "implied_prob_p1",
]


class LGBMModel:
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        lgbm_cfg = config.get("models", {}).get("lgbm", {})

        self.base_model = LGBMClassifier(
            n_estimators=lgbm_cfg.get("n_estimators", 500),
            learning_rate=lgbm_cfg.get("learning_rate", 0.05),
            max_depth=lgbm_cfg.get("max_depth", 6),
            num_leaves=lgbm_cfg.get("num_leaves", 31),
            min_child_samples=lgbm_cfg.get("min_child_samples", 20),
            subsample=lgbm_cfg.get("subsample", 0.8),
            colsample_bytree=lgbm_cfg.get("colsample_bytree", 0.8),
            random_state=42,
            verbose=-1,
        )
        self.model = CalibratedClassifierCV(self.base_model, method="isotonic", cv=5)
        self.feature_cols = FEATURE_COLS
        self._is_fitted = False

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.feature_cols if c in df.columns]
        missing = set(self.feature_cols) - set(available)
        if missing:
            logger.warning(f"Features manquantes (remplacement par 0) : {missing}")
        X = df[available].copy()
        for col in missing:
            X[col] = 0.0
        return X[self.feature_cols]

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> None:
        X = self._get_features(df_train)
        y = df_train["target"].values
        logger.info(f"Entraînement LightGBM sur {len(X)} exemples, {len(self.feature_cols)} features")
        if df_val is not None:
            from sklearn.frozen import FrozenEstimator
            self.base_model.fit(X, y)
            X_val = self._get_features(df_val)
            y_val = df_val["target"].values
            self.model = CalibratedClassifierCV(FrozenEstimator(self.base_model), method="isotonic")
            self.model.fit(X_val, y_val)
            logger.info(f"Calibration isotonique sur val set ({len(X_val)} exemples)")
        else:
            self.model.fit(X, y)
        self._is_fitted = True
        logger.info("Entraînement terminé")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Modèle non entraîné — appeler fit() d'abord")
        X = self._get_features(df)
        return self.model.predict_proba(X)[:, 1]

    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        X = self._get_features(df)
        y = df["target"].values
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False)  # pas de shuffle → temporal
        scores = cross_val_score(self.base_model, X, y, cv=cv, scoring="neg_log_loss")
        return {
            "mean_log_loss": round(-scores.mean(), 4),
            "std_log_loss": round(scores.std(), 4),
        }

    def shap_analysis(self, df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
        X = self._get_features(df).head(n_samples)
        # CalibratedClassifierCV entraîne des clones — récupérer le premier estimateur fitté
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

    def save(self, path: str = "data/lgbm_model.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Modèle sauvegardé : {path}")

    @classmethod
    def load(cls, path: str = "data/lgbm_model.pkl") -> "LGBMModel":
        with open(path, "rb") as f:
            return pickle.load(f)
