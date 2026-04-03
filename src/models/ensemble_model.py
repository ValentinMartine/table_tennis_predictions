import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from .lgbm_model import LGBMModel
from .xgb_model import XGBModel

class EnsembleModel:
    """Combinaison pondérée de LGBM et XGBoost."""
    def __init__(self, lgbm_weight=0.5, xgb_weight=0.5):
        self.lgbm = LGBMModel()
        self.xgb = XGBModel()
        self.lgbm_weight = lgbm_weight
        self.xgb_weight = xgb_weight
        self._is_fitted = False

    def fit(self, df_train, df_val=None):
        logger.info(f"Entraînement de l'Ensemble (LGBM weight={self.lgbm_weight}, XGB weight= {self.xgb_weight})")
        self.lgbm.fit(df_train, df_val)
        self.xgb.fit(df_train, df_val)
        self._is_fitted = True

    def predict_proba(self, df):
        if not self._is_fitted:
            raise RuntimeError("Ensemble non entraîné")
        p_lgbm = self.lgbm.predict_proba(df)
        p_xgb = self.xgb.predict_proba(df)
        return self.lgbm_weight * p_lgbm + self.xgb_weight * p_xgb

    def save(self, path="data/ensemble_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Ensemble sauvegardé : {path}")

    @classmethod
    def load(cls, path="data/ensemble_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
