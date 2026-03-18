# ret_pred/tree_models/xgb.py
from typing import Any
import numpy as np

from .registry import register_tree_model


@register_tree_model("xgb")
class XGBRegressor:
    """XGBoost regressor wrapper (sklearn API)"""
    def __init__(self, **params: Any):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("Please pip install xgboost") from e

        self.model = xgb.XGBRegressor(**(params or {}))

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
