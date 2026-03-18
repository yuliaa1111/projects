# ret_pred/tree_models/catboost.py
import numpy as np
from .registry import register_tree_model

@register_tree_model("catboost")
class CatBoostRegressorWrapper:
    def __init__(self, **params):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError("Please pip install catboost") from e

        self.model = CatBoostRegressor(**(params or {}))

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X))
