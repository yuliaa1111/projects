# ret_pred/tree_models/lgbm.py
from typing import Any
import numpy as np

from .registry import register_tree_model


@register_tree_model("lgbm")
class LGBMRegressor:

    def __init__(self, **params: Any):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("Please pip install lightgbm") from e

        print("[DEBUG][LGBM] raw params:", params)

        self.model = lgb.LGBMRegressor(**(params or {}))

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
