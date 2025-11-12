"""Model helpers for Focus-Buddy.

This module provides a small model factory and training/prediction helpers
so different regressors can be used without changing call-sites across the
project. Supported model names: 'ridge', 'linear', 'polynomial',
'random_forest', 'gbr', 'svr', and optionally 'xgboost' (if xgboost is
installed).
"""

from typing import Dict, Iterable, List, Optional
import warnings

import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from xgboost import XGBRegressor  # optional
    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None
    _HAS_XGBOOST = False


def _get_model(name: str, **params):
    name = name.lower()
    if name in ("ridge", "linear_ridge"):
        return Ridge(**params)
    if name in ("linear", "ols"):
        return LinearRegression(**params)
    if name == "polynomial":
        degree = params.pop("degree", 2)
        # standardize then polynomial features then linear regression
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lin", Ridge())
        ])
    if name in ("random_forest", "rf"):
        return RandomForestRegressor(**params)
    if name in ("gbr", "gradient_boosting", "gradientboosting"):
        return GradientBoostingRegressor(**params)
    if name == "svr":
        # scale inputs for SVR
        return Pipeline([("scaler", StandardScaler()), ("svr", SVR(**params))])
    if name == "xgboost":
        if not _HAS_XGBOOST:
            raise ImportError("xgboost is not installed; install xgboost or choose another model")
        return XGBRegressor(**params)
    raise ValueError(f"Unknown model name: {name}")


def train_models(
    df_or_list,
    target_columns: Iterable[str] = ("temp", "sound"),
    feature_columns: Optional[Iterable[str]] = None,
    model_name: str = "ridge",
    model_params: Optional[dict] = None,
) -> Dict[str, object]:
    """Train one regressor per target column and return a dict of models.

    - df: DataFrame containing features + target columns
    - target_columns: iterable of column names to predict
    - feature_columns: list of columns to use as features. If None, select all
      numeric columns except targets and 'time'.
    - model_name: string key for model factory
    - model_params: optional dict passed to model constructor

    Returns a dict mapping target_column -> fitted estimator.
    """
    # allow passing a list of daily DataFrames; concatenate into one DF
    if isinstance(df_or_list, (list, tuple)):
        df = pd.concat(df_or_list, ignore_index=True)
    else:
        df = df_or_list

    if model_params is None:
        model_params = {}

    if feature_columns is None:
        feature_columns = [
            c for c, dtype in df.dtypes.items()
            if pd.api.types.is_numeric_dtype(dtype) and c not in set(target_columns) and c != "time"
        ]

    models = {}
    for target in target_columns:
        X = df[list(feature_columns)]
        y = df[target]
        model = _get_model(model_name, **(model_params or {}))
        model.fit(X, y)
        try:
            score = model.score(X, y)
        except Exception:
            score = None
        print(f"Trained {model_name} for '{target}' (R²: {score})")
        models[target] = model

    return models


def predict_from_models(models: Dict[str, object], X_pred: pd.DataFrame, feature_columns: Optional[List[str]] = None):
    """Given a dict of trained models and a prediction DataFrame, return
    predictions in a dict mapping target -> ndarray.

    If feature_columns is provided, that ordering will be used when selecting
    columns from X_pred. Otherwise the function will select columns present
    in X_pred that match the model training columns.
    """
    preds = {}
    if feature_columns is not None:
        X_features = X_pred[list(feature_columns)]
    else:
        # assume X_pred already contains the right feature columns
        X_features = X_pred.copy()

    for target, model in models.items():
        preds[target] = model.predict(X_features)

    return preds


__all__ = ["train_models", "predict_from_models"]
