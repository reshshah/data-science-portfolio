import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing import build_preprocessor


def train_dummy(X_train, y_train) -> DummyClassifier:
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    return dummy


def _build_logistic_regression(model_cfg: dict, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=model_cfg["max_iter"],
        class_weight=model_cfg["class_weight"],
        random_state=random_state,
    )


def _build_xgboost(model_cfg: dict, random_state: int, y_train) -> XGBClassifier:
    scale_pos_weight = 1.0
    if model_cfg.get("balance_classes", False):
        negative, positive = np.bincount(y_train)
        scale_pos_weight = negative / positive
    return XGBClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        learning_rate=model_cfg["learning_rate"],
        subsample=model_cfg["subsample"],
        colsample_bytree=model_cfg["colsample_bytree"],
        min_child_weight=model_cfg["min_child_weight"],
        reg_lambda=model_cfg["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
    )


def _build_lightgbm(model_cfg: dict, random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=model_cfg["n_estimators"],
        num_leaves=model_cfg["num_leaves"],
        max_depth=model_cfg["max_depth"],
        learning_rate=model_cfg["learning_rate"],
        min_child_samples=model_cfg["min_child_samples"],
        class_weight=model_cfg["class_weight"],
        random_state=random_state,
        verbose=-1,
    )


def build_estimator(model_cfg: dict, random_state: int, y_train):
    model_type = model_cfg["type"]
    if model_type == "logistic_regression":
        return _build_logistic_regression(model_cfg, random_state)
    if model_type == "xgboost":
        return _build_xgboost(model_cfg, random_state, y_train)
    if model_type == "lightgbm":
        return _build_lightgbm(model_cfg, random_state)
    raise ValueError(f"Unknown model type: {model_type}")


def train_model(X_train, y_train, numeric_features: list, categorical_features: list, config: dict) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    estimator = build_estimator(config["model"], config["random_state"], y_train)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_dummy_regressor(X_train, y_train) -> DummyRegressor:
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    return dummy


def _build_ridge(model_cfg: dict, random_state: int) -> Ridge:
    return Ridge(alpha=model_cfg["alpha"], random_state=random_state)


def _build_xgboost_regressor(model_cfg: dict, random_state: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        learning_rate=model_cfg["learning_rate"],
        subsample=model_cfg["subsample"],
        colsample_bytree=model_cfg["colsample_bytree"],
        min_child_weight=model_cfg["min_child_weight"],
        reg_lambda=model_cfg["reg_lambda"],
        random_state=random_state,
    )


def _build_lightgbm_regressor(model_cfg: dict, random_state: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=model_cfg["n_estimators"],
        num_leaves=model_cfg["num_leaves"],
        max_depth=model_cfg["max_depth"],
        learning_rate=model_cfg["learning_rate"],
        min_child_samples=model_cfg["min_child_samples"],
        random_state=random_state,
        verbose=-1,
    )


def build_regression_estimator(model_cfg: dict, random_state: int):
    model_type = model_cfg["type"]
    if model_type == "ridge":
        return _build_ridge(model_cfg, random_state)
    if model_type == "xgboost":
        return _build_xgboost_regressor(model_cfg, random_state)
    if model_type == "lightgbm":
        return _build_lightgbm_regressor(model_cfg, random_state)
    raise ValueError(f"Unknown regression model type: {model_type}")


def train_regression_model(X_train, y_train, numeric_features: list, categorical_features: list, config: dict) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    estimator = build_regression_estimator(config["model"], config["random_state"])
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])
    pipe.fit(X_train, y_train)
    return pipe
