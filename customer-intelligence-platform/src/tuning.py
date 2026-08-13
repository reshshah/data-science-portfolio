"""Cross-validated hyperparameter search over the same Pipeline(preprocess,
model) shape used everywhere else, so a tuned model is a drop-in replacement
for a manually-configured one.

Search happens entirely within train (StratifiedKFold), never touching
validation or test -- validation/test remain held out for the honest
evaluation `models/tune_classifier.py` runs afterward on the winning model.
"""

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.trainer import build_estimator


def run_hyperparameter_search(
    X_train,
    y_train,
    numeric_features: list,
    categorical_features: list,
    config: dict,
) -> RandomizedSearchCV:
    """Randomized search over config["tuning"]["param_distributions"].

    Config keys under "tuning":
        param_distributions: dict of {hyperparameter: list-of-values or
            scipy distribution}, unprefixed (e.g. "n_estimators", not
            "model__n_estimators" -- the model__ prefix is added here since
            it's an implementation detail of the Pipeline, not something a
            config author should need to know).
        cv_folds: number of StratifiedKFold splits (default 5)
        n_iter: number of random parameter combinations to try (default 20)
        scoring: sklearn scoring string (default "roc_auc")
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    base_estimator = build_estimator(config["model"], config["random_state"], y_train)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", base_estimator),
    ])

    tuning_cfg = config["tuning"]
    param_distributions = {
        f"model__{key}": value
        for key, value in tuning_cfg["param_distributions"].items()
    }
    cv = StratifiedKFold(
        n_splits=tuning_cfg.get("cv_folds", 5),
        shuffle=True,
        random_state=config["random_state"],
    )

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=tuning_cfg.get("n_iter", 20),
        scoring=tuning_cfg.get("scoring", "roc_auc"),
        cv=cv,
        random_state=config["random_state"],
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search
