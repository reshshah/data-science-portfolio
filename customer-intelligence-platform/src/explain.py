"""Model-agnostic explainability: SHAP values for a fitted Pipeline(preprocess, model).

Works on the transformed (post-preprocessing) feature space, since that's
what the model actually sees. Dispatches to the fast, exact explainer for
tree models (TreeExplainer) and the fast, exact explainer for linear models
(LinearExplainer) rather than the slow, approximate general-purpose one --
every model type this project trains supports one of the two.
"""

import numpy as np
import pandas as pd
import shap

TREE_MODEL_TYPES = {"XGBClassifier", "LGBMClassifier"}
LINEAR_MODEL_TYPES = {"LogisticRegression"}


def _transform(pipe, X: pd.DataFrame) -> pd.DataFrame:
    preprocessor = pipe.named_steps["preprocess"]
    # Strip the ColumnTransformer's "numeric__"/"categorical__" prefixes --
    # useful for routing internally, just noise on a plot axis or in a CSV.
    feature_names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)


def compute_shap_values(
    pipe,
    X: pd.DataFrame,
    sample_size: int = 500,
    background_size: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP values for a sample of X, in the model's transformed feature space.

    Returns (shap_values, X_transformed) -- shap_values has one row per
    sampled observation, one column per transformed feature, and is always
    oriented toward the positive class for binary classifiers.
    """
    X_sample = X.sample(min(sample_size, len(X)), random_state=random_state) if len(X) > sample_size else X
    X_transformed = _transform(pipe, X_sample)
    model = pipe.named_steps["model"]
    model_type = type(model).__name__

    if model_type in TREE_MODEL_TYPES:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        # Binary classifiers: some tree explainers return a list [neg_class,
        # pos_class] or a 3D array (n_samples, n_features, n_classes) --
        # normalize to the positive-class contributions either way.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    elif model_type in LINEAR_MODEL_TYPES:
        background = shap.sample(X_transformed, min(background_size, len(X_transformed)), random_state=random_state)
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_transformed)
    else:
        raise ValueError(f"No SHAP explainer configured for model type: {model_type}")

    return shap_values, X_transformed


def summarize_shap(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Mean absolute SHAP value per feature -- a model-agnostic importance
    ranking that (unlike raw coefficients/split-gain) is comparable across
    model types and reflects actual impact on predictions, not just how
    the model happens to weight a feature."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
