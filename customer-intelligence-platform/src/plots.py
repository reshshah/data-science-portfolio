from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc_pr(y, prob, plot_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y, prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y, prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.tight_layout()
    plt.savefig(plot_dir / "precision_recall_curve.png")
    plt.close()


def plot_predicted_vs_actual(y, pred, plot_dir: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y, pred, alpha=0.4, s=15)
    max_value = max(max(y), max(pred))
    plt.plot([0, max_value], [0, max_value], "--", color="gray")
    plt.xlabel("Actual future_revenue_180d")
    plt.ylabel("Predicted future_revenue_180d")
    plt.title("Predicted vs. Actual")
    plt.tight_layout()
    plt.savefig(plot_dir / "predicted_vs_actual.png")
    plt.close()


def feature_importance(pipe) -> pd.DataFrame:
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        # LogisticRegression.coef_ is 2D (n_classes, n_features); Ridge.coef_
        # is 1D (n_features,) for single-output regression.
        weight = model.coef_[0] if model.coef_.ndim == 2 else model.coef_
    else:
        weight = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": weight})
        .assign(abs_coef=lambda d: d.coefficient.abs())
        .sort_values("abs_coef", ascending=False)
    )


def plot_feature_importance(fi: pd.DataFrame, plot_dir: Path, top_n: int = 20) -> None:
    top = fi.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["coefficient"])
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_importance.png")
    plt.close()
