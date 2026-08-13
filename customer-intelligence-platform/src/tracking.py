"""MLflow run tracking, scoped to a local SQLite store under the project root.

No tracking server required -- `mlflow ui --backend-store-uri
sqlite:///<root>/mlruns.db` from the project root browses runs locally.
(The plain filesystem backend, e.g. `file:./mlruns`, is deprecated as of
MLflow 3.x and rejects new writes -- SQLite is the supported local option.)
"""

from pathlib import Path

import mlflow


def start_run(config: dict, root: Path, run_name: str | None = None):
    """Point MLflow at <root>/mlruns.db, select the experiment, start a run.

    One experiment per task (config["name"], e.g. "churn"); one run per
    training invocation, named after the model type by default so runs are
    distinguishable in the MLflow UI without opening each one.
    """
    db_path = root / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(config["name"])
    return mlflow.start_run(run_name=run_name or config["model"]["type"])


def log_config_params(config: dict) -> None:
    """Flatten config["model"] (hyperparameters) and threshold settings into
    MLflow params -- everything needed to reproduce this exact run."""
    params = {f"model__{k}": v for k, v in config["model"].items()}
    params["target"] = config["target"]
    params["random_state"] = config["random_state"]
    mlflow.log_params(params)


def log_metrics(prefix: str, metrics: dict) -> None:
    """Log the numeric entries of a metrics dict (drops confusion_matrix,
    which isn't a scalar MLflow can chart)."""
    mlflow.log_metrics({
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    })
