import sqlite3

import mlflow

from src.tracking import log_config_params, log_metrics, start_run


def test_start_run_creates_a_local_sqlite_store(tmp_path):
    config = {
        "name": "smoke_test",
        "target": "churn_label_180d",
        "random_state": 42,
        "model": {"type": "logistic_regression", "max_iter": 200},
    }

    with start_run(config, tmp_path):
        log_config_params(config)
        log_metrics("validation", {"roc_auc": 0.8, "confusion_matrix": [[1, 2], [3, 4]]})

    db_path = tmp_path / "mlruns.db"
    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    run_name, status = conn.execute("select name, status from runs").fetchone()
    assert run_name == "logistic_regression"
    assert status == "FINISHED"

    params = dict(conn.execute("select key, value from params").fetchall())
    assert params["model__type"] == "logistic_regression"
    assert params["target"] == "churn_label_180d"

    metrics = dict(conn.execute("select key, value from metrics").fetchall())
    assert metrics["validation_roc_auc"] == 0.8
    # non-scalar values (confusion_matrix) must never reach mlflow.log_metrics
    assert "validation_confusion_matrix" not in metrics


def test_start_run_names_the_run_after_model_type_by_default(tmp_path):
    config = {
        "name": "smoke_test_2",
        "target": "propensity_label_30d",
        "random_state": 42,
        "model": {"type": "xgboost"},
    }

    with start_run(config, tmp_path) as run:
        active_run_name = mlflow.get_run(run.info.run_id).info.run_name

    assert active_run_name == "xgboost"
