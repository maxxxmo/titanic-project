from src.evaluation.evaluate import evaluate_model
from src.ingestion.load_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.training import train_model
from src.utils.mlflow_utils import start_mlflow_experiment, log_model_run
from src.evaluation.metrics_registry import METRICS_REGISTRY
from sklearn.linear_model import LogisticRegression


def run_pipeline(
    data_path: str,
    enable_mlflow: bool = True,
):
    # ingestion
    df = load_data(data_path)

    # preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # training
    model_cls = LogisticRegression
    params = {"max_iter": 200}

    if enable_mlflow:
        start_mlflow_experiment()

    model = train_model(model_cls, params, X_train, y_train)

    # evaluation
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        metrics_registry=METRICS_REGISTRY
    )

    if enable_mlflow:
        metrics_float = {k: float(v) for k, v in metrics.items()}
        log_model_run(
            model,
            params=params,
            metrics=metrics_float,
            run_name="logreg_v1"
        )

    return metrics


def main():
    run_pipeline(
        data_path="data/raw/titanic.csv",
        enable_mlflow=True
    )


if __name__ == "__main__":
    main()
