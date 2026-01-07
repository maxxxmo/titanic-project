import mlflow
import mlflow.sklearn

def start_mlflow_experiment(experiment_name="Titanic_Classification"):
    """
    Set the MLflow experiment. If it doesn't exist, create it.
    """
    mlflow.set_experiment(experiment_name)


def log_model_run(model, params: dict, metrics: dict, run_name=None):
    """
    Track a single model run in MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
