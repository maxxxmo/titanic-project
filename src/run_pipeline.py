from src.evaluation.evaluate import evaluate_model
from src.ingestion.load_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.training import train_model
from src.utils.mlflow_utils import start_mlflow_experiment, log_model_run
from sklearn.linear_model import LogisticRegression  # modèle spécifique
# Before the add of mlflow to test pipeline core
from src.evaluation.metrics_registry import METRICS_REGISTRY
# from src.training.model_registry import MODEL_REGISTRY

def main():
    # data ingestion

    print("Loading data...")
    df = load_data("data/raw/titanic.csv")
    # preprocessing
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # for training model
    model_cls = LogisticRegression
    params = {"max_iter": 200}
    
    
    print("Starting MLflow experiment...")
    start_mlflow_experiment()

    print("Training model...")
    model = train_model(model_cls, params, X_train, y_train)

    print("Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test, metrics_registry=METRICS_REGISTRY)
    # metrics = {"accuracy": accuracy}
    
    
    print(f"Model accuracy: {accuracy}")

    print("Logging run to MLflow...")
    metrics_float = {k: float(v) for k, v in accuracy.items()}
    log_model_run(model, params=params, metrics=metrics_float, run_name="logreg_v1")
    
    
    
    
if __name__ == "__main__":
    main()