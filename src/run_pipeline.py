from src.evaluation.evaluate import evaluate_model
from src.evaluation.metrics_registry import METRICS_REGISTRY
from src.ingestion.load_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.training import train_model
from src.training.model_registry import MODEL_REGISTRY

def main():
    # data ingestion

    print("Loading data...")
    df = load_data("data/raw/titanic.csv")
    # preprocessing
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # for training model


    results = []

    for model_name, config in MODEL_REGISTRY.items():
        print(f"Training {model_name}...")

        model = train_model(
            model_cls=config["class"],
            params=config["params"],
            X_train=X_train,
            y_train=y_train
        )
        
        
    # evaluating model

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        metrics_registry=METRICS_REGISTRY
    )

    print(metrics)




if __name__ == "__main__":
    main()