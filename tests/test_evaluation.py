from src.evaluation.evaluate import evaluate_model
from sklearn.linear_model import LogisticRegression

def test_evaluate_model_returns_metrics(processed_titanic_df):
    X = processed_titanic_df.drop(columns=["Survived"])
    y = processed_titanic_df["Survived"]

    model = LogisticRegression().fit(X, y)
    metrics = evaluate_model(model, X, y)

    assert isinstance(metrics, dict)
    assert len(metrics) > 0
