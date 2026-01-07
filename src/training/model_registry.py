from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


MODEL_REGISTRY = {
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {"max_iter": 200}
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 100, "random_state": 42}
    },
    "svm": {
        "class": SVC,
        "params": {"kernel": "rbf", "probability": True}
    }
}
