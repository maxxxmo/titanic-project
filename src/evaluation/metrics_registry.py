from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


METRICS_REGISTRY = {
    "accuracy": {
        "func": accuracy_score,
        "needs_proba": False
    },
    "precision": {
        "func": precision_score,
        "needs_proba": False
    },
    "recall": {
        "func": recall_score,
        "needs_proba": False
    },
    "f1": {
        "func": f1_score,
        "needs_proba": False
    },
    "roc_auc": {
        "func": roc_auc_score,
        "needs_proba": True
    }
}
