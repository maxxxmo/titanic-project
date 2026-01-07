# create a function for each metrics and a global function for evaluation

def evaluate_model(model, X_test, y_test, metrics_registry):
    """
    Evaluate a model on multiple metrics.
    """
    y_pred = model.predict(X_test)

    results = {}

    for metric_name, config in metrics_registry.items():
        if config["needs_proba"]:
            y_score = model.predict_proba(X_test)[:, 1]
            score = config["func"](y_test, y_score)
        else:
            score = config["func"](y_test, y_pred)

        results[metric_name] = score

    return results
