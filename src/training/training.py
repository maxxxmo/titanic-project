

def train_model(model_cls, params, X_train, y_train):
    """
    Generic training function for sklearn models.
    """
    model = model_cls(**params)
    model.fit(X_train, y_train)
    return model
