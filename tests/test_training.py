from src.training.training import train_model

def test_train_model_returns_fitted_model(processed_titanic_df):
    X = processed_titanic_df.drop(columns=["Survived"])
    y = processed_titanic_df["Survived"]

    model = train_model(X, y)

    assert hasattr(model, "predict")
