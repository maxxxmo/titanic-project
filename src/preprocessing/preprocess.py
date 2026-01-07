import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame):
    """
    Basic preprocessing:
    - select features
    - handle missing values
    - train/test split
    """
    features = ["Pclass", "Sex", "Age", "Fare"]
    target = "Survived"

    df = df[features + [target]].copy()

    # Simple encoding
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)
