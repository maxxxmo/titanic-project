import pandas as pd
import pytest

@pytest.fixture
def raw_titanic_df():
    return pd.DataFrame({
        "Pclass": [1, 3],
        "Sex": ["male", "female"],
        "Age": [22, 38],
        "Fare": [7.25, 71.83],
        "Survived": [0, 1],
    })


@pytest.fixture
def processed_titanic_df():
    return pd.DataFrame({
        "Pclass": [1, 3],
        "Age": [22, 38],
        "Fare": [7.25, 71.83],
        "Sex_male": [1, 0],
        "Sex_female": [0, 1],
        "Survived": [0, 1],
    })
