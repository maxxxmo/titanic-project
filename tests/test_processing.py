from src.preprocessing.preprocess import preprocess_data

def test_preprocess_no_missing_values(raw_titanic_df):
    df = preprocess_data(raw_titanic_df)

    assert df.isnull().sum().sum() == 0
    assert "Survived" in df.columns
