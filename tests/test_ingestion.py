from src.ingestion.load_data import load_data

def test_load_data_returns_dataframe(tmp_path):
    csv = tmp_path / "titanic.csv"
    csv.write_text(
        "Pclass,Sex,Age,Fare,Survived\n"
        "1,male,22,7.25,0\n"
        "3,female,38,71.83,1\n"
    )

    df = load_data(csv)

    assert df.shape[0] == 2
    assert "Survived" in df.columns
