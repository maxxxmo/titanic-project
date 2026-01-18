from unittest.mock import patch
from src.run_pipeline import run_pipeline

@patch("src.utils.mlflow_utils.log_experiment")
def test_pipeline_runs(mock_mlflow, tmp_path):
    run_pipeline(
        data_path="titanic/data/raw/titanic.csv",
        output_dir=tmp_path
    )

    assert mock_mlflow.called
