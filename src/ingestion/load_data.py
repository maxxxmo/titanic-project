import pandas as pd
from pathlib import Path


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load Titanic dataset from CSV file.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_csv(path)
    return df
