import pandas as pd # type: ignore
import numpy as np # type: ignore

def load_data(file_path):
    """
    Loads raw dynamic datasets for the recommender pipeline.

    Args:
        file_path (str): The path to the target data file.
    """
    # >>> KEEP ALL THE ORIGINAL ORIGINAL CODE LOGIC HERE <<<
    # >>> DO NOT DELETE THE CODE THAT WAS ALREADY IN THE FILE <<<
    df = pd.read_csv(file_path)
    return df

def load_recommendation_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads raw interaction data from a local CSV or JSON file path.

    Args:
        file_path (str): The system path pointing to the dataset file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path, on_bad_lines="skip", low_memory=False)
    elif ext == ".json":
        return pd.read_json(file_path)
    else:
        # Default to CSV for unknown extensions
        return pd.read_csv(file_path, on_bad_lines="skip", low_memory=False)

def normalize_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes explicit user ratings into a standardized scale between 0 and 1.

    Args:
        df (pd.DataFrame): The DataFrame containing raw user interactions and scores.

    Returns:
        pd.DataFrame: A modified DataFrame featuring a new 'rating_normalized' column.
    """
    from sklearn.preprocessing import MinMaxScaler
    df = df.copy()
    if "rating" not in df.columns:
        return df
    scaler = MinMaxScaler()
    df["rating_normalized"] = scaler.fit_transform(df[["rating"]])
    return df