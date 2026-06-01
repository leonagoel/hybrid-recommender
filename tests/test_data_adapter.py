import pandas as pd

from data_adapter import adapt_data


def test_description_column_is_detected():
    df = pd.DataFrame({
        "title": ["Laptop"],
        "description": ["High performance laptop"],
        "category": ["Electronics"]
    })

    adapted_df, meta = adapt_data(df)

    assert adapted_df["description"].iloc[0] == "High performance laptop"
    assert meta["desc_col"] == "description"