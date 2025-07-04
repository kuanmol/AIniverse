import pandas as pd

def load_csv(file):
    """Reads uploaded CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        return None

def get_data_summary(df):
    """Returns basic dataset info for analysis display."""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "column_types": df.dtypes.apply(lambda x: x.name).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict(),
    }
    return summary

def get_preview(df, n=5):
    """Returns head and tail preview."""
    return {
        "head": df.head(n).to_dict(orient="records"),
        "tail": df.tail(n).to_dict(orient="records")
    }
