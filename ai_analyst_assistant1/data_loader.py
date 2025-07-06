import pandas as pd
import logging
import re

def load_csv(file):
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file)
        logging.info("CSV loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")
        return None

def get_preview(df):
    """Return head and tail of the DataFrame."""
    return {"head": df.head(), "tail": df.tail()}

def get_data_summary(df):
    """Generate a summary of the DataFrame."""
    try:
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "column_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else {}
        }
        logging.info("CSV summary generated")
        return summary
    except Exception as e:
        logging.error(f"Error generating CSV summary: {str(e)}")
        return {}