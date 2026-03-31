"""Data ingestion module for loading raw data."""

import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)


def load_raw_data(file_source) -> pd.DataFrame:
    """
    Load raw data from a CSV file path or file-like object.

    Args:
        file_source: A file path (str/Path) or a file-like object (bytes or IO).

    Returns:
        pd.DataFrame: The loaded dataframe.

    Raises:
        ValueError: If the file is empty or cannot be parsed.
    """
    try:
        if isinstance(file_source, bytes):
            df = pd.read_csv(io.BytesIO(file_source))
        elif isinstance(file_source, (str,)) or hasattr(file_source, "__fspath__"):
            df = pd.read_csv(file_source)
        else:
            # File-like object
            df = pd.read_csv(file_source)

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df

    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file has no data.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Could not parse CSV: {e}")
