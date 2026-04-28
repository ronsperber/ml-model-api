import pandas as pd
from typing import Any
from loaders.fred_data import get_processed_fred

def get_fred_fields() -> dict[str, Any]:
    df = get_processed_fred()
    feature_cols = set(df.columns) - {"sasdate", "UNRATE"}
    fred_fields = {col: (float, ...) for col in feature_cols}
    return fred_fields
