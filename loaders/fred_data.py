import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_fred_md(max_months: int = 3) -> pd.DataFrame:
    now = datetime.now()
    for months_back in range(0, max_months):
        date = now - relativedelta(months=months_back)
        url = f"https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/{date.year}-{date.month:02d}-md.csv"
        try:
            df = pd.read_csv(url)
            return df.iloc[1:].reset_index(drop=True)
        except Exception:
            continue
    raise RuntimeError("Could not fetch FRED-MD data")

def get_dropcols(df: pd.DataFrame, min_missing: int = 100) -> list[str]:
    dropcols = []
    for c in df.columns:
        if df[c].isna().sum() > min_missing:
            dropcols.append(c)
    return dropcols

def get_fillcols(df : pd.DataFrame, min_missing : int = 40) -> list[str]:
    fill_cols = []
    for c in df.columns:
        if df[c].isna().sum() > min_missing:
            fill_cols.append(c)
    return fill_cols

def add_lag(
        df:pd.DataFrame,
        cols: list[str] | str,
        ns:list[int] | None = None
        )->pd.DataFrame:
    if ns is None:
        ns = [1]
    if isinstance(cols, str):
        cols = [cols]
    copy_df = df.copy()
    for col in cols:
        for n in ns:
            copy_df[f"{col}_LAG_{n}"] = copy_df[col].shift(n)
    return copy_df


def add_delta(
        df: pd.DataFrame,
        cols: str | list[str],
        ns: list[int] | None=None,
        keep_lag:bool = False
        )->pd.DataFrame:
    if ns is None:
        ns=[1]
    if isinstance(cols, str):
        cols = [cols]
    lags_added = add_lag(df, cols, ns)
    lag_cols = []
    for col in cols:
        for n in ns:
            lag_col = f"{col}_LAG_{n}"
            lags_added[f"{col}_DELTA_{n}"] = lags_added[col] -lags_added[lag_col]
            lag_cols.append(lag_col)
    if not keep_lag:
        lags_added = lags_added.drop(columns=lag_cols)
    return lags_added
# get the latest FRED-MD data
def get_processed_fred(keep_last = False):
    df = get_fred_md()
    # remove columns with more than a fixed number of nulls (100 by default)
    dropcols = get_dropcols(df)
    df_dropped = df.drop(columns = dropcols )
    # get columns with too many missing values to interpolate on
    # these will be filled in but after train/test split to avoid leakage
    fill_cols = get_fillcols(df_dropped)
    # for remaining missing values use linear interpolation to fill missing values
    interp_cols = set(df_dropped.columns) - set(fill_cols) - {'sasdate'}
    for c in interp_cols:
        df_dropped[c] = df_dropped[c].interpolate(method="linear")
    # add the delta for UNRATE
    X_model = add_delta(df_dropped, "UNRATE")
    # add some lag features
    UE_cols = [c for c in X_model.columns if c.startswith("UE")]
    X_model = add_lag(X_model, cols=UE_cols + ["CLAIMSx"], ns=[1,2,3])
    # except for the fill_cols, we delete nulls (created by the lag columns)
    non_fill_cols = [c for c in X_model.columns if c not in fill_cols]
    X_model = X_model.dropna(subset=non_fill_cols)
    # get the UNRATE shifted by a month so the target is UNRATE the next month
    if not keep_last:
        X_model["UNRATE"] = X_model["UNRATE"].shift(-1)
        X_model = X_model.dropna(subset=["UNRATE"])
    return X_model
if __name__ == "__main__":
    X_model = get_processed_fred()
    X_model.to_csv("data/fred_processed.csv", index=False)

