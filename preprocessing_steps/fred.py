from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fill_cols = None
        self.fill_vals = None
    def fit(self, X:pd.DataFrame, y=None):
        self.fill_cols = []
        self.fill_vals = []
        for c in X.columns:
            if X[c].isna().sum() > 0:
                self.fill_cols.append(c)
                self.fill_vals.append(X[c].median())
        return self
    def transform(self, X:pd.DataFrame, y=None):
        if self.fill_cols is None or self.fill_vals is None:
            raise RuntimeError("MedianImputer must be fit before transforming")
        df = X.copy()
        for col, val in zip(self.fill_cols, self.fill_vals):
            df[col] = df[col].fillna(val)
        return df

fred_steps = [("median_imputer", MedianImputer())]
