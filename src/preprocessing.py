import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


def to_bool_if_binary(col: pd.Series) -> pd.Series:
    if col.dtype in ["int64", "float64", "int32", "float32"]:
        vals = set(col.dropna().unique())
        if vals <= {0, 1}:
            return col.fillna(False).astype(bool)
    return col


def infer_task_type(y: pd.Series) -> str:
    binary_y = to_bool_if_binary(y)
    if binary_y.dtype == "bool" or y.dtype == "bool":
        return "binary"
    elif pd.api.types.is_numeric_dtype(y):
        return "continuous"
    else:
        return "categorical"


class UnixTimestampTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        return self

    def transform(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out_cols = []
        for col in X_df.columns:
            dt = pd.to_datetime(X_df[col], errors="coerce")
            vals = dt.astype("int64").to_numpy(dtype="float64", copy=False)
            vals[dt.isna().to_numpy()] = np.nan
            out_cols.append(vals / 1_000_000_000.0)
        if not out_cols:
            return np.empty((len(X_df), 0), dtype="float64")
        return np.column_stack(out_cols)


class CommaSeparatedMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, separator: str = ","):
        self.separator = separator

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self._mlbs = []
        for col in X_df.columns:
            labels = X_df[col].apply(self._split_tokens).tolist()
            mlb = MultiLabelBinarizer(sparse_output=True)
            mlb.fit(labels)
            self._mlbs.append(mlb)
        return self

    def transform(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        blocks = []
        for idx, col in enumerate(X_df.columns):
            labels = X_df[col].apply(self._split_tokens).tolist()
            block = self._mlbs[idx].transform(labels)
            blocks.append(block.tocsr())
        if not blocks:
            return sparse.csr_matrix((len(X_df), 0), dtype=np.float64)
        return sparse.hstack(blocks, format="csr")

    def _split_tokens(self, value):
        if isinstance(value, list):
            raw_tokens = value
        elif isinstance(value, str):
            raw_tokens = value.split(self.separator)
        elif pd.isna(value):
            raw_tokens = []
        else:
            raw_tokens = [str(value)]
        return [token.strip() for token in raw_tokens if str(token).strip()]
