"""OutlierDetector: one interface to three outlier-detection methods."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class OutlierDetector:
    """Detect outliers with one of three methods.

    method         'tukey', 'zscore' or 'isolation_forest'
    k              Tukey fence multiplier            (tukey only)
    threshold      modified Z-score cutoff           (zscore only)
    contamination  share of rows to flag             (isolation_forest only)

    After fit_detect(), the results are in mask_, n_outliers_ and pct_outliers_.
    """

    VALID_METHODS = ("tukey", "zscore", "isolation_forest")

    def __init__(self, method: str = "tukey", threshold: float = 3.5,
                 k: float = 1.5, contamination: float = 0.05,
                 random_state: int = 42):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got '{method}'")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not 0 < contamination < 0.5:
            raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")

        self.method = method
        self.threshold = threshold
        self.k = k
        self.contamination = contamination
        self.random_state = random_state

        self.mask_ = None
        self.n_outliers_ = None
        self.pct_outliers_ = None

    def fit_detect(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        """Return a boolean mask, True for outliers.

        'tukey' and 'zscore' take one column (a Series);
        'isolation_forest' takes a DataFrame of numeric columns.
        """
        if self.method != "isolation_forest" and not isinstance(data, pd.Series):
            raise TypeError(f"'{self.method}' works on one column: pass a pd.Series")

        if self.method == "zscore":
            mask = self._zscore(data)
        elif self.method == "tukey":
            mask = self._tukey(data)
        else:
            mask = self._isolation_forest(data)

        self.mask_ = mask
        # int() and float() turn numpy numbers into plain Python ones,
        # so summary() prints cleanly
        self.n_outliers_ = int(mask.sum())
        self.pct_outliers_ = float(mask.mean())
        return mask

    def summary(self) -> dict:
        """The method, how many rows it flagged, and the parameter it used."""
        if self.mask_ is None:
            raise RuntimeError("Call fit_detect() first.")

        if self.method == "zscore":
            params = {"threshold": self.threshold}
        elif self.method == "tukey":
            params = {"k": self.k}
        else:
            params = {"contamination": self.contamination}

        return {
            "method": self.method,
            "n_outliers": self.n_outliers_,
            "pct_outliers": round(self.pct_outliers_, 4),
            "params": params,
        }

    def _zscore(self, series: pd.Series) -> pd.Series:
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:    # over half the values are identical; avoid dividing by zero
            return pd.Series(False, index=series.index)
        modified_z = 0.6745 * (series - median) / mad
        return np.abs(modified_z) > self.threshold

    def _tukey(self, series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return (series < q1 - self.k * iqr) | (series > q3 + self.k * iqr)

    def _isolation_forest(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(data, pd.Series):
            data = data.to_frame()
        iso = IsolationForest(contamination=self.contamination,
                              random_state=self.random_state)
        preds = iso.fit_predict(data)
        return pd.Series(preds == -1, index=data.index)
