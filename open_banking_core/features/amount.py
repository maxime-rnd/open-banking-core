"""Numeric amount features."""

from math import log1p
from typing import Any

import numpy as np
import pandas as pd

from .base import FeatureTransformer
from .registry import register_feature


@register_feature("amount")
class AmountFeatures(FeatureTransformer):
    """Derive raw, absolute, and log absolute amount values."""

    def __init__(
        self,
        column: str = "amount",
        include_raw: bool = True,
        include_absolute: bool = True,
        include_log_absolute: bool = True,
    ) -> None:
        self.column = column
        self.include_raw = include_raw
        self.include_absolute = include_absolute
        self.include_log_absolute = include_log_absolute

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AmountFeatures":
        if self.column not in X:
            raise ValueError(f"amount column {self.column!r} is missing")
        self.feature_names_ = [
            name for name, enabled in (
                ("amount", self.include_raw),
                ("amount_absolute", self.include_absolute),
                ("amount_log_absolute", self.include_log_absolute),
            ) if enabled
        ]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(X[self.column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("amount feature input must contain finite numeric values")
        columns = []
        if self.include_raw:
            columns.append(values)
        if self.include_absolute:
            columns.append(np.abs(values))
        if self.include_log_absolute:
            columns.append(np.log1p(np.abs(values)))
        return np.column_stack(columns) if columns else np.empty((len(values), 0), dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)
