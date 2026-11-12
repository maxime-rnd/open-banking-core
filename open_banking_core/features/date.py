"""Calendar and operation/execution date delta features."""

from typing import Any

import numpy as np
import pandas as pd

from .base import FeatureTransformer
from .registry import register_feature


@register_feature("date")
class DateFeatures(FeatureTransformer):
    """Extract calendar values and elapsed days between transaction dates."""

    def __init__(
        self,
        operation_date_column: str = "operation_date",
        execution_date_column: str = "execution_date",
        include_day_of_week: bool = True,
        include_day_of_month: bool = True,
        include_date_delta: bool = True,
    ) -> None:
        self.operation_date_column = operation_date_column
        self.execution_date_column = execution_date_column
        self.include_day_of_week = include_day_of_week
        self.include_day_of_month = include_day_of_month
        self.include_date_delta = include_date_delta

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DateFeatures":
        missing = {self.operation_date_column, self.execution_date_column} - set(X.columns)
        if missing:
            raise ValueError(f"date columns are missing: {', '.join(sorted(missing))}")
        self.feature_names_ = []
        if self.include_day_of_week:
            self.feature_names_.append("operation_day_of_week")
        if self.include_day_of_month:
            self.feature_names_.append("operation_day_of_month")
        if self.include_date_delta:
            self.feature_names_.append("execution_minus_operation_days")
        self.feature_names_.extend(["operation_date_missing", "execution_date_missing"])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        operation = pd.to_datetime(X[self.operation_date_column], errors="coerce")
        execution = pd.to_datetime(X[self.execution_date_column], errors="coerce")
        columns = []
        if self.include_day_of_week:
            columns.append(operation.dt.dayofweek.to_numpy(dtype=float))
        if self.include_day_of_month:
            columns.append(operation.dt.day.to_numpy(dtype=float))
        if self.include_date_delta:
            delta = (execution - operation).dt.total_seconds().to_numpy(dtype=float) / 86400.0
            columns.append(delta)
        columns.extend([operation.isna().to_numpy(dtype=float), execution.isna().to_numpy(dtype=float)])
        return np.column_stack(columns)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)
