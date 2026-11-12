"""Simple training orchestration around the configured SignRouter."""

import pandas as pd

from open_banking_core.config import CoreConfig
from open_banking_core.pipelines import SignRouter, build_sign_router


class Trainer:
    """Fit debit and credit pipelines on a prepared training DataFrame."""

    def __init__(self, config: CoreConfig, target_column: str | None = None) -> None:
        self.config = config
        self.target_column = target_column or config.data.target_column or "category"

    def fit(self, train_data: pd.DataFrame) -> SignRouter:
        """Build and fit the sign router; train/test splitting stays upstream."""
        if self.target_column not in train_data:
            raise ValueError(f"training target column {self.target_column!r} is missing")
        if "amount" not in train_data:
            raise ValueError("training data must include the canonical amount column")
        if train_data[self.target_column].isna().any():
            raise ValueError("training target labels must not be null")
        X = train_data.drop(columns=[self.target_column])
        y = train_data[self.target_column]
        router = build_sign_router(self.config)
        router.fit(X, y)
        return router
