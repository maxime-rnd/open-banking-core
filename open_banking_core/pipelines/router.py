"""Scikit-learn-compatible router for independent debit and credit models."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class SignRouter(ClassifierMixin, BaseEstimator):
    """Fit and predict with separate classifiers chosen by amount sign."""

    def __init__(self, debit_pipeline: BaseEstimator, credit_pipeline: BaseEstimator) -> None:
        self.debit_pipeline = debit_pipeline
        self.credit_pipeline = credit_pipeline

    @staticmethod
    def _masks(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if "amount" not in X:
            raise ValueError("amount column is required for sign routing")
        amounts = pd.to_numeric(X["amount"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(amounts).all():
            raise ValueError("amount values must be finite numbers")
        return amounts <= 0, amounts > 0

    def fit(self, X: pd.DataFrame, y: Any) -> "SignRouter":
        """Fit debit and credit models on their own rows and labels."""
        debit_mask, credit_mask = self._masks(X)
        if not debit_mask.any() or not credit_mask.any():
            raise ValueError("training data must contain at least one debit and one credit")
        labels = np.asarray(y)
        if len(labels) != len(X):
            raise ValueError("X and y must contain the same number of rows")
        self.debit_pipeline_ = clone(self.debit_pipeline)
        self.credit_pipeline_ = clone(self.credit_pipeline)
        self.debit_pipeline_.fit(X.iloc[debit_mask], labels[debit_mask])
        self.credit_pipeline_.fit(X.iloc[credit_mask], labels[credit_mask])
        self.classes_ = np.unique(labels)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict categories with sign-specific pipelines in original row order."""
        if not hasattr(self, "debit_pipeline_"):
            raise ValueError("SignRouter must be fitted before predict")
        debit_mask, credit_mask = self._masks(X)
        predictions = np.empty(len(X), dtype=object)
        if debit_mask.any():
            predictions[debit_mask] = self.debit_pipeline_.predict(X.iloc[debit_mask])
        if credit_mask.any():
            predictions[credit_mask] = self.credit_pipeline_.predict(X.iloc[credit_mask])
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return sign-specific probabilities aligned to the router class order."""
        if not hasattr(self, "debit_pipeline_"):
            raise ValueError("SignRouter must be fitted before predict_proba")
        debit_mask, credit_mask = self._masks(X)
        probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)
        for mask, pipeline in (
            (debit_mask, self.debit_pipeline_),
            (credit_mask, self.credit_pipeline_),
        ):
            if not mask.any():
                continue
            if not hasattr(pipeline, "predict_proba"):
                raise ValueError(f"{type(pipeline).__name__} does not provide predict_proba")
            branch_probabilities = pipeline.predict_proba(X.iloc[mask])
            branch_classes = pipeline.classes_
            for branch_index, label in enumerate(branch_classes):
                matching = np.flatnonzero(self.classes_ == label)
                if matching.size:
                    probabilities[np.ix_(mask, matching)] = branch_probabilities[:, [branch_index]]
        return probabilities
