"""Canonical transaction input schema."""

from datetime import date
from math import isfinite

from pydantic import BaseModel, ConfigDict, field_validator


class Transaction(BaseModel):
    """A single transaction using the core's canonical field names."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    operation_date: date | None = None
    execution_date: date | None = None
    description: str
    amount: float

    @field_validator("description")
    @classmethod
    def description_must_not_be_null(cls, value: str) -> str:
        """Allow an empty description while requiring a string value."""
        return value

    @field_validator("amount")
    @classmethod
    def amount_must_be_finite(cls, value: float) -> float:
        """Reject NaN and infinities as they cannot be routed reliably."""
        if not isfinite(value):
            raise ValueError("amount must be a finite number")
        return value
