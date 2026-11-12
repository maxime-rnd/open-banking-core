"""Row-level validation for canonical transaction data."""

from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from open_banking_core.config import CoreConfig

CANONICAL_COLUMNS = ("operation_date", "execution_date", "description", "amount")


@dataclass(frozen=True)
class ValidationReport:
    """Valid canonical records plus errors keyed by one-based CSV row number."""

    valid_rows: pd.DataFrame
    errors: tuple[dict[str, Any], ...]

    @property
    def rejected_count(self) -> int:
        """Number of rejected source rows."""
        return len(self.errors)


def validate_transactions(data: pd.DataFrame, config: CoreConfig) -> ValidationReport:
    """Validate mapped raw rows and return canonical values and row errors."""
    mapping = config.data.columns
    required = {mapping.description, mapping.amount, mapping.operation_date, mapping.execution_date}
    if config.data.target_column:
        required.add(config.data.target_column)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing required CSV columns: {', '.join(missing)}")

    valid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row_number, (_, row) in enumerate(data.iterrows(), start=2):
        record: dict[str, Any] = {}
        row_errors: list[str] = []
        for canonical in ("operation_date", "execution_date"):
            value = row[mapping.model_dump()[canonical]]
            if pd.isna(value) or str(value).strip() == "":
                record[canonical] = None
                continue
            try:
                parsed = pd.to_datetime(value, format=config.data.date_format, errors="raise")
                record[canonical] = parsed.date()
            except (TypeError, ValueError):
                row_errors.append(f"{canonical} must match {config.data.date_format}")

        description = row[mapping.description]
        if pd.isna(description):
            row_errors.append("description must be a string (empty string is allowed)")
        else:
            record["description"] = str(description)

        if config.data.target_column:
            label = row[config.data.target_column]
            if pd.isna(label) or str(label).strip() == "":
                row_errors.append("target label must not be empty")
            else:
                record["category"] = str(label)

        raw_amount = row[mapping.amount]
        try:
            amount = float(raw_amount)
            if not isfinite(amount):
                raise ValueError
            record["amount"] = amount
        except (TypeError, ValueError):
            row_errors.append("amount must be a finite number")

        if row_errors:
            errors.append({"row": row_number, "errors": row_errors})
        else:
            valid.append(record)

    columns = [*CANONICAL_COLUMNS, *( ["category"] if config.data.target_column else [])]
    return ValidationReport(valid_rows=pd.DataFrame(valid, columns=columns), errors=tuple(errors))
