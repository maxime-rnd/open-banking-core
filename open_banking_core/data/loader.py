"""CSV loading with explicit column mapping and row-level validation."""

from pathlib import Path

import pandas as pd

from open_banking_core.config import CoreConfig
from .validation import ValidationReport, validate_transactions


def load_transactions(path: str | Path, config: CoreConfig) -> ValidationReport:
    """Read a CSV and validate it, returning valid rows and row errors.

    Unknown columns, including ``account_id``, are ignored after reading.
    The caller can inspect rejected rows through the returned report.
    """
    raw = pd.read_csv(
        path,
        sep=config.data.delimiter,
        encoding=config.data.encoding,
        dtype={config.data.columns.description: "string"},
        # Preserve empty descriptions as ""; blank numeric values are rejected
        # by validation and blank dates are interpreted as absent.
        keep_default_na=False,
    )
    report = validate_transactions(raw, config)
    if raw.empty or report.valid_rows.empty:
        raise ValueError("CSV contains no valid transaction rows")
    return report
