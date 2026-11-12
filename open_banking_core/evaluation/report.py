"""Structured standard evaluation report."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationReport:
    """Classification scores, per-class breakdown, and amount slices."""

    metrics: dict[str, float]
    labels: list[str]
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]]
    normalized_confusion_matrix: list[list[float]]
    confidence_bins: list[dict[str, float | int]]
    confidence_thresholds: list[dict[str, float | int | None]]
    roc_curves: dict[str, dict[str, Any]]
    precision_recall_curves: dict[str, dict[str, Any]]
    amount_slices: dict[str, dict[str, float | int]]
    permutation_importance: dict[str, dict[str, float]]
    test_seen_description_rate: float | None

    def write_html(self, path: str, *, include_plotlyjs: bool = True) -> None:
        """Write an interactive HTML report; Plotly is an optional extra."""
        from .html_report import write_html_report

        write_html_report(self, path, include_plotlyjs=include_plotlyjs)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable report data."""
        return asdict(self)
