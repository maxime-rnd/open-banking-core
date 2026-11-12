"""Interactive, standalone HTML report rendering with Plotly."""

from pathlib import Path
from typing import Any

from .report import EvaluationReport


def _figures(report: EvaluationReport) -> list[tuple[str, Any]]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("HTML reports require the optional extra: pip install open-banking-core[reports]") from exc

    figures: list[tuple[str, Any]] = []
    labels = report.labels

    confusion = make_subplots(1, 2, subplot_titles=("Confusion matrix", "Row-normalized confusion matrix"))
    confusion.add_trace(go.Heatmap(z=report.confusion_matrix, x=labels, y=labels, colorscale="Blues", showscale=False), row=1, col=1)
    confusion.add_trace(go.Heatmap(z=report.normalized_confusion_matrix, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1, showscale=True), row=1, col=2)
    confusion.update_layout(title="Confusion matrices", height=520)
    confusion.update_xaxes(title_text="Predicted label")
    confusion.update_yaxes(title_text="True label", autorange="reversed")
    figures.append(("Confusion matrices", confusion))

    per_class = go.Figure()
    for metric in ("precision", "recall", "f1"):
        per_class.add_trace(go.Bar(name=metric, x=list(report.per_class), y=[report.per_class[label][metric] for label in report.per_class]))
    per_class.update_layout(title="Per-class precision, recall, and F1", barmode="group", yaxis_range=[0, 1])
    figures.append(("Per-class metrics", per_class))

    calibration = make_subplots(specs=[[{"secondary_y": True}]])
    calibration.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", mode="lines", line={"dash": "dash"}), secondary_y=False)
    calibration.add_trace(
        go.Scatter(
            x=[item["mean_confidence"] for item in report.confidence_bins],
            y=[item["accuracy"] for item in report.confidence_bins],
            mode="lines+markers",
            name="Observed accuracy",
            text=[f"n={item['count']}" for item in report.confidence_bins],
        ),
        secondary_y=False,
    )
    calibration.add_trace(go.Bar(x=[item["mean_confidence"] for item in report.confidence_bins], y=[item["count"] for item in report.confidence_bins], name="Samples per bin", opacity=0.25), secondary_y=True)
    calibration.update_layout(title="Confidence calibration", barmode="overlay")
    calibration.update_xaxes(title_text="Mean predicted confidence", range=[0, 1])
    calibration.update_yaxes(title_text="Accuracy", range=[0, 1], secondary_y=False)
    calibration.update_yaxes(title_text="Samples", secondary_y=True)
    figures.append(("Confidence calibration", calibration))

    coverage = make_subplots(specs=[[{"secondary_y": True}]])
    coverage.add_trace(go.Scatter(x=[item["threshold"] for item in report.confidence_thresholds], y=[item["coverage"] for item in report.confidence_thresholds], mode="lines+markers", name="Coverage"), secondary_y=False)
    coverage.add_trace(go.Scatter(x=[item["threshold"] for item in report.confidence_thresholds], y=[item["accuracy"] for item in report.confidence_thresholds], mode="lines+markers", name="Accuracy of covered predictions"), secondary_y=True)
    coverage.update_layout(title="Selective prediction: coverage and accuracy")
    coverage.update_xaxes(title_text="Minimum confidence threshold", range=[0, 1])
    coverage.update_yaxes(title_text="Coverage", range=[0, 1], secondary_y=False)
    coverage.update_yaxes(title_text="Accuracy among covered rows", range=[0, 1], secondary_y=True)
    figures.append(("Coverage and accuracy by threshold", coverage))

    amount = go.Figure()
    amount.add_trace(go.Bar(name="Accuracy", x=list(report.amount_slices), y=[value["accuracy"] for value in report.amount_slices.values()]))
    amount.add_trace(go.Bar(name="Macro F1", x=list(report.amount_slices), y=[value["macro_f1"] for value in report.amount_slices.values()]))
    amount.update_layout(title="Performance by absolute amount", barmode="group", yaxis_range=[0, 1])
    figures.append(("Amount slices", amount))

    if report.permutation_importance:
        ordered = sorted(report.permutation_importance.items(), key=lambda item: item[1]["mean"])
        importance = go.Figure(go.Bar(
            x=[value["mean"] for _, value in ordered],
            y=[name for name, _ in ordered],
            orientation="h",
            error_x={"type": "data", "array": [value["std"] for _, value in ordered]},
        ))
        importance.update_layout(title="Block-wise permutation importance", xaxis_title="Decrease in configured score (larger means more important)")
        figures.append(("Permutation importance", importance))

    if report.roc_curves:
        roc = go.Figure()
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", mode="lines", line={"dash": "dash"}))
        for label, values in report.roc_curves.items():
            roc.add_trace(go.Scatter(x=values["false_positive_rate"], y=values["true_positive_rate"], name=f"{label} (AUC {values['auc']:.3f})"))
        roc.update_layout(title="One-vs-rest ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        figures.append(("ROC curves", roc))

    if report.precision_recall_curves:
        pr = go.Figure()
        for label, values in report.precision_recall_curves.items():
            pr.add_trace(go.Scatter(x=values["recall"], y=values["precision"], name=f"{label} (AP {values['average_precision']:.3f})"))
        pr.update_layout(title="One-vs-rest precision-recall curves", xaxis_title="Recall", yaxis_title="Precision")
        figures.append(("Precision-recall curves", pr))
    return figures


def write_html_report(report: EvaluationReport, path: str | Path, *, include_plotlyjs: bool = True) -> None:
    """Render report metrics and interactive charts to one HTML document."""
    from html import escape

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metric_cells = "".join(
        f"<div class='metric'><span>{escape(name)}</span><strong>{value:.4f}</strong></div>"
        for name, value in report.metrics.items()
    )
    seen = "n/a" if report.test_seen_description_rate is None else f"{report.test_seen_description_rate:.1%}"
    chart_parts = []
    for index, (title, figure) in enumerate(_figures(report)):
        js_option: bool | str = True if include_plotlyjs and index == 0 else ("cdn" if not include_plotlyjs and index == 0 else False)
        chart_parts.append(
            f"<section><h2>{escape(title)}</h2>"
            f"{figure.to_html(full_html=False, include_plotlyjs=js_option, config={'responsive': True})}</section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open Banking evaluation report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1280px; margin: 2rem auto; padding: 0 1rem; color: #172033; background: #f7f9fc; }}
h1 {{ margin-bottom: .25rem; }} .subtitle {{ color: #5b6576; margin-bottom: 1.5rem; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .75rem; }}
.metric {{ background: white; border: 1px solid #dce2eb; border-radius: 10px; padding: 1rem; display: flex; flex-direction: column; gap: .4rem; }}
.metric span {{ color: #5b6576; }} .metric strong {{ font-size: 1.5rem; }}
section {{ background: white; border: 1px solid #dce2eb; border-radius: 10px; padding: 1rem; margin: 1rem 0; overflow: hidden; }}
h2 {{ font-size: 1.15rem; margin: .2rem 0 .75rem; }}
</style></head><body>
<h1>Evaluation report</h1><p class="subtitle">Test descriptions already seen in train: {seen}</p>
<div class="metrics">{metric_cells}</div>{''.join(chart_parts)}
</body></html>"""
    target.write_text(document, encoding="utf-8")
