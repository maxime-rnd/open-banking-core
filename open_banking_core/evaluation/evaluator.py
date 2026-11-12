"""Evaluation metrics, confidence diagnostics, and feature-block importance."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from open_banking_core.config import CoreConfig, EvaluationConfig
from open_banking_core.data.splitting import normalize_description
from open_banking_core.pipelines import SignRouter
from .metrics import compute_metrics
from .report import EvaluationReport


class Evaluator:
    """Calculate classification, confidence, and block-wise diagnostics."""

    def __init__(self, config: CoreConfig | EvaluationConfig, target_column: str | None = None) -> None:
        self.evaluation = config.evaluation if isinstance(config, CoreConfig) else config
        self.target_column = target_column or (
            config.data.target_column if isinstance(config, CoreConfig) else None
        ) or "category"

    def evaluate(
        self,
        router: SignRouter,
        test_data: pd.DataFrame,
        *,
        train_data: pd.DataFrame | None = None,
    ) -> EvaluationReport:
        """Evaluate a fitted router on held-out rows and return a rich report."""
        if test_data.empty:
            raise ValueError("test data must contain at least one row")
        if self.target_column not in test_data:
            raise ValueError(f"evaluation target column {self.target_column!r} is missing")
        if "amount" not in test_data:
            raise ValueError("evaluation data must include the canonical amount column")

        y_true = test_data[self.target_column].to_numpy()
        X_test = test_data.drop(columns=[self.target_column])
        y_pred = router.predict(X_test)
        probabilities = router.predict_proba(X_test)
        labels = list(router.classes_)
        classes = np.asarray(labels)
        confidence = probabilities.max(axis=1)
        probability_labels = classes[probabilities.argmax(axis=1)]

        metrics = self._metrics(y_true, y_pred, probabilities, classes, confidence, probability_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        per_class = {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        }
        matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(int)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
        confidence_bins = self._confidence_bins(y_true, probability_labels, confidence)
        threshold_report = self._threshold_coverage(y_true, probability_labels, confidence)
        amount_slices = self._amount_slices(test_data, y_true, y_pred)
        roc_curves, pr_curves = self._class_curves(y_true, probabilities, classes)
        permutation = self._permutation_importance(router, X_test, y_true)
        seen_rate = self._seen_description_rate(train_data, test_data) if train_data is not None else None

        return EvaluationReport(
            metrics=metrics,
            labels=[str(label) for label in labels],
            per_class=per_class,
            confusion_matrix=matrix.tolist(),
            normalized_confusion_matrix=normalized.tolist(),
            confidence_bins=confidence_bins,
            confidence_thresholds=threshold_report,
            roc_curves=roc_curves,
            precision_recall_curves=pr_curves,
            amount_slices=amount_slices,
            permutation_importance=permutation,
            test_seen_description_rate=seen_rate,
        )

    def _metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: np.ndarray,
        classes: np.ndarray,
        confidence: np.ndarray,
        probability_labels: np.ndarray,
    ) -> dict[str, float]:
        names = self.evaluation.metrics
        metrics: dict[str, float] = {}
        label_indices = {label: index for index, label in enumerate(classes)}
        probability_metrics = {"log_loss", "brier_score", "ece"}
        custom_names = [name for name in names if name not in probability_metrics | {"weighted_accuracy", "top_2_accuracy"}]
        if custom_names:
            metrics.update(compute_metrics(custom_names, y_true, y_pred))
        if "weighted_accuracy" in names:
            sample_weight = np.asarray([self.evaluation.label_weights.get(str(label), 1.0) for label in y_true])
            metrics["weighted_accuracy"] = float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
        if "top_2_accuracy" in names:
            top_k = min(2, probabilities.shape[1])
            top = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
            true_indices = np.asarray([label_indices[label] for label in y_true])
            metrics["top_2_accuracy"] = float(np.mean([truth in candidates for truth, candidates in zip(true_indices, top)]))
        if "log_loss" in names:
            metrics["log_loss"] = float(log_loss(y_true, probabilities, labels=classes))
        if "brier_score" in names:
            one_hot = np.zeros_like(probabilities)
            one_hot[np.arange(len(y_true)), [label_indices[label] for label in y_true]] = 1.0
            metrics["brier_score"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        if "ece" in names:
            metrics["ece"] = self._ece(y_true, probability_labels, confidence, self.evaluation.confidence_bins)
        return metrics

    def _confidence_bins(
        self, y_true: np.ndarray, predictions: np.ndarray, confidence: np.ndarray
    ) -> list[dict[str, float | int]]:
        bins = self.evaluation.confidence_bins
        bin_indices = np.minimum((confidence * bins).astype(int), bins - 1)
        output = []
        for index in range(bins):
            mask = bin_indices == index
            if not mask.any():
                continue
            output.append({
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()),
                "accuracy": float(np.mean(y_true[mask] == predictions[mask])),
            })
        return output

    @staticmethod
    def _ece(y_true: np.ndarray, predictions: np.ndarray, confidence: np.ndarray, bins: int | None = None) -> float:
        # Use equal-width confidence bins and weight each calibration gap by coverage.
        n_bins = bins or 10
        indices = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
        return float(sum(
            (mask.sum() / len(y_true)) * abs(float(np.mean(y_true[mask] == predictions[mask])) - float(confidence[mask].mean()))
            for index in range(n_bins)
            if (mask := indices == index).any()
        ))

    def _threshold_coverage(
        self, y_true: np.ndarray, predictions: np.ndarray, confidence: np.ndarray
    ) -> list[dict[str, float | int | None]]:
        output = []
        for threshold in self.evaluation.confidence_thresholds:
            mask = confidence >= threshold
            count = int(mask.sum())
            output.append({
                "threshold": float(threshold),
                "coverage": float(count / len(y_true)),
                "count": count,
                "accuracy": float(np.mean(y_true[mask] == predictions[mask])) if count else None,
            })
        return output

    def _amount_slices(
        self, data: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, dict[str, float | int]]:
        absolute_amounts = np.abs(pd.to_numeric(data["amount"], errors="raise").to_numpy(dtype=float))
        if not np.isfinite(absolute_amounts).all():
            raise ValueError("evaluation amounts must be finite numbers")
        edges = self.evaluation.amount_bins
        groups = np.digitize(absolute_amounts, edges[1:], right=False)
        output: dict[str, dict[str, float | int]] = {}
        for group_index in range(len(edges)):
            mask = groups == group_index
            if not mask.any():
                continue
            if group_index == 0:
                name = f"<{edges[1]:g}"
            elif group_index == len(edges) - 1:
                name = f">={edges[-1]:g}"
            else:
                name = f"[{edges[group_index]:g}, {edges[group_index + 1]:g})"
            output[name] = {
                "count": int(mask.sum()),
                "accuracy": float(np.mean(y_true[mask] == y_pred[mask])),
                "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
            }
        return output

    @staticmethod
    def _class_curves(
        y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        roc_curves: dict[str, dict[str, Any]] = {}
        pr_curves: dict[str, dict[str, Any]] = {}
        for index, label in enumerate(classes):
            binary_truth = y_true == label
            positives = int(binary_truth.sum())
            negatives = len(binary_truth) - positives
            if positives == 0 or negatives == 0:
                continue
            fpr, tpr, _ = roc_curve(binary_truth, probabilities[:, index])
            precision, recall, _ = precision_recall_curve(binary_truth, probabilities[:, index])
            roc_curves[str(label)] = {
                "false_positive_rate": fpr.tolist(),
                "true_positive_rate": tpr.tolist(),
                "auc": float(roc_auc_score(binary_truth, probabilities[:, index])),
            }
            pr_curves[str(label)] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "average_precision": float(average_precision_score(binary_truth, probabilities[:, index])),
            }
        return roc_curves, pr_curves

    def _permutation_importance(
        self, router: SignRouter, X: pd.DataFrame, y_true: np.ndarray
    ) -> dict[str, dict[str, float]]:
        settings = self.evaluation.permutation_importance
        if not settings.enabled:
            return {}
        rng = np.random.default_rng(settings.random_state)
        results: dict[str, dict[str, float]] = {}
        sign_masks = {"debit": X.amount.to_numpy(dtype=float) <= 0, "credit": X.amount.to_numpy(dtype=float) > 0}
        for sign, sign_mask in sign_masks.items():
            if not sign_mask.any():
                continue
            branch = router.debit_pipeline_ if sign == "debit" else router.credit_pipeline_
            sign_X = X.iloc[sign_mask]
            sign_y = y_true[sign_mask]
            union = branch.named_steps["features"]
            classifier = branch.named_steps["classifier"]
            blocks: list[tuple[str, Any]] = []
            for feature_name, transformer in union.transformer_list:
                if hasattr(transformer, "transform_components"):
                    components = transformer.transform_components(sign_X)
                    blocks.extend((f"{feature_name}.{component_name}", matrix) for component_name, matrix in components.items())
                else:
                    blocks.append((feature_name, transformer.transform(sign_X)))
            baseline_features = self._stack([matrix for _, matrix in blocks])
            baseline_predictions = classifier.predict(baseline_features)
            baseline_score = self._importance_score(settings.scoring, sign_y, baseline_predictions)
            for block_index, (block_name, matrix) in enumerate(blocks):
                drops = []
                for _ in range(settings.n_repeats):
                    permutation = rng.permutation(len(sign_y))
                    changed_blocks = [
                        value[permutation] if current_index == block_index else value
                        for current_index, (_, value) in enumerate(blocks)
                    ]
                    prediction = classifier.predict(self._stack(changed_blocks))
                    drops.append(baseline_score - self._importance_score(settings.scoring, sign_y, prediction))
                results[f"{sign}.{block_name}"] = {
                    "mean": float(np.mean(drops)),
                    "std": float(np.std(drops)),
                }
        return results

    @staticmethod
    def _stack(blocks: list[Any]) -> Any:
        matrices = [matrix if sparse.issparse(matrix) else sparse.csr_matrix(matrix) for matrix in blocks]
        return sparse.hstack(matrices, format="csr") if len(matrices) > 1 else matrices[0]

    @staticmethod
    def _importance_score(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        average = "macro" if name == "macro_f1" else "weighted"
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))

    @staticmethod
    def _seen_description_rate(train_data: pd.DataFrame, test_data: pd.DataFrame) -> float:
        if "description" not in train_data or "description" not in test_data:
            raise ValueError("train and test data need descriptions to compute seen-description rate")
        known = {normalize_description(value) for value in train_data.description}
        return float(sum(normalize_description(value) in known for value in test_data.description) / len(test_data))
