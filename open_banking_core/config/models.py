"""Pydantic configuration models, usable directly or loaded from YAML."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ColumnMapping(BaseModel):
    """Maps canonical transaction fields to input CSV column names."""

    model_config = ConfigDict(extra="forbid")

    operation_date: str
    execution_date: str
    description: str
    amount: str
    account_id: str | None = None


class DataConfig(BaseModel):
    """CSV parsing and column mapping settings."""

    model_config = ConfigDict(extra="forbid")

    columns: ColumnMapping
    target_column: str | None = None
    date_format: str = "%Y-%m-%d"
    encoding: str = "utf-8"
    delimiter: str = ","


class SplitConfig(BaseModel):
    """Settings for reproducible dataset splitting."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = "stratified_random"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 42
    group_by: Literal["normalized_description"] | None = None


class FeatureSpec(BaseModel):
    """Registered feature name and constructor parameters."""

    model_config = ConfigDict(extra="forbid")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    """Registered classifier name and constructor parameters."""

    model_config = ConfigDict(extra="forbid")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineBranchConfig(BaseModel):
    """Features and classifier settings for one transaction sign."""

    model_config = ConfigDict(extra="forbid")

    features: list[FeatureSpec] | None = None
    model: ModelSpec


class PermutationImportanceConfig(BaseModel):
    """Repeated feature-block permutation settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    scoring: Literal["macro_f1", "weighted_f1", "accuracy"] = "macro_f1"
    n_repeats: int = Field(default=5, ge=1)
    random_state: int = 42


class EvaluationConfig(BaseModel):
    """Metrics and absolute amount bins for the standard evaluation report."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(default_factory=lambda: [
        "accuracy", "balanced_accuracy", "weighted_accuracy", "macro_precision", "weighted_precision",
        "macro_recall", "weighted_recall", "macro_f1", "weighted_f1",
        "top_2_accuracy", "log_loss", "brier_score", "ece",
    ])
    amount_bins: list[float] = Field(default_factory=lambda: [0.0, 20.0, 100.0, 1000.0], min_length=2)
    label_weights: dict[str, float] = Field(default_factory=dict)
    confidence_bins: int = Field(default=10, ge=1)
    confidence_thresholds: list[float] = Field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9])
    permutation_importance: PermutationImportanceConfig = Field(default_factory=PermutationImportanceConfig)

    @model_validator(mode="after")
    def validate_amount_bins(self) -> "EvaluationConfig":
        if any(value < 0 for value in self.amount_bins):
            raise ValueError("evaluation amount bins must be non-negative absolute amounts")
        if self.amount_bins[0] != 0:
            raise ValueError("evaluation amount bins must start at zero")
        if self.amount_bins != sorted(set(self.amount_bins)):
            raise ValueError("evaluation amount bins must be strictly increasing")
        if any(weight <= 0 for weight in self.label_weights.values()):
            raise ValueError("evaluation label weights must be positive")
        if any(value < 0 or value > 1 for value in self.confidence_thresholds):
            raise ValueError("confidence thresholds must be between zero and one")
        if self.confidence_thresholds != sorted(set(self.confidence_thresholds)):
            raise ValueError("confidence thresholds must be strictly increasing")
        return self


class CoreConfig(BaseModel):
    """Top-level configuration shared by data and split foundations."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    split: SplitConfig = SplitConfig()
    features: list[FeatureSpec] = Field(default_factory=list)
    pipelines: dict[Literal["debit", "credit"], PipelineBranchConfig] = Field(default_factory=dict)
    evaluation: EvaluationConfig = EvaluationConfig()

    @model_validator(mode="after")
    def validate_pipeline_branches(self) -> "CoreConfig":
        """Require debit and credit branches together when branches are configured."""
        if self.pipelines and set(self.pipelines) != {"debit", "credit"}:
            raise ValueError("pipelines must define both 'debit' and 'credit'")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CoreConfig":
        """Load and validate a configuration from a YAML file."""
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        if not isinstance(payload, dict):
            raise ValueError("configuration YAML must contain a mapping at its root")
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_delimiter(self) -> "CoreConfig":
        """Require a single-character CSV delimiter."""
        if len(self.data.delimiter) != 1:
            raise ValueError("data.delimiter must be exactly one character")
        return self
