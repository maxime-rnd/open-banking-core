"""Boolean text features based on named regular expressions."""

from typing import Any

import numpy as np
import pandas as pd

from .base import FeatureTransformer
from .registry import register_feature
from .text import clean_text


@register_feature("regex")
class RegexFeatures(FeatureTransformer):
    """Create one binary feature column per configured regular expression."""

    def __init__(
        self,
        column: str = "description",
        patterns: dict[str, str] | None = None,
        clean: dict[str, Any] | None = None,
        case_insensitive: bool = True,
    ) -> None:
        self.column = column
        self.patterns = patterns
        self.clean = clean
        self.case_insensitive = case_insensitive

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RegexFeatures":
        import re

        self.pattern_names_ = list((self.patterns or {}).keys())
        flags = re.IGNORECASE if self.case_insensitive else 0
        self.compiled_patterns_ = [re.compile(pattern, flags) for pattern in (self.patterns or {}).values()]
        if self.column not in X:
            raise ValueError(f"regex source column {self.column!r} is missing")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        values = [clean_text(value, self.clean) for value in X[self.column].tolist()]
        return np.asarray(
            [[bool(pattern.search(value)) for pattern in self.compiled_patterns_] for value in values],
            dtype=np.float32,
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.pattern_names_, dtype=object)
