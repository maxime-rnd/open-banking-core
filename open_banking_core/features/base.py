"""Small common interface for registered scikit-learn feature transformers."""

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Base class for transformers that consume a pandas DataFrame."""

