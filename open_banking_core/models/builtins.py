"""Built-in classifiers, with LightGBM loaded only when requested."""

from sklearn.linear_model import LogisticRegression

from .registry import register_model


@register_model("logistic_regression")
class RegisteredLogisticRegression(LogisticRegression):
    """Logistic Regression available through the model registry."""


@register_model("lightgbm")
class RegisteredLightGBMClassifier:
    """Lazy constructor wrapper for LightGBM's sklearn classifier."""

    def __new__(cls, **params: object) -> object:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("the LightGBM model requires `pip install lightgbm`") from exc
        return LGBMClassifier(**params)
