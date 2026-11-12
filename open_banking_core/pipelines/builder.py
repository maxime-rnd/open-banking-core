"""Build a single sklearn classifier pipeline from typed configuration."""

from sklearn.pipeline import FeatureUnion, Pipeline

from open_banking_core.config import CoreConfig, PipelineBranchConfig
from open_banking_core.features import build_feature
from open_banking_core.models import build_model


def build_pipeline(config: PipelineBranchConfig, *, default_features: list | None = None) -> Pipeline:
    """Build a FeatureUnion plus classifier for one sign-specific branch."""
    specs = config.features if config.features is not None else default_features or []
    if not specs:
        raise ValueError("a pipeline branch must configure at least one feature")
    union = FeatureUnion(
        transformer_list=[(f"feature_{index}_{spec.type}", build_feature(spec)) for index, spec in enumerate(specs)]
    )
    return Pipeline([("features", union), ("classifier", build_model(config.model))])


def build_sign_router(config: CoreConfig) -> "SignRouter":
    """Build independent debit and credit pipelines from a CoreConfig."""
    from .router import SignRouter

    if set(config.pipelines) != {"debit", "credit"}:
        raise ValueError("CoreConfig must define both debit and credit pipelines")
    return SignRouter(
        debit_pipeline=build_pipeline(config.pipelines["debit"], default_features=config.features),
        credit_pipeline=build_pipeline(config.pipelines["credit"], default_features=config.features),
    )
