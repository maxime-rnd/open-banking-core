"""Tests for phase-one schemas, configuration, CSV validation, and splitting."""

import tempfile
import unittest
import sys
import types
import json
import contextlib
import io
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from open_banking_core import CoreConfig, SplitConfig, Transaction, load_transactions, split_by_sign, validate_transactions
from open_banking_core.data import split_dataset
from open_banking_core.config import (
    ColumnMapping,
    DataConfig,
    EvaluationConfig,
    FeatureSpec,
    ModelSpec,
    PermutationImportanceConfig,
    PipelineBranchConfig,
)
from open_banking_core.features import (
    AmountFeatures,
    DateFeatures,
    RegexFeatures,
    TextFeature,
    build_embedding_provider,
    build_feature,
    clean_text,
    register_embedding_provider,
    register_feature,
)
from open_banking_core.features.base import FeatureTransformer
from open_banking_core.models import build_model, register_model, registered_models
from open_banking_core.pipelines import SignRouter, build_pipeline, build_sign_router
from open_banking_core.training import Trainer
from open_banking_core.evaluation import Evaluator, register_metric
from open_banking_core.cli import main as cli_main
from sklearn.dummy import DummyClassifier


def example_config() -> CoreConfig:
    return CoreConfig(
        data=DataConfig(
            columns=ColumnMapping(
                operation_date="booking_date",
                execution_date="value_date",
                description="label",
                amount="amount",
                account_id="account_id",
            ),
            target_column="category",
            date_format="%Y-%m-%d",
        ),
        split=SplitConfig(test_size=0.25, random_state=7),
        features=[FeatureSpec(type="amount", params={"column": "amount"})],
    )


class TransactionSchemaTests(unittest.TestCase):
    def test_empty_description_and_optional_dates_are_allowed(self) -> None:
        transaction = Transaction(description="", amount=-1.5)
        self.assertEqual(transaction.description, "")
        self.assertIsNone(transaction.operation_date)

    def test_null_amount_and_description_are_rejected(self) -> None:
        for payload in (
            {"description": "x", "amount": None},
            {"description": None, "amount": 1},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                Transaction.model_validate(payload)


class ConfigTests(unittest.TestCase):
    def test_yaml_and_python_construction_are_equivalent(self) -> None:
        yaml_text = """\
data:
  columns:
    operation_date: booking_date
    execution_date: value_date
    description: label
    amount: amount
    account_id: account_id
  target_column: category
  date_format: '%Y-%m-%d'
split:
  strategy: stratified_random
  test_size: 0.25
  random_state: 7
  group_by: null
features:
  - type: amount
    params:
      column: amount
"""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            self.assertEqual(CoreConfig.from_yaml(path), example_config())


class DataTests(unittest.TestCase):
    def test_validation_preserves_empty_description_and_reports_bad_rows(self) -> None:
        raw = pd.DataFrame(
            {
                "booking_date": ["2026-11-01", "bad-date"],
                "value_date": ["", "2026-11-03"],
                "label": ["", "invalid date"],
                "amount": ["-3.0", "2.0"],
                "category": ["groceries", "refund"],
                "account_id": ["ignored", "ignored"],
            }
        )
        report = validate_transactions(raw, example_config())
        self.assertEqual(len(report.valid_rows), 1)
        self.assertEqual(report.valid_rows.iloc[0].description, "")
        self.assertIsNone(report.valid_rows.iloc[0].execution_date)
        self.assertEqual(report.errors[0]["row"], 3)

    def test_csv_loader_ignores_account_id_and_preserves_empty_text(self) -> None:
        csv_text = "booking_date,value_date,label,amount,category,account_id\n2026-11-01,,,-3,groceries,abc\n"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "transactions.csv"
            path.write_text(csv_text, encoding="utf-8")
            report = load_transactions(path, example_config())
        self.assertEqual(report.valid_rows.iloc[0].description, "")

    def test_splits_are_independent_by_sign_and_seeded(self) -> None:
        rows = []
        for sign, category in ((-1, "debit"), (1, "credit")):
            for index in range(8):
                rows.append({"amount": sign * (index + 1), "category": category, "description": f"tx {index}"})
        data = pd.DataFrame(rows)
        first = split_by_sign(data, "category", example_config())
        second = split_by_sign(data, "category", example_config())
        self.assertEqual(set(first), {"debit", "credit"})
        for key in first:
            self.assertEqual(list(first[key].test.index), list(second[key].test.index))
            self.assertEqual(set(first[key].train.category), {key})
            self.assertEqual(set(first[key].test.category), {key})

    def test_grouped_split_keeps_normalized_descriptions_apart(self) -> None:
        data = pd.DataFrame(
            [
                {"amount": -1, "category": "debit", "description": f"Merchant {index}"}
                for index in range(8)
                for _ in range(2)
            ]
        )
        config = example_config().model_copy(
            update={"split": SplitConfig(test_size=0.25, random_state=7, group_by="normalized_description")}
        )
        result = split_dataset(data, "category", config)
        self.assertEqual(result.test_seen_description_rate, 0.0)

    def test_example_csv_is_filtered_by_sign_before_train_test_split(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        report = load_transactions(project_root / "examples" / "transactions.csv", example_config())
        splits = split_by_sign(report.valid_rows, "category", example_config())
        self.assertEqual(set(splits), {"debit", "credit"})
        self.assertTrue((splits["debit"].train.amount <= 0).all())
        self.assertTrue((splits["debit"].test.amount <= 0).all())
        self.assertTrue((splits["credit"].train.amount > 0).all())
        self.assertTrue((splits["credit"].test.amount > 0).all())
        for split in splits.values():
            self.assertTrue(set(split.train.index).isdisjoint(split.test.index))


class FeatureTests(unittest.TestCase):
    def test_text_cleaning_and_tfidf_representation(self) -> None:
        self.assertEqual(clean_text("  CAFÉ   #12 "), "café #12")
        data = pd.DataFrame({"description": ["CARD PAYMENT ALPHA", "SALARY CREDIT BETA"]})
        feature = TextFeature(
            clean={"unicode_normalization": "NFKD", "lowercase": True},
            representation={"type": "tfidf", "tfidf": {"ngram_range": (1, 2)}},
        ).fit(data)
        matrix = feature.transform(data)
        self.assertEqual(matrix.shape[0], 2)
        self.assertGreater(matrix.shape[1], 2)

    def test_custom_embedding_provider_and_hybrid_representation(self) -> None:
        @register_embedding_provider("test-deterministic")
        class DeterministicProvider:
            def __init__(self, width: int = 2) -> None:
                self.width = width

            def embed(self, texts: list[str]) -> pd.DataFrame:
                return pd.DataFrame([[len(text), 1] for text in texts]).to_numpy(dtype=float)

        data = pd.DataFrame({"description": ["market alpha", "salary beta"]})
        feature = TextFeature(
            representation={
                "type": "hybrid",
                "tfidf": {"min_df": 1},
                "embedding": {"provider": "test-deterministic", "width": 2},
            }
        ).fit(data)
        matrix = feature.transform(data)
        self.assertEqual(matrix.shape[0], 2)
        self.assertEqual(len(feature.get_feature_names_out()), matrix.shape[1])
        self.assertEqual(build_embedding_provider("test-deterministic", width=3).width, 3)

    def test_vllm_provider_uses_openai_client_and_configured_base_url(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                calls.append({"client": kwargs})
                self.embeddings = self

            def create(self, **kwargs: object) -> types.SimpleNamespace:
                calls.append(kwargs)
                values = kwargs["input"]
                return types.SimpleNamespace(
                    data=[types.SimpleNamespace(index=index, embedding=[float(len(value)), 1.0]) for index, value in enumerate(values)]
                )

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeClient  # type: ignore[attr-defined]
        provider = build_embedding_provider(
            "vllm",
            model="internal-model",
            base_url="https://vllm.internal/v1",
            api_key_env="TEST_VLLM_KEY",
            batch_size=8,
        )
        with patch.dict(sys.modules, {"openai": fake_openai}), patch.dict("os.environ", {"TEST_VLLM_KEY": "secret"}):
            vectors = provider.embed(["statement", ""])
        self.assertEqual(vectors.shape, (2, 2))
        self.assertEqual(vectors[1].tolist(), [0.0, 0.0])
        self.assertEqual(calls[0]["client"]["base_url"], "https://vllm.internal/v1")
        self.assertEqual(calls[1]["model"], "internal-model")

    def test_builtin_feature_transformers_and_custom_registration(self) -> None:
        data = pd.DataFrame(
            {
                "description": ["IBAN FR761234", "CARD PAYMENT"],
                "amount": [-24.5, 0.0],
                "operation_date": ["2026-01-01", None],
                "execution_date": ["2026-01-03", "2026-01-02"],
            }
        )
        regex = RegexFeatures(patterns={"has_iban": r"\bIBAN\b"}).fit(data)
        self.assertEqual(regex.transform(data).tolist(), [[1.0], [0.0]])
        amount = AmountFeatures().fit(data)
        self.assertEqual(amount.transform(data).shape, (2, 3))
        dates = DateFeatures().fit(data)
        self.assertEqual(dates.transform(data)[0, 2], 2.0)

        @register_feature("test-custom-feature")
        class CustomFeature(FeatureTransformer):
            def __init__(self, column: str) -> None:
                self.column = column

            def fit(self, X: pd.DataFrame, y: object = None) -> "CustomFeature":
                return self

            def transform(self, X: pd.DataFrame) -> pd.Series:
                return X[self.column]

        custom = build_feature({"type": "test-custom-feature", "params": {"column": "amount"}})
        self.assertIsInstance(custom, CustomFeature)


def sign_pipeline_config() -> CoreConfig:
    text = FeatureSpec(
        type="text",
        params={
            "column": "description",
            "representation": {"type": "tfidf", "tfidf": {"min_df": 1}},
        },
    )
    amount = FeatureSpec(type="amount", params={"column": "amount"})
    branch = lambda: PipelineBranchConfig(
        features=[text, amount],
        model=ModelSpec(type="logistic_regression", params={"max_iter": 500, "random_state": 11}),
    )
    return CoreConfig(
        data=DataConfig(
            columns=ColumnMapping(
                operation_date="booking_date",
                execution_date="value_date",
                description="description",
                amount="amount",
            )
        ),
        pipelines={"debit": branch(), "credit": branch()},
    )


class PipelineTests(unittest.TestCase):
    def test_model_registry_builds_core_and_user_models(self) -> None:
        @register_model("test_dummy_classifier")
        class RegisteredDummyClassifier(DummyClassifier):
            pass

        model = build_model({"type": "test_dummy_classifier", "params": {"strategy": "most_frequent"}})
        self.assertIsInstance(model, RegisteredDummyClassifier)
        self.assertIn("lightgbm", registered_models())

    def test_builder_creates_feature_union_and_classifier_pipeline(self) -> None:
        config = sign_pipeline_config()
        pipeline = build_pipeline(config.pipelines["debit"])
        self.assertEqual([name for name, _ in pipeline.steps], ["features", "classifier"])

    def test_example_yaml_builds_lightgbm_and_local_embedding_branches(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = CoreConfig.from_yaml(project_root / "examples" / "pipeline_config.yaml")
        router = build_sign_router(config)
        self.assertEqual(set(config.pipelines), {"debit", "credit"})
        self.assertEqual(type(router.debit_pipeline.named_steps["classifier"]).__name__, "LGBMClassifier")
        self.assertEqual(type(router.credit_pipeline.named_steps["classifier"]).__name__, "RegisteredLogisticRegression")
        text_feature = router.credit_pipeline.named_steps["features"].transformer_list[0][1]
        self.assertEqual(text_feature.representation["type"], "embedding")

    def test_lightgbm_classifier_pipeline_fits(self) -> None:
        config = PipelineBranchConfig(
            features=[FeatureSpec(type="amount", params={"column": "amount"})],
            model=ModelSpec(type="lightgbm", params={"n_estimators": 5, "verbosity": -1, "random_state": 4}),
        )
        pipeline = build_pipeline(config)
        X = pd.DataFrame({"amount": [-4, -3, -2, -1, 1, 2, 3, 4]})
        y = ["debit-a"] * 4 + ["credit-a"] * 4
        pipeline.fit(X, y)
        self.assertEqual(pipeline.predict(X).shape, (8,))

    def test_sign_router_fits_independent_models_and_preserves_prediction_order(self) -> None:
        debit_descriptions = ["market groceries", "supermarket groceries", "monthly rent", "apartment rent"]
        credit_descriptions = ["employer payroll", "monthly salary", "merchant refund", "card refund"]
        rows = []
        labels = []
        for index, (description, category) in enumerate(zip(debit_descriptions, ["groceries", "groceries", "rent", "rent"])):
            rows.append({"description": description, "amount": -(index + 1)})
            labels.append(category)
        for index, (description, category) in enumerate(zip(credit_descriptions, ["salary", "salary", "refund", "refund"])):
            rows.append({"description": description, "amount": index + 1})
            labels.append(category)
        X = pd.DataFrame(rows)
        y = pd.Series(labels)

        router = build_sign_router(sign_pipeline_config())
        self.assertIsInstance(router, SignRouter)
        router.fit(X, y)
        predictions = router.predict(X.iloc[[5, 0, 7, 3]])

        self.assertEqual(len(predictions), 4)
        debit_vocab = router.debit_pipeline_.named_steps["features"].transformer_list[0][1].vectorizer_.vocabulary_
        credit_vocab = router.credit_pipeline_.named_steps["features"].transformer_list[0][1].vectorizer_.vocabulary_
        self.assertIn("groceries", debit_vocab)
        self.assertNotIn("salary", debit_vocab)
        self.assertIn("salary", credit_vocab)
        self.assertNotIn("groceries", credit_vocab)

    def test_sign_router_routes_zero_to_debit_and_rejects_invalid_amount(self) -> None:
        config = sign_pipeline_config()
        pipeline = build_pipeline(config.pipelines["debit"])
        router = SignRouter(pipeline, build_pipeline(config.pipelines["credit"]))
        X = pd.DataFrame(
            [
                {"description": "market groceries", "amount": 0},
                {"description": "employer payroll", "amount": 1},
            ]
        )
        y = ["groceries", "salary"]
        router.fit(pd.concat([X, pd.DataFrame([
            {"description": "monthly rent", "amount": -2},
            {"description": "merchant refund", "amount": 2},
        ])], ignore_index=True), [*y, "rent", "refund"])
        self.assertEqual(router.predict(X).shape, (2,))
        with self.assertRaises(ValueError):
            router.predict(pd.DataFrame([{"description": "unknown", "amount": float("nan")}]))

    def test_train_test_flow_splits_by_sign_before_router_fit(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data_config = example_config()
        valid = load_transactions(project_root / "examples" / "transactions.csv", data_config).valid_rows
        sign_splits = split_by_sign(valid, target="category", config=data_config.split)
        train = pd.concat([sign_splits["debit"].train, sign_splits["credit"].train]).sort_index()
        test = pd.concat([sign_splits["debit"].test, sign_splits["credit"].test]).sort_index()

        router = build_sign_router(sign_pipeline_config())
        router.fit(train.drop(columns="category"), train["category"])
        predictions = router.predict(test.drop(columns="category"))

        self.assertEqual(len(predictions), len(test))
        self.assertTrue(set(train.index).isdisjoint(test.index))


class TrainingEvaluationTests(unittest.TestCase):
    def _prepared_example(self) -> tuple[CoreConfig, pd.DataFrame, pd.DataFrame]:
        project_root = Path(__file__).resolve().parents[1]
        config = CoreConfig.from_yaml(project_root / "examples" / "simple_training.yaml")
        data = load_transactions(project_root / "examples" / "transactions.csv", config).valid_rows
        sign_splits = split_by_sign(data, "category", config.split)
        train = pd.concat([sign_splits["debit"].train, sign_splits["credit"].train]).sort_index()
        test = pd.concat([sign_splits["debit"].test, sign_splits["credit"].test]).sort_index()
        return config, train, test

    def test_trainer_and_evaluator_return_standard_report(self) -> None:
        config, train, test = self._prepared_example()
        router = Trainer(config).fit(train)
        report = Evaluator(config).evaluate(router, test, train_data=train)
        output = report.to_dict()
        self.assertEqual(
            set(output["metrics"]),
            {
                "accuracy", "balanced_accuracy", "weighted_accuracy", "macro_precision", "weighted_precision",
                "macro_recall", "weighted_recall", "macro_f1", "weighted_f1", "top_2_accuracy",
                "log_loss", "brier_score", "ece",
            },
        )
        self.assertEqual(len(output["confusion_matrix"]), 4)
        self.assertEqual(len(output["normalized_confusion_matrix"]), 4)
        self.assertEqual(output["test_seen_description_rate"], 0.0)
        self.assertTrue(output["amount_slices"])
        self.assertTrue(output["confidence_bins"])
        self.assertEqual(len(output["confidence_thresholds"]), 5)
        self.assertTrue(output["roc_curves"])
        self.assertTrue(output["precision_recall_curves"])
        self.assertIn("debit.feature_0_text.tfidf", output["permutation_importance"])
        self.assertIn("credit.feature_1_amount", output["permutation_importance"])

    def test_custom_metric_registration(self) -> None:
        @register_metric("test_exact_match")
        def exact_match(y_true: object, y_pred: object) -> float:
            return float((y_true == y_pred).mean())

        config, train, test = self._prepared_example()
        config.evaluation.metrics = ["test_exact_match"]
        router = Trainer(config).fit(train)
        report = Evaluator(config).evaluate(router, test)
        self.assertIn("test_exact_match", report.metrics)

    def test_hybrid_text_permutation_reports_tfidf_and_embedding_separately(self) -> None:
        @register_embedding_provider("test-hybrid-evaluation")
        class EvaluationEmbeddingProvider:
            def __init__(self, width: int = 2) -> None:
                self.width = width

            def embed(self, texts: list[str]) -> pd.DataFrame:
                return pd.DataFrame([[len(text), sum(map(ord, text)) % 7] for text in texts]).to_numpy(dtype=float)

        base = sign_pipeline_config()
        hybrid = FeatureSpec(
            type="text",
            params={
                "column": "description",
                "representation": {
                    "type": "hybrid",
                    "tfidf": {"min_df": 1},
                    "embedding": {"provider": "test-hybrid-evaluation"},
                },
            },
        )
        branches = {
            sign: PipelineBranchConfig(
                features=[hybrid, FeatureSpec(type="amount", params={"column": "amount"})],
                model=branch.model,
            )
            for sign, branch in base.pipelines.items()
        }
        config = CoreConfig(
            data=base.data,
            split=SplitConfig(test_size=0.25, random_state=7),
            pipelines=branches,
            evaluation=EvaluationConfig(
                permutation_importance=PermutationImportanceConfig(n_repeats=1)
            ),
        )
        project_root = Path(__file__).resolve().parents[1]
        data = load_transactions(project_root / "examples" / "transactions.csv", example_config()).valid_rows
        splits = split_by_sign(data, "category", config.split)
        train = pd.concat([splits["debit"].train, splits["credit"].train]).sort_index()
        test = pd.concat([splits["debit"].test, splits["credit"].test]).sort_index()
        router = Trainer(config).fit(train)
        report = Evaluator(config).evaluate(router, test, train_data=train)
        self.assertIn("debit.feature_0_text.tfidf", report.permutation_importance)
        self.assertIn("debit.feature_0_text.embedding", report.permutation_importance)

    def test_train_and_evaluate_cli_round_trip(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "examples" / "simple_training.yaml"
        data_path = project_root / "examples" / "transactions.csv"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            run_path = Path(directory) / "run.joblib"
            report_path = Path(directory) / "report.json"
            html_path = Path(directory) / "report.html"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(["train", "--config", str(config_path), "--data", str(data_path), "--output", str(run_path)]),
                    0,
                )
                self.assertEqual(
                    cli_main([
                        "evaluate", "--run", str(run_path), "--data", str(data_path),
                        "--output", str(report_path), "--html", str(html_path),
                    ]),
                    0,
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("macro_f1", report["metrics"])
            if importlib.util.find_spec("plotly"):
                html = html_path.read_text(encoding="utf-8")
                self.assertIn("Confusion matrices", html)
                self.assertIn("plotly.js", html)


if __name__ == "__main__":
    unittest.main()
