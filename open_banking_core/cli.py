"""Small command-line interface for training and evaluating local runs."""

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

from open_banking_core.config import CoreConfig
from open_banking_core.data import load_transactions, split_by_sign
from open_banking_core.evaluation import Evaluator
from open_banking_core.training import Trainer


def _prepare_data(config: CoreConfig, data_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = load_transactions(data_path, config)
    if report.errors:
        first_errors = "; ".join(f"row {item['row']}: {', '.join(item['errors'])}" for item in report.errors[:5])
        raise ValueError(f"input contains {len(report.errors)} invalid rows: {first_errors}")
    target = config.data.target_column or "category"
    sign_splits = split_by_sign(report.valid_rows, target=target, config=config.split)
    train_data = pd.concat([sign_splits["debit"].train, sign_splits["credit"].train]).sort_index()
    test_data = pd.concat([sign_splits["debit"].test, sign_splits["credit"].test]).sort_index()
    return train_data, test_data


def _train(args: argparse.Namespace) -> int:
    config = CoreConfig.from_yaml(args.config)
    train_data, _ = _prepare_data(config, args.data)
    router = Trainer(config).fit(train_data)
    run_path = Path(args.output)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format_version": 1, "router": router, "config": config}, run_path)
    print(json.dumps({"status": "trained", "train_rows": len(train_data), "run": str(run_path)}))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    run = joblib.load(args.run)
    if not isinstance(run, dict) or run.get("format_version") != 1:
        raise ValueError("unsupported local run file")
    config: CoreConfig = run["config"]
    train_data, test_data = _prepare_data(config, args.data)
    result = Evaluator(config).evaluate(run["router"], test_data, train_data=train_data)
    output = result.to_dict()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html:
        result.write_html(args.html, include_plotlyjs=not args.html_cdn)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``obc train`` and ``obc evaluate``."""
    parser = argparse.ArgumentParser(prog="obc", description="Open Banking Core training utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train both sign-specific pipelines")
    train.add_argument("--config", required=True, help="path to the full YAML configuration")
    train.add_argument("--data", required=True, help="path to the source CSV")
    train.add_argument("--output", required=True, help="path for the local joblib run")
    train.set_defaults(handler=_train)
    evaluate = commands.add_parser("evaluate", help="evaluate a saved run on the configured holdout split")
    evaluate.add_argument("--run", required=True, help="path to the local joblib run")
    evaluate.add_argument("--data", required=True, help="path to the source CSV")
    evaluate.add_argument("--output", help="optional JSON report output path")
    evaluate.add_argument("--html", help="optional interactive HTML report path; requires the reports extra")
    evaluate.add_argument("--html-cdn", action="store_true", help="load Plotly.js from a CDN to reduce HTML size")
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI subcommand and surface configuration/data errors cleanly."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    return 2
