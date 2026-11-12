"""Reproducible stratified splitting, independently by transaction sign."""

import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from open_banking_core.config import CoreConfig, SplitConfig


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test rows and the fraction of test descriptions seen in train."""

    train: pd.DataFrame
    test: pd.DataFrame
    test_seen_description_rate: float


SplitStrategy = Callable[[pd.DataFrame, str, SplitConfig], DatasetSplit]
_SPLIT_STRATEGIES: dict[str, SplitStrategy] = {}


def register_split_strategy(name: str) -> Callable[[SplitStrategy], SplitStrategy]:
    """Register a named data split implementation."""
    def decorator(strategy: SplitStrategy) -> SplitStrategy:
        if name in _SPLIT_STRATEGIES:
            raise ValueError(f"split strategy {name!r} is already registered")
        _SPLIT_STRATEGIES[name] = strategy
        return strategy
    return decorator


def normalize_description(value: object) -> str:
    """Normalize case, accents, and punctuation for duplicate grouping."""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\W+", " ", text, flags=re.UNICODE).strip()


def _seen_rate(train: pd.DataFrame, test: pd.DataFrame) -> float:
    if test.empty:
        return 0.0
    seen = {normalize_description(value) for value in train["description"]}
    return sum(normalize_description(value) in seen for value in test["description"]) / len(test)


@register_split_strategy("stratified_random")
def _stratified_random(data: pd.DataFrame, target: str, config: SplitConfig) -> DatasetSplit:
    if target not in data:
        raise ValueError(f"target column {target!r} is missing")
    rng = random.Random(config.random_state)
    test_indices: list[object] = []
    for _, group in data.groupby(target, sort=True, dropna=False):
        indices = list(group.index)
        if len(indices) < 2:
            raise ValueError("each label needs at least two rows for a stratified split")
        rng.shuffle(indices)
        n_test = min(len(indices) - 1, max(1, round(len(indices) * config.test_size)))
        test_indices.extend(indices[:n_test])
    test = data.loc[test_indices].copy()
    train = data.drop(index=test_indices).copy()
    return DatasetSplit(train, test, _seen_rate(train, test))


@register_split_strategy("stratified_group")
def _stratified_group(data: pd.DataFrame, target: str, config: SplitConfig) -> DatasetSplit:
    if target not in data or "description" not in data:
        raise ValueError("grouped splitting requires target and description columns")
    grouped = data.copy()
    grouped["_normalized_description"] = grouped.description.map(normalize_description)
    group_label_counts: dict[str, Counter] = {}
    groups: dict[str, list[object]] = defaultdict(list)
    for index, row in grouped.iterrows():
        key = row["_normalized_description"]
        groups[key].append(index)
        group_label_counts.setdefault(key, Counter())[row[target]] += 1

    labels = list(data[target].drop_duplicates())
    for label in labels:
        if sum(counts[label] > 0 for counts in group_label_counts.values()) < 2:
            raise ValueError(f"label {label!r} occurs in fewer than two description groups")
    totals = Counter(data[target])
    desired = {label: totals[label] * config.test_size for label in labels}
    target_rows = len(data) * config.test_size
    rng = random.Random(config.random_state)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    group_keys.sort(key=lambda key: len(groups[key]), reverse=True)
    selected: set[str] = set()
    counts: Counter = Counter()
    rows = 0
    for key in group_keys:
        candidate = counts + group_label_counts[key]
        candidate_rows = rows + len(groups[key])
        leaves_train = all(totals[label] - candidate[label] >= 1 for label in labels)
        if not leaves_train:
            continue
        current_loss = sum((counts[label] - desired[label]) ** 2 / max(totals[label], 1) for label in labels)
        proposed_loss = sum((candidate[label] - desired[label]) ** 2 / max(totals[label], 1) for label in labels)
        if proposed_loss < current_loss or (not selected and rows < target_rows):
            selected.add(key)
            counts = candidate
            rows = candidate_rows
    if not selected or any(counts[label] == 0 for label in labels):
        raise ValueError("unable to create a grouped stratified split with these labels and groups")
    test_indices = [idx for key in selected for idx in groups[key]]
    test = data.loc[test_indices].copy()
    train = data.drop(index=test_indices).copy()
    return DatasetSplit(train, test, _seen_rate(train, test))


def split_dataset(data: pd.DataFrame, target: str, config: SplitConfig | CoreConfig) -> DatasetSplit:
    """Apply the configured split strategy to one sign-specific dataset."""
    if isinstance(config, CoreConfig):
        config = config.split
    strategy_name = "stratified_group" if config.group_by == "normalized_description" else config.strategy
    try:
        strategy = _SPLIT_STRATEGIES[strategy_name]
    except KeyError as exc:
        raise ValueError(f"unknown split strategy {strategy_name!r}") from exc
    return strategy(data, target, config)


def split_by_sign(data: pd.DataFrame, target: str, config: SplitConfig | CoreConfig) -> dict[str, DatasetSplit]:
    """Split debit and credit rows independently using the amount sign rule."""
    if "amount" not in data:
        raise ValueError("amount column is missing")
    debit = data.loc[data.amount <= 0].copy()
    credit = data.loc[data.amount > 0].copy()
    if debit.empty or credit.empty:
        raise ValueError("both debit and credit rows are required")
    return {
        "debit": split_dataset(debit, target, config),
        "credit": split_dataset(credit, target, config),
    }
