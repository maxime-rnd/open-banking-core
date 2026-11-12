# Phase 1 sample dataset

`transactions.csv` is a small synthetic dataset for exercising the data layer.
It contains both signs, two category labels per pipeline, optional blank dates,
and an `account_id` column that is deliberately ignored.

The configured target column is kept as canonical `category` so each sign can
be stratified independently:

```python
from open_banking_core import CoreConfig, load_transactions, split_by_sign

config = CoreConfig.from_yaml("examples/base.yaml")
report = load_transactions("examples/transactions.csv", config)
if report.errors:
    print(report.errors)

# First route rows by amount sign, then split each dataset into train and test.
splits = split_by_sign(report.valid_rows, target="category", config=config)
debit_train = splits["debit"].train
debit_test = splits["debit"].test
credit_train = splits["credit"].train
credit_test = splits["credit"].test
```
