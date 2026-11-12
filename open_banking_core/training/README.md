# Training and evaluation

The simple trainer expects a prepared training DataFrame with canonical
transaction fields and the category target. It fits the configured debit and
credit models through `SignRouter`; call `split_by_sign` first so each sign has
its own train/test split.

```python
import pandas as pd

from open_banking_core import CoreConfig, load_transactions, split_by_sign
from open_banking_core.evaluation import Evaluator
from open_banking_core.training import Trainer

config = CoreConfig.from_yaml("examples/simple_training.yaml")
data = load_transactions("examples/transactions.csv", config).valid_rows
splits = split_by_sign(data, "category", config.split)
train = pd.concat([splits["debit"].train, splits["credit"].train]).sort_index()
test = pd.concat([splits["debit"].test, splits["credit"].test]).sort_index()

router = Trainer(config).fit(train)
report = Evaluator(config).evaluate(router, test, train_data=train)
print(report.to_dict())
```

The CLI provides the same seeded holdout flow:

```console
obc train --config examples/simple_training.yaml --data examples/transactions.csv --output runs/example.joblib
obc evaluate --run runs/example.joblib --data examples/transactions.csv --output runs/report.json --html runs/report.html
```

Evaluation includes weighted accuracy, log loss, multiclass Brier score, ECE,
confidence bins and threshold coverage, ROC and precision-recall curves,
confusion matrices, amount slices, and block-wise permutation importance.
Configure label weights and diagnostic settings under `evaluation` in YAML.
Permutation importance permutes each fitted feature block on the holdout rows;
in hybrid text mode, TF-IDF and embeddings are measured separately. Interpret
these scores as holdout diagnostics and avoid using the same holdout repeatedly
to select models.

Install the report dependency with `pip install open-banking-core[reports]`.
HTML reports are self-contained by default; pass `--html-cdn` to use a CDN and
reduce the file size. The joblib run is a simple local interim format.
Versioned, portable bundles are planned for the artifacts and serving phase.
