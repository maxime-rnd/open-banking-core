# Building sign-specific pipelines

`build_pipeline` constructs one sklearn `Pipeline` from a branch config.
`build_sign_router` creates a `SignRouter` with independent debit and credit
pipelines. Features are combined with scikit-learn `FeatureUnion`, which runs
each transformer on the same transaction DataFrame and concatenates its output
matrices. See the [scikit-learn FeatureUnion guide](https://scikit-learn.org/stable/modules/compose.html#featureunion-composite-feature-spaces).

The required data order is sign split first, train/test split second, then fit
the router on the combined training rows. The router independently routes those
training rows to each model. The example CSV and config can be used as follows:

```python
import pandas as pd

from open_banking_core import CoreConfig, build_sign_router, load_transactions, split_by_sign

config = CoreConfig.from_yaml("examples/pipeline_config.yaml")
report = load_transactions("examples/transactions.csv", config)
splits = split_by_sign(report.valid_rows, "category", config.split)
train = pd.concat([splits["debit"].train, splits["credit"].train]).sort_index()
test = pd.concat([splits["debit"].test, splits["credit"].test]).sort_index()

router = build_sign_router(config)
router.fit(train.drop(columns="category"), train["category"])
predictions = router.predict(test.drop(columns="category"))
```

`amount <= 0` routes to debit, so zero is handled by the debit branch. Add a
custom estimator with `@register_model("name")`; it must follow the
scikit-learn estimator interface. LightGBM is integrated through its sklearn
classifier API ([LightGBM classifier reference](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)).
