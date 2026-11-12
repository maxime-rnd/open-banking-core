# Features

Feature transformers accept pandas DataFrames and follow scikit-learn's
`fit` / `transform` interface. Core transformers are registered when
`open_banking_core.features` is imported. A project can register its own
transformer without editing the package:

```python
from open_banking_core.features import FeatureTransformer, register_feature

@register_feature("merchant_prefix")
class MerchantPrefix(FeatureTransformer):
    def __init__(self, column: str = "description") -> None:
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column].str.extract(r"^([A-Z]+)", expand=False).fillna("")
```

Use `build_feature` with a `FeatureSpec` after importing the module that
registers the project transformer. Text representation is explicit in config:
`tfidf`, `embedding`, or `hybrid`. Text cleaning happens before all of them.

Embedding providers implement `embed(texts) -> numpy.ndarray` and can also be
registered by project code with `@register_embedding_provider("name")`. The
built-in `local` provider loads a Sentence Transformers model lazily. Install
it with `pip install open-banking-core[embeddings]`. The `openai` and `vllm`
providers use the OpenAI Python client's embeddings endpoint; vLLM's OpenAI
compatible server exposes `/v1/embeddings`. Install the client with
`pip install open-banking-core[remote-embeddings]`. API keys are read from the
configured environment variable name and are never stored in YAML.

See [features.yaml](../../examples/features.yaml) for a full configuration.
