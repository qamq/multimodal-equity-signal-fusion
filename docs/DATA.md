# Data and input contracts

Licensed CRSP, Compustat, WRDS and other vendor inputs, saved expert predictions, and model checkpoints are not distributed. The public demo generates deterministic synthetic observations. No external account or credentials are needed for tests or the demo.

Use unique `(Date, StockID)` keys, normalized daily timestamps, and finite expert probabilities in `[0, 1]`. Probabilities used in logits are clipped numerically. Data acquisition, expert training, and point-in-time feature validation remain the caller's responsibility.

## Model inputs

| Column | Meaning and consumers |
|---|---|
| `Date` | Decision date; required by every model |
| `StockID` | Stock identifier; required except by `RankBlend` |
| `p_cnn`, `p_rf` | Contemporaneously available out-of-sample expert probabilities |
| `y` | Binary 5-day outcome for Logistic fitting |
| `fwd_ret` | Realized forward return for Ridge fitting |
| `y_5d`, `y_20d` | HMM outcomes; values `0`, `1`, or missing |
| `fwd_ret_5d`, `fwd_ret_20d` | Optional HMM source returns; absent label columns can be derived at threshold zero |
| `label_end_5d`, `label_end_20d` | Actual last observation dates used to construct the respective HMM targets |
| `ctx_*` | Optional contemporaneous HMM context; selected columns must also be supplied to prediction |

Positive returns map to `1`; zero and negative returns map to `0` under the default threshold. A missing forward return stays missing. `HMMDataBuilder` supports a configured threshold and maps configurable source endpoint names, `label_end_5d_col` and `label_end_20d_col`, to the canonical names above.

## Identifier constraints

`RankBlend` does not use identifiers. Logistic and Ridge accept nonnumeric identifiers and use their string representation for prediction lookup; use a consistent representation across fitting and retrieval.

The HMM, its data builder and diagnostics, and several historical pipeline utilities still convert identifiers through numeric integer form. Supply **integer-like IDs**, such as `10001` or `"10001"`. Ticker strings are unsupported in those components. Nonnumeric IDs may be coerced to missing and filtered; fractional IDs may be truncated, and leading zeros are not preserved. Validate or map identifiers before using these interfaces. This package does not provide a universal security-identifier mapping.

## HMM outcome availability

Endpoint metadata must come from the return-generation source. The package does not estimate trading calendars from weekdays or the observed forecast grid. An unknown endpoint makes that outcome unavailable for training.

For a refit whose first forecast date is `t`, an outcome is usable only if its return/label is nonmissing and its endpoint is **strictly before `t`**. An endpoint equal to `t` is excluded. The configured observation-date embargo remains an additional constraint. The 20-day emission can use its own available label; the conditional 5-day emission requires both outcomes.

Prediction requires keys, expert probabilities and the fitted context columns, **not outcomes or endpoint metadata**. Missing context values retain the model's existing zero-fill convention; forward returns are never imputed. The synthetic demo supplies explicitly defined synthetic endpoints, which are not an exchange-calendar rule.

For annual HMM tuning, validation outcomes must also be realized before January 1 of the prediction year. Nondefault delayed/other portfolio holding periods require actual `score_end` dates for cutoff-aware scoring. Details are in [Methods](METHODS.md).

## Private adapters and local files

`pipeline/` retains lazy calls to the original `Scripts.Data`, `Scripts.Portfolio` and `Scripts.Experiments` modules. Those names refer to external, unbundled infrastructure; installing this package or its `research` extra does not provide it.

Keep licensed inputs outside the repository, or in ignored local input/output directories. `data/` is ignored by default except its README and the explicitly public `data/synthetic/` and `data/fixtures/` locations. Those opt-in locations are only for small, intentionally redistributable examples; do not place vendor data or credentials there.
