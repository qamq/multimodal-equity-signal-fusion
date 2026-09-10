# Model implementations and APIs

The package implements the four final-paper model families. The paper reports the original empirical pipeline; the public HMM contains subsequent correctness hardening and is not an exact numerical replication target. No full historical HMM rerun has followed those changes.

## Common expert panel

Two out-of-sample expert forecasts enter the models: `p_cnn` for the chart-image CNN and `p_rf` for the structured-data RF. Inputs and identifier restrictions are specified in [Data](DATA.md). The public package combines forecasts; it does not train the two experts.

## 50/50 within-date rank blend

`RankBlend()` averages the two within-date percentile ranks, using average ranks for ties. Its `predict(df)` returns a Series with the caller's index and needs no fitting or labels. The result is a ranking score, not a calibrated probability. Configurable weights are normalized to sum to one; the final-paper benchmark uses the defaults of one half each.

## Walk-forward logistic stack

`WalkForwardLogisticStack.fit(df)` clips expert probabilities, converts them to logits, fits a historical-sample scaler, and estimates an L2-regularized logistic model for each target year. Defaults retain a one-observation-date label embargo. When usable prior dates are insufficient, the optional default warm start produces a rank blend; single-class training labels otherwise raise an error.

`predict(df)` **retrieves the historical walk-forward predictions generated during fit**. It preserves caller index/order and raises an error for stock-date keys without cached predictions. It is not an arbitrary-date inference API. The target-year rows to forecast must have been included in the fitting panel. Cached output is `predictions_` with `Date`, `StockID`, `up_prob`.

## Walk-forward expected-return / Ridge fusion

`WalkForwardExpectedReturnFusion.fit(df)` regresses realized returns on the same historically standardized expert logits, using Ridge regularization. Years with inadequate training observations are skipped; a fit producing no predictions raises an error.

`predict(df)` retrieves cached historical predictions and rejects unseen keys, as in the Logistic model. `predictions_` stores the expected-return score under `mu_hat`. For compatibility, the returned Series is currently named `up_prob`, but its values are **return estimates, not probabilities**. Rename it when assigning it to a downstream frame.

## Multi-horizon HMM

`HMMMultihorizon` models a date-level Markov state with state-specific emissions:

1. A 20-day logistic block uses the RF expert and context.
2. A 5-day logistic block conditions on the 20-day outcome, both experts, and context.

An EM-style loop alternates posterior inference with transition and weighted logistic updates. Missing outcomes contribute zero emission log-likelihood. A row can contribute to the 20-day block without a 5-day label. The conditional 5-day block requires both observed outcomes; the implementation does not impute its conditioning label. After the final M-step, the filter is recomputed under the returned parameters.

`fit(df)` generates chronological forecasts with monthly, quarterly or annual refit blocks. It caches keyed row forecasts in `row_pred_` and forecast-date state probabilities in `state_probs_by_date_`.

`predict(df)` returns a Series with exactly the supplied index/order. It retrieves cached forecasts and can extend the latest model to new, chronologically later dates. It uses no target columns and does not refit parameters. New-date state propagation advances once per supplied date; callers must supply the intended forecast sequence. Invalid inputs and historical rows without cached forecasts return missing predictions. Required context columns are fixed during fitting. Integer-like identifiers are required.

At prediction, the unknown 20-day outcome is marginalized out; 5-day probabilities are averaged over propagated state probabilities.

## Information cutoffs

| Path | Availability rule |
|---|---|
| Logistic / Ridge annual fits | Use nonmissing outcomes on dates before the first target-year date, excluding the latest `label_lag_periods` labeled dates |
| HMM refits, including annual runners and frozen-configuration runs | Apply the existing observation-date embargo, then require each actual target endpoint strictly before the block's first forecast date |
| Annual HMM tuning for year `y` | Training/validation outcome availability is capped at January 1 of `y`; inner refits retain their earlier block cutoffs |
| Fixed-development HMM tuning | Outer cutoff is January 1 after `dev_end_year`; inner refits retain their earlier cutoffs |

Logistic and Ridge retain their original date-count embargo; they do **not** independently inspect actual horizon endpoints. The caller must ensure that the supplied labels and embargo cover the target horizon, particularly on irregular grids.

HMM endpoint columns are `label_end_5d` and `label_end_20d`. Unknown endpoints are unavailable; no blanket calendar-year exclusion replaces the endpoint checks. Validation scoring is restricted separately from prediction coverage. Default weekly/monthly zero-delay scoring uses the relevant source endpoint; a different holding period requires actual `score_end` metadata matching its scoring returns.

## Historical research infrastructure

The orchestration layer includes annual tuning, signal alignment and portfolio adapters. Some operations require the original private data, expert-running and portfolio modules. The public synthetic example exercises the models directly. See [Reproducibility](REPRODUCIBILITY.md) for the distinction between these interfaces and historical empirical replication.
