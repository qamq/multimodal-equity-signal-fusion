# Reproducibility

## Public validation

With Python 3.10 or newer, from the repository directory:

~~~bash
python -m pip install -e ".[dev]"
pytest
pytest tests/test_hmm.py tests/test_hmm_correctness.py
python examples/synthetic_demo.py
python -m pip check
~~~

For a distribution build:

~~~bash
python -m pip install build
python -m build
~~~

Build outputs are local artifacts and are ignored. CI runs tests, the synthetic demo, dependency checks and a package build on Python 3.10 and 3.13. The local publication checks use Python 3.13; configuring CI is not evidence of a remote run.

The tests exercise model behavior, target timing, missing labels, prediction alignment and final-state consistency. Package import and diagnostics-loader checks do not require licensed infrastructure. Pipeline tests isolate unavailable research services; they do not validate external market data or the private portfolio engine.

The deterministic synthetic demo uses generated expert scores and returns. Its horizon-end dates are definitions of that synthetic process, not an inferred market calendar.

## Original empirical results and maintained HMM

The paper's 2001–2024 results were produced by the original empirical research pipeline. The maintained public HMM subsequently added missing-label preservation, label-independent forecast coverage, index/order preservation, final-filter recomputation and explicit horizon-realization cutoffs.

The complete historical HMM estimation and tuning procedure has **not** been rerun after these changes because of its computational cost. Accordingly, the paper's HMM numbers are not replication targets for this maintained implementation. Synthetic old-versus-corrected comparisons assess software impact; they do not establish full historical equivalence or replace the empirical study.

## External requirements

Historical notebook execution requires licensed data, saved out-of-sample CNN/RF forecasts, research manifests and portfolio/cost artifacts. The following dependencies are not distributed:

- `Scripts.Data`: market/accounting data, forward returns and research configuration.
- `Scripts.Experiments`: original expert-running infrastructure.
- `Scripts.Portfolio`: portfolio construction, cost forecasts, optimization, statistical validation and capacity/NAV workflows.

Installing `.[research]` supplies general analysis libraries, not these private modules or access to vendor data. The notebooks preserve the empirical workflow with sanitized configuration and no stored outputs. They have been syntax/structure checked, not executed against licensed data.

## Adapting the research workflow

1. Generate out-of-sample expert forecasts with point-in-time features.
2. Align a common stock-date universe using consistent identifiers.
3. Preserve missing outcomes and carry the actual endpoint of each forward target.
4. Apply the model-specific availability rules in [Methods](METHODS.md).
5. Rank and evaluate signals on common keys using a separately validated portfolio layer.
6. Record data versions, cutoffs, fitted configurations and execution assumptions.

Use only data you are entitled to access and redistribute. See [Notebooks](../notebooks/README.md) for explicit environment configuration. Reproducing the historical study additionally requires its original software environment, data versions, and original empirical artifacts.
