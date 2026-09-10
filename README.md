# Multimodal Signal Fusion for Cross-Sectional Equity Returns

**Prediction, implementation, and capacity**

Research code accompanying [*Multimodal Signal Fusion for Cross-Sectional Equity Returns: Prediction, Implementation, and Capacity*](paper/multimodal_signal_fusion.pdf), by Quinn McMurtry.

The study asks whether chart images and structured financial data contain complementary equity-return signals, whether state-dependent fusion improves on simple combinations, and how signal persistence affects turnover, trading costs, and capacity.

A 5-day chart-image CNN and a 20-day structured-data random forest supply out-of-sample expert forecasts. The original empirical study compares fusion methods on a common 2001–2024 stock-date universe, then examines weekly long-short portfolios, transaction costs, nonlinear market impact, and scale.

## Four final-paper methods

| Method | Construction | Output |
|---|---|---|
| 50/50 within-date rank blend | Average the two experts' percentile ranks | Ranking score |
| Walk-forward logistic stack | Annual L2-regularized regression on standardized expert logits | 5-day directional probability |
| Walk-forward expected-return fusion | Annual Ridge regression on standardized expert logits | Expected-return ranking signal |
| Multi-horizon HMM | Date-level regimes with conditional 5-day and 20-day logistic emissions | State-weighted 5-day probability |

~~~mermaid
flowchart LR
    A[Chart images] --> B[5-day CNN forecasts]
    C[Structured financial data] --> D[20-day RF forecasts]
    B --> E[Common stock-date panel]
    D --> E
    E --> F[Rank blend / Logistic / Ridge / HMM]
    F --> G[Cross-sectional portfolios]
    G --> H[Turnover, costs, and capacity]
~~~

The paper finds that fusion adds useful information, while greater complexity does not consistently improve implementable portfolios. The untuned rank blend leads the headline gross equal-weight fusion comparison. Signal persistence and nonlinear impact help explain why portfolio rankings change with trading costs and AUM. These are findings of the **original empirical research pipeline**; see the paper for estimates, assumptions, and limitations.

### Implementation note

The public HMM includes correctness hardening introduced after the original empirical study: missing-label preservation, forecasts independent of future-label availability, prediction-index alignment, final-filter consistency, and explicit target-realization cutoffs.

The complete 2001–2024 HMM estimation and tuning procedure has not been rerun after these corrections because of its computational cost. The numerical HMM results in the paper are outputs of the original empirical pipeline, rather than replication targets for the maintained public HMM. This package is the maintained reference implementation going forward. Synthetic tests validate software behavior and do not constitute a full historical replication.

## What runs publicly

The four model implementations, experimental ensemble code, tests, and synthetic demo run without licensed data. Historical notebooks and research adapters also require saved expert forecasts, licensed inputs, and private data/portfolio modules that are not distributed here. The expert training systems and full implementation/capacity engine are not part of this package.

Timing controls are documented in [Methods](docs/METHODS.md). The HMM checks each target's actual end date before every refit; Logistic and Ridge retain their observation-date embargo and require callers to validate outcome availability.

## Quick start

From the downloaded repository directory, using Python **3.10 or newer**:

~~~bash
python -m venv .venv
~~~

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then run:

~~~bash
python -m pip install -e ".[dev]"
pytest
python examples/synthetic_demo.py
~~~

Optional notebook and research-analysis dependencies:

~~~bash
python -m pip install -e ".[research]"
~~~

This extra installs analysis libraries; it does not install the private research infrastructure.

A minimal example using the bundled synthetic panel:

~~~python
from examples.synthetic_demo import make_panel
from multimodal_fusion import RankBlend, WalkForwardLogisticStack

panel = make_panel(n_stocks=20)
panel["rank_blend"] = RankBlend().predict(panel)

stack = WalkForwardLogisticStack(start_year=2017, end_year=2021).fit(panel)
rows = panel[panel["Date"].dt.year.between(2017, 2021)].copy()
rows["stack_probability"] = stack.predict(rows)  # Retrieves fitted walk-forward forecasts.
~~~

See [Data](docs/DATA.md) for input schemas, identifier constraints, and HMM horizon-end metadata.

## Source guide

| Location | Purpose |
|---|---|
| [`core/`](src/multimodal_fusion/core/) | Four final-paper model families and HMM label-availability helpers |
| [`experimental/`](src/multimodal_fusion/experimental/) | Bandits, dynamic weighting, Hedge, EM responsibility, mixture-of-experts gates, and RL-style blending |
| [`pipeline/`](src/multimodal_fusion/pipeline/) | Walk-forward orchestration and adapters to private research infrastructure |
| [`diagnostics/`](src/multimodal_fusion/diagnostics/) | HMM state and emission diagnostics |
| [`notebooks/`](notebooks/README.md) | Three selected research records, with outputs removed |
| [`tests/`](tests/) and [`examples/`](examples/) | Public validation and deterministic synthetic data |
| [`paper/`](paper/) | Accompanying paper |

Experimental methods are retained for research provenance and are **not headline final-paper models**. Their scope is described in [Experimental methods](docs/EXPERIMENTAL_METHODS.md).

## Reproducibility and citation

No licensed source data, private predictions, or model checkpoints are redistributed. [Reproducibility](docs/REPRODUCIBILITY.md) distinguishes public validation from historical research replication.

Citation metadata is available in [CITATION.cff](CITATION.cff). The study uses simulated execution and is a research implementation, not a live trading system.

## Rights

Copyright © 2026 Quinn McMurtry. All rights reserved.

Publicly viewable for research review and portfolio demonstration. No general open-source licence is granted. See [NOTICE.md](NOTICE.md); third-party dependencies retain their own terms.
