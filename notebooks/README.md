# Research notebooks

Only these three notebooks are intended for public review:

| Notebook | Role |
|---|---|
| [01_experimental_ensemble_benchmark.ipynb](01_experimental_ensemble_benchmark.ipynb) | Broad exploratory ensemble benchmark with annual tuning; not the headline four-model comparison |
| [02_fusion_ablation_statistical_validation.ipynb](02_fusion_ablation_statistical_validation.ipynb) | Original fusion ablation, statistical inference, implementation and capacity workflow |
| [03_economic_mechanism_audit.ipynb](03_economic_mechanism_audit.ipynb) | Post-processing of saved diversification, turnover, short-sleeve capacity and state diagnostics |

Outputs, execution counts and attachments are removed. Python metadata is aligned with the public package; this does not certify execution of the private workflow under that interpreter. Numerical settings and research calculations are retained.

## External environment

These notebooks are research records, not self-contained public demos. Notebook 01 runs experimental models and needs private data/portfolio modules. Notebook 02 consumes original saved expert/HMM forecasts and private implementation infrastructure. Notebook 03 reads saved analysis tables. No licensed inputs or frozen predictions are supplied.

Install this public package separately before executing notebook 01. Its ensemble imports use `multimodal_fusion`. External `Scripts.*` imports identify the original research infrastructure and intentionally remain external.

Set the following environment variables to existing local resources; the notebooks do not guess personal filesystem locations:

| Variable | Used by | Meaning |
|---|---|---|
| `FUSION_RESEARCH_ROOT` | 01, 02 | Original private project directory containing `Scripts/` and its data configuration |
| `FUSION_RF_ROOT` | 01 | Existing RF run directory with its saved prediction artifacts |
| `FUSION_CNN_ROOT` | 01 | Existing CNN run directory with its saved prediction artifacts |
| `FUSION_ANALYSIS_DIR` | 02, 03 | Analysis output directory; 03 requires previously generated tables |

Notebook 02 still requires its original manifests and transaction-cost inputs below the configured research root. Configuration changes do not supply replacements for missing private services. Its potentially expensive run switches retain the original settings; inspect them before execution.

## HMM provenance

Saved HMM predictions used by the empirical notebooks belong to the original study. The maintained public HMM includes post-study correctness hardening and has not undergone a complete 2001–2024 historical rerun. See [Reproducibility](../docs/REPRODUCIBILITY.md) before interpreting differences between new model outputs and the paper.

For an executable public entry point, use `python examples/synthetic_demo.py` from the repository root.
