# Experimental ensemble methods

These methods are exploratory alternatives retained for research provenance. They are **not the four headline final-paper models**, which live in `core/`. Their presence does not imply that they received the HMM-specific correctness hardening or have been independently validated for live use.

| Module in `multimodal_fusion.experimental` | Purpose |
|---|---|
| `fixed_blend` | Convex combination of raw expert probabilities; distinct from the final-paper rank blend |
| `dynamic_weight` | Online date-level weights based on historical score differences |
| `reliability_hedge` | Multiplicative/Hedge updates from expert losses |
| `em_responsibility` | Walk-forward EM responsibility weights, with optional calibration and prior smoothing |
| `contextual_bandit` | Contextual selection over a grid of CNN weights |
| `moe_gating` | Stock-date mixture-of-experts gate |
| `moe_gating_date` | Date-level logistic gate |
| `rl_policy` | Softmax blending policy over discrete weights |
| `manager` | Common orchestration interface for these experimental models |
| `search_spaces` | Candidate spaces for the historical annual tuning workflow |

`EnsembleManager` and its configuration are imported from `multimodal_fusion.experimental.manager`. The first public notebook records the broader benchmark. Research pipeline scoring requires external `Scripts.Portfolio` infrastructure; the modules themselves can be imported without it.

Each experimental method retains its original fitting, label-lag, caching and identifier behavior. Consult its class/configuration docstrings before using it. The small public tests are software checks, not reproductions of the original empirical comparison.
