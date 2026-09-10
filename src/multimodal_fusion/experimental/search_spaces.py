
"""
multimodal_fusion.experimental.search_spaces

Randomized search spaces for annual ensemble tuning.

Design principles
-----------------
- Keep spaces small and auditable.
- Preserve model-identity choices supplied in base_kwargs.
  Examples:
    * DynamicWeight objective is fixed when you run separate objective variants.
    * EM plain vs EM tail are separate model families, so use_tail is never toggled
      unless the base config leaves it unspecified.
    * Baseline vs enriched variants preserve their feature/context configuration.
    * Row-level MoE vs date-level MoE remain distinct method families.
- Use incumbent-aware local perturbations plus fresh global draws.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np


# Recursively merge two nested configuration dictionaries.
def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


# Sample one float from a log-uniform distribution.
def _log_uniform(rng: np.random.RandomState, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(float(low)), np.log(float(high)))))


# Sample one float from a uniform distribution.
def _uniform(rng: np.random.RandomState, low: float, high: float) -> float:
    return float(rng.uniform(float(low), float(high)))


# Sample one item uniformly from a candidate list.
def _choice(rng: np.random.RandomState, items: List[Any]) -> Any:
    return deepcopy(items[int(rng.randint(0, len(items)))])


# Apply multiplicative log-scale jitter around an incumbent value.
def _jitter_log(
    rng: np.random.RandomState,
    value: float,
    *,
    low: float,
    high: float,
    mult_low: float = 0.6,
    mult_high: float = 1.8,
) -> float:
    val = max(float(value), 1e-12)
    fac = float(np.exp(rng.uniform(np.log(mult_low), np.log(mult_high))))
    return float(np.clip(val * fac, float(low), float(high)))


# Apply additive linear jitter around an incumbent value.
def _jitter_linear(
    rng: np.random.RandomState,
    value: float,
    *,
    low: float,
    high: float,
    width: float,
) -> float:
    val = float(value)
    out = val + rng.uniform(-float(width), float(width))
    return float(np.clip(out, float(low), float(high)))


# Sample a nearby discrete value around the incumbent choice.
def _neighbor_choice(rng: np.random.RandomState, value: Any, ordered_items: List[Any]) -> Any:
    items = list(ordered_items)
    if value not in items:
        return _choice(rng, items)
    i = items.index(value)
    lo = max(0, i - 1)
    hi = min(len(items) - 1, i + 1)
    return deepcopy(items[int(rng.randint(lo, hi + 1))])


# Convert a nested configuration object into a hashable representation.
def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


# Extract the nested config dictionary from base kwargs when present.
def _cfg_base(base_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = base_kwargs or {}
    cfg = base.get("config", {})
    return cfg if isinstance(cfg, dict) else {}


# Return the default random-search budget for one method family.
def default_budget_for_method(method: str) -> int:
    method_name = str(method).lower().strip()
    if method_name in {"moe_gating", "rl_policy"}:
        return 10
    if method_name in {"contextual_bandit", "em_responsibility", "moe_gating_date"}:
        return 12
    if method_name in {"dynamic_weight", "reliability_hedge"}:
        return 16
    return 8


# Draw one fresh global random candidate for the requested method.
def sample_one_global(
    method: str,
    rng: np.random.RandomState,
    *,
    base_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    method_name = str(method).lower().strip()
    base = deepcopy(base_kwargs or {})
    cfg_base = _cfg_base(base_kwargs)

    if method_name == "dynamic_weight":
        out: Dict[str, Any] = {}

        update_freq = str(base.get("update_freq", "week")).lower().strip()
        if update_freq == "month":
            lookbacks = [12, 24, 36]
        elif update_freq == "quarter":
            lookbacks = [8, 12, 16]
        else:
            lookbacks = [13, 26, 39, 52, 78, 104]

        if "objective" not in base:
            out["objective"] = _choice(
                rng,
                [
                    "logloss",
                    "tail_quantile_logloss",
                    "tail_margin_weighted_logloss",
                    "focal_logloss",
                    "rank_auc",
                    "rank_spearman_ic",
                    "rank_decile_spread",
                ],
            )

        objective = str(out.get("objective", base.get("objective", "logloss"))).lower().strip()

        if "lookback_periods" not in base:
            out["lookback_periods"] = _choice(rng, lookbacks)
        if "temp" not in base:
            out["temp"] = round(_log_uniform(rng, 0.03, 0.30), 6)
        if "label_lag" not in base:
            out["label_lag"] = 1

        if objective == "tail_quantile_logloss" and "tail_q" not in base:
            out["tail_q"] = round(_uniform(rng, 0.05, 0.20), 4)
        if objective == "tail_margin_weighted_logloss" and "margin_gamma" not in base:
            out["margin_gamma"] = _choice(rng, [1.0, 2.0, 3.0])
        if objective == "focal_logloss" and "focal_gamma" not in base:
            out["focal_gamma"] = _choice(rng, [1.0, 2.0, 3.0])
        if objective in {"rank_spearman_ic", "rank_decile_spread"} and "ret_col" not in base:
            out["ret_col"] = "fwd_ret"
        if objective == "rank_decile_spread" and "spread_q" not in base:
            out["spread_q"] = round(_uniform(rng, 0.05, 0.20), 4)

        return out

    if method_name == "reliability_hedge":
        out = {}
        if "eta" not in base:
            out["eta"] = round(_log_uniform(rng, 0.05, 20.0), 6)
        if "reward" not in base:
            out["reward"] = _choice(rng, ["neg_logloss", "neg_brier"])
        if "label_lag" not in base:
            out["label_lag"] = 1
        return out

    if method_name == "contextual_bandit":
        cfg = {}
        if "reward" not in cfg_base:
            cfg["reward"] = _choice(rng, ["neg_logloss", "neg_logloss_tail", "neg_brier", "neg_brier_tail"])
        if "tail_q" not in cfg_base:
            cfg["tail_q"] = round(_uniform(rng, 0.05, 0.20), 4)
        if bool(cfg_base.get("include_perf_context", True)) and "perf_lookback_periods" not in cfg_base:
            cfg["perf_lookback_periods"] = _choice(rng, [4, 8, 13, 26])
        if "prior_lambda" not in cfg_base:
            cfg["prior_lambda"] = round(_log_uniform(rng, 5.0, 200.0), 6)
        if "ts_noise" not in cfg_base:
            cfg["ts_noise"] = round(_log_uniform(rng, 0.003, 0.20), 6)
        if "forget_gamma" not in cfg_base:
            cfg["forget_gamma"] = round(_uniform(rng, 0.94, 1.0), 6)
        if "switch_penalty" not in cfg_base:
            cfg["switch_penalty"] = round(_uniform(rng, 0.0, 0.10), 6)
        if "label_lag" not in cfg_base:
            cfg["label_lag"] = 1
        return {"config": cfg}

    if method_name == "moe_gating_date":
        out = {}
        if "lr" not in base:
            out["lr"] = round(_log_uniform(rng, 0.005, 0.20), 6)
        if "l2" not in base:
            out["l2"] = round(_log_uniform(rng, 1e-5, 1e-1), 8)
        if "forget_gamma" not in base:
            out["forget_gamma"] = round(_uniform(rng, 0.95, 1.0), 6)
        if "label_lag" not in base:
            out["label_lag"] = 1
        if bool(base.get("tail_only", False)) and "tail_q" not in base:
            out["tail_q"] = round(_uniform(rng, 0.05, 0.20), 4)
        return out

    if method_name == "em_responsibility":
        cfg = {}
        if "lookback_periods" not in cfg_base:
            cfg["lookback_periods"] = _choice(rng, [None, 13, 26, 52, 104])
        if "calibrate" not in cfg_base:
            cfg["calibrate"] = _choice(rng, [True, False])
        if "beta_a" not in cfg_base:
            cfg["beta_a"] = round(_uniform(rng, 1.2, 4.0), 4)
        if "beta_b" not in cfg_base:
            cfg["beta_b"] = round(_uniform(rng, 1.2, 4.0), 4)
        if "prior_strength" not in cfg_base:
            cfg["prior_strength"] = round(_uniform(rng, 0.0, 2.0), 6)
        if "label_lag" not in cfg_base:
            cfg["label_lag"] = 1
        # Keep plain EM vs tail EM separate by design.
        if bool(cfg_base.get("use_tail", False)) and "tail_q" not in cfg_base:
            cfg["tail_q"] = round(_uniform(rng, 0.05, 0.20), 4)
        return {"config": cfg}

    if method_name == "moe_gating":
        out = {}
        if "lr" not in base:
            out["lr"] = round(_log_uniform(rng, 0.005, 0.20), 6)
        if "l2" not in base:
            out["l2"] = round(_log_uniform(rng, 1e-5, 1e-1), 8)
        if "refit_freq" not in base:
            out["refit_freq"] = _choice(rng, ["month", "quarter", "year"])
        if "train_window_years" not in base:
            out["train_window_years"] = _choice(rng, [1, 2, 3])
        if "label_lag" not in base:
            out["label_lag"] = 1
        return out

    if method_name == "rl_policy":
        cfg = {}
        if "reward" not in cfg_base:
            cfg["reward"] = _choice(rng, ["neg_logloss", "neg_logloss_tail", "neg_brier", "neg_brier_tail"])
        if "tail_q" not in cfg_base:
            cfg["tail_q"] = round(_uniform(rng, 0.05, 0.20), 4)
        if "policy_temperature" not in cfg_base:
            cfg["policy_temperature"] = round(_log_uniform(rng, 0.30, 3.0), 6)
        if "lr" not in cfg_base:
            cfg["lr"] = round(_log_uniform(rng, 0.005, 0.20), 6)
        if "l2" not in cfg_base:
            cfg["l2"] = round(_log_uniform(rng, 1e-6, 1e-2), 8)
        if "forget_gamma" not in cfg_base:
            cfg["forget_gamma"] = round(_uniform(rng, 0.95, 1.0), 6)
        if "soft_target_scale" not in cfg_base:
            cfg["soft_target_scale"] = round(_uniform(rng, 1.0, 10.0), 6)
        if "switch_penalty" not in cfg_base:
            cfg["switch_penalty"] = round(_uniform(rng, 0.0, 0.10), 6)
        if "label_lag" not in cfg_base:
            cfg["label_lag"] = 1
        return {"config": cfg}

    if method_name == "fixed_blend":
        return {}

    raise ValueError(f"No random-search space registered for method='{method}'.")


# Draw one incumbent-centered local perturbation candidate.
def sample_one_local(
    method: str,
    incumbent: Dict[str, Any],
    rng: np.random.RandomState,
    *,
    base_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    method_name = str(method).lower().strip()
    base = deepcopy(base_kwargs or {})
    cfg_base = _cfg_base(base_kwargs)
    cand = deepcopy(incumbent)

    if method_name == "dynamic_weight":
        update_freq = str(base.get("update_freq", cand.get("update_freq", "week"))).lower().strip()
        if update_freq == "month":
            lookbacks = [12, 24, 36]
        elif update_freq == "quarter":
            lookbacks = [8, 12, 16]
        else:
            lookbacks = [13, 26, 39, 52, 78, 104]

        if "lookback_periods" not in base:
            cand["lookback_periods"] = _neighbor_choice(rng, cand.get("lookback_periods", 52), lookbacks)
        if "temp" not in base:
            cand["temp"] = round(_jitter_log(rng, float(cand.get("temp", 0.10)), low=0.03, high=0.30), 6)

        if "objective" not in base and rng.rand() < 0.20:
            cand["objective"] = _choice(
                rng,
                [
                    "logloss",
                    "tail_quantile_logloss",
                    "tail_margin_weighted_logloss",
                    "focal_logloss",
                    "rank_auc",
                    "rank_spearman_ic",
                    "rank_decile_spread",
                ],
            )

        objective = str(cand.get("objective", base.get("objective", "logloss"))).lower().strip()

        if objective == "tail_quantile_logloss" and "tail_q" not in base:
            cand["tail_q"] = round(_jitter_linear(rng, float(cand.get("tail_q", 0.10)), low=0.05, high=0.20, width=0.03), 4)
        if objective == "tail_margin_weighted_logloss" and "margin_gamma" not in base:
            cand["margin_gamma"] = _neighbor_choice(rng, cand.get("margin_gamma", 2.0), [1.0, 2.0, 3.0])
        if objective == "focal_logloss" and "focal_gamma" not in base:
            cand["focal_gamma"] = _neighbor_choice(rng, cand.get("focal_gamma", 2.0), [1.0, 2.0, 3.0])
        if objective == "rank_decile_spread" and "spread_q" not in base:
            cand["spread_q"] = round(_jitter_linear(rng, float(cand.get("spread_q", 0.10)), low=0.05, high=0.20, width=0.03), 4)
        if objective in {"rank_spearman_ic", "rank_decile_spread"} and "ret_col" not in base:
            cand["ret_col"] = "fwd_ret"

        cand["label_lag"] = int(base.get("label_lag", cand.get("label_lag", 1)))
        return cand

    if method_name == "reliability_hedge":
        if "eta" not in base:
            cand["eta"] = round(_jitter_log(rng, float(cand.get("eta", 5.0)), low=0.05, high=20.0), 6)
        if "reward" not in base and rng.rand() < 0.15:
            cand["reward"] = _choice(rng, ["neg_logloss", "neg_brier"])
        cand["label_lag"] = int(base.get("label_lag", cand.get("label_lag", 1)))
        return cand

    if method_name == "contextual_bandit":
        cfg = deepcopy(cand.get("config", {}))
        if "tail_q" not in cfg_base:
            cfg["tail_q"] = round(_jitter_linear(rng, float(cfg.get("tail_q", 0.10)), low=0.05, high=0.20, width=0.03), 4)
        if bool(cfg_base.get("include_perf_context", True)) and "perf_lookback_periods" not in cfg_base:
            cfg["perf_lookback_periods"] = _neighbor_choice(rng, cfg.get("perf_lookback_periods", 13), [4, 8, 13, 26])
        if "prior_lambda" not in cfg_base:
            cfg["prior_lambda"] = round(_jitter_log(rng, float(cfg.get("prior_lambda", 50.0)), low=5.0, high=200.0), 6)
        if "ts_noise" not in cfg_base:
            cfg["ts_noise"] = round(_jitter_log(rng, float(cfg.get("ts_noise", 0.03)), low=0.003, high=0.20), 6)
        if "forget_gamma" not in cfg_base:
            cfg["forget_gamma"] = round(_jitter_linear(rng, float(cfg.get("forget_gamma", 0.985)), low=0.94, high=1.0, width=0.02), 6)
        if "switch_penalty" not in cfg_base:
            cfg["switch_penalty"] = round(_jitter_linear(rng, float(cfg.get("switch_penalty", 0.05)), low=0.0, high=0.10, width=0.03), 6)
        if "reward" not in cfg_base and rng.rand() < 0.10:
            cfg["reward"] = _choice(rng, ["neg_logloss", "neg_logloss_tail", "neg_brier", "neg_brier_tail"])
        cfg["label_lag"] = int(cfg_base.get("label_lag", cfg.get("label_lag", 1)))
        cand["config"] = cfg
        return cand

    if method_name == "moe_gating_date":
        if "lr" not in base:
            cand["lr"] = round(_jitter_log(rng, float(cand.get("lr", 0.05)), low=0.005, high=0.20), 6)
        if "l2" not in base:
            cand["l2"] = round(_jitter_log(rng, float(cand.get("l2", 1e-3)), low=1e-5, high=1e-1), 8)
        if "forget_gamma" not in base:
            cand["forget_gamma"] = round(_jitter_linear(rng, float(cand.get("forget_gamma", 1.0)), low=0.95, high=1.0, width=0.02), 6)
        if bool(base.get("tail_only", False)) and "tail_q" not in base:
            cand["tail_q"] = round(_jitter_linear(rng, float(cand.get("tail_q", 0.10)), low=0.05, high=0.20, width=0.03), 4)
        cand["label_lag"] = int(base.get("label_lag", cand.get("label_lag", 1)))
        return cand

    if method_name == "em_responsibility":
        cfg = deepcopy(cand.get("config", {}))
        if "lookback_periods" not in cfg_base:
            cfg["lookback_periods"] = _neighbor_choice(rng, cfg.get("lookback_periods", None), [None, 13, 26, 52, 104])
        if "beta_a" not in cfg_base:
            cfg["beta_a"] = round(_jitter_linear(rng, float(cfg.get("beta_a", 2.0)), low=1.2, high=4.0, width=0.5), 4)
        if "beta_b" not in cfg_base:
            cfg["beta_b"] = round(_jitter_linear(rng, float(cfg.get("beta_b", 2.0)), low=1.2, high=4.0, width=0.5), 4)
        if "prior_strength" not in cfg_base:
            cfg["prior_strength"] = round(_jitter_linear(rng, float(cfg.get("prior_strength", 0.0)), low=0.0, high=2.0, width=0.4), 6)
        if "calibrate" not in cfg_base and rng.rand() < 0.15:
            cfg["calibrate"] = _choice(rng, [True, False])
        if bool(cfg_base.get("use_tail", False)) and "tail_q" not in cfg_base:
            cfg["tail_q"] = round(_jitter_linear(rng, float(cfg.get("tail_q", 0.10)), low=0.05, high=0.20, width=0.03), 4)
        cfg["by_date"] = bool(cfg_base.get("by_date", cfg.get("by_date", True)))
        cfg["label_lag"] = int(cfg_base.get("label_lag", cfg.get("label_lag", 1)))
        if "use_tail" in cfg_base:
            cfg["use_tail"] = bool(cfg_base["use_tail"])
        cand["config"] = cfg
        return cand

    if method_name == "moe_gating":
        if "lr" not in base:
            cand["lr"] = round(_jitter_log(rng, float(cand.get("lr", 0.05)), low=0.005, high=0.20), 6)
        if "l2" not in base:
            cand["l2"] = round(_jitter_log(rng, float(cand.get("l2", 1e-3)), low=1e-5, high=1e-1), 8)
        if "refit_freq" not in base:
            cand["refit_freq"] = _neighbor_choice(rng, cand.get("refit_freq", "quarter"), ["month", "quarter", "year"])
        if "train_window_years" not in base:
            cand["train_window_years"] = _neighbor_choice(rng, cand.get("train_window_years", 2), [1, 2, 3])
        cand["label_lag"] = int(base.get("label_lag", cand.get("label_lag", 1)))
        return cand

    if method_name == "rl_policy":
        cfg = deepcopy(cand.get("config", {}))
        if "tail_q" not in cfg_base:
            cfg["tail_q"] = round(_jitter_linear(rng, float(cfg.get("tail_q", 0.10)), low=0.05, high=0.20, width=0.03), 4)
        if "policy_temperature" not in cfg_base:
            cfg["policy_temperature"] = round(_jitter_log(rng, float(cfg.get("policy_temperature", 1.0)), low=0.30, high=3.0), 6)
        if "lr" not in cfg_base:
            cfg["lr"] = round(_jitter_log(rng, float(cfg.get("lr", 0.05)), low=0.005, high=0.20), 6)
        if "l2" not in cfg_base:
            base_l2 = float(cfg.get("l2", 1e-5 if float(cfg.get("l2", 0.0)) <= 0 else cfg.get("l2", 1e-5)))
            cfg["l2"] = round(_jitter_log(rng, base_l2, low=1e-6, high=1e-2), 8)
        if "forget_gamma" not in cfg_base:
            cfg["forget_gamma"] = round(_jitter_linear(rng, float(cfg.get("forget_gamma", 0.995)), low=0.95, high=1.0, width=0.02), 6)
        if "soft_target_scale" not in cfg_base:
            cfg["soft_target_scale"] = round(_jitter_linear(rng, float(cfg.get("soft_target_scale", 5.0)), low=1.0, high=10.0, width=2.0), 6)
        if "switch_penalty" not in cfg_base:
            cfg["switch_penalty"] = round(_jitter_linear(rng, float(cfg.get("switch_penalty", 0.0)), low=0.0, high=0.10, width=0.03), 6)
        if "reward" not in cfg_base and rng.rand() < 0.10:
            cfg["reward"] = _choice(rng, ["neg_logloss", "neg_logloss_tail", "neg_brier", "neg_brier_tail"])
        cfg["label_lag"] = int(cfg_base.get("label_lag", cfg.get("label_lag", 1)))
        cand["config"] = cfg
        return cand

    return deepcopy(incumbent)


# Assemble a deduplicated set of incumbent, local, and global candidates.
def sample_candidates(
    method: str,
    *,
    budget: int,
    random_seed: int,
    base_kwargs: Optional[Dict[str, Any]] = None,
    incumbent_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    base = deepcopy(base_kwargs or {})
    if int(budget) <= 0:
        return [deepcopy(incumbent_kwargs or base)]

    rng = np.random.RandomState(int(random_seed))
    out: List[Dict[str, Any]] = []
    seen = set()
    max_tries = max(10 * int(budget), 50)

    def _add(candidate: Dict[str, Any]) -> None:
        frozen = _freeze(candidate)
        if frozen in seen:
            return
        seen.add(frozen)
        out.append(candidate)

    incumbent = None
    if incumbent_kwargs is not None:
        incumbent = _deep_update(base, incumbent_kwargs)
        _add(incumbent)

    local_budget = 0
    if incumbent is not None and int(budget) > 1:
        local_budget = max(1, int(np.floor((int(budget) - 1) / 2)))

    tries = 0
    while len(out) < min(int(budget), 1 + local_budget) and tries < max_tries and incumbent is not None:
        tries += 1
        cand = sample_one_local(method, incumbent, rng, base_kwargs=base)
        _add(_deep_update(base, cand))

    tries = 0
    while len(out) < int(budget) and tries < max_tries:
        tries += 1
        cand = sample_one_global(method, rng, base_kwargs=base)
        _add(_deep_update(base, cand))

    if not out:
        out = [base]
    return out
