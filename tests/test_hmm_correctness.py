from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multimodal_fusion import HMMMultihorizon, HMMMultihorizonConfig
from multimodal_fusion.core.hmm_labels import mask_unrealized_hmm_labels, realized_outcome_mask
from multimodal_fusion.core.hmm_multihorizon import _ConstantBernoulliModel
from multimodal_fusion.pipeline.hmm_data import HMMDataBuilder, HMMDataConfig
from multimodal_fusion.pipeline import hmm_annual_runner, hmm_annual_tuner, hmm_tuner


@pytest.fixture
def small_panel():
    rng = np.random.RandomState(27)
    dates = pd.date_range("2019-08-02", "2020-03-27", freq="W-FRI")
    d = pd.MultiIndex.from_product([dates, [str(i) for i in range(6)]], names=["Date", "StockID"]).to_frame(index=False)
    d["p_cnn"] = rng.uniform(0.1, 0.9, len(d))
    d["p_rf"] = rng.uniform(0.1, 0.9, len(d))
    d["ctx_market"] = np.sin(np.repeat(np.arange(len(dates)), 6))
    for horizon, weeks in ((5, 1), (20, 4)):
        d[f"fwd_ret_{horizon}d"] = rng.normal(size=len(d))
        d[f"y_{horizon}d"] = (d[f"fwd_ret_{horizon}d"] > 0).astype(float)
        d[f"label_end_{horizon}d"] = d["Date"] + pd.Timedelta(weeks=weeks)
    return d


def model_config(**kwargs):
    return replace(HMMMultihorizonConfig(
        max_iter=1, min_history_dates=4, label_lag_periods=0,
        train_window_years=None, context_cols=["ctx_market"], random_state=9,
    ), **kwargs)


def mock_label_source(monkeypatch, source):
    scripts = ModuleType("Scripts")
    data = ModuleType("Scripts.Data")
    data.equity_data = SimpleNamespace(get_period_ret=lambda *a, **kw: source.copy())
    scripts.Data = data
    monkeypatch.setitem(sys.modules, "Scripts", scripts)
    monkeypatch.setitem(sys.modules, "Scripts.Data", data)


def test_builder_preserves_missing_return_labels_and_source_endpoints(monkeypatch):
    source = pd.DataFrame({
        "Date": pd.to_datetime(["2019-12-20"] * 4), "StockID": [0, 1, 2, 3],
        "Ret_5d": [0.1, 0.0, -0.1, np.nan], "Ret_20d": [np.nan, -0.2, 0.0, 0.2],
        "end_short": pd.to_datetime(["2019-12-30"] * 4),
        "end_long": pd.to_datetime(["2020-01-22"] * 4),
    })
    mock_label_source(monkeypatch, source)
    builder = HMMDataBuilder(HMMDataConfig("", "", verbose=False, label_end_5d_col="end_short", label_end_20d_col="end_long"))
    inputs = source[["Date", "StockID"]].astype({"StockID": str})
    out = builder._attach_multihorizon_labels(inputs)
    pd.testing.assert_series_equal(out["y_5d"], pd.Series([1, 0, 0, pd.NA], dtype="Int64", name="y_5d"))
    pd.testing.assert_series_equal(out["y_20d"], pd.Series([pd.NA, 0, 0, 1], dtype="Int64", name="y_20d"))
    assert pd.isna(out.loc[3, "fwd_ret_5d"])
    assert out["label_end_5d"].eq(pd.Timestamp("2019-12-30")).all()
    assert out["label_end_20d"].eq(pd.Timestamp("2020-01-22")).all()


@pytest.mark.parametrize("horizon", [5, 20])
def test_core_derives_labels_without_turning_nan_into_zero(small_panel, horizon):
    d = small_panel.iloc[:4].drop(columns=[f"y_{horizon}d"]).copy()
    d[f"fwd_ret_{horizon}d"] = [0.1, 0.0, -0.1, np.nan]
    out = HMMMultihorizon(model_config())._prepare_panel(d, require_labels=True)
    np.testing.assert_equal(out[f"y_{horizon}d"].to_numpy(), [1, 0, 0, np.nan])
    assert len(out) == 4
    # Already-materialized class 0 must not resurrect an explicitly missing return.
    d[f"y_{horizon}d"] = 0
    out = HMMMultihorizon(model_config())._prepare_panel(d, require_labels=True)
    assert pd.isna(out.loc[3, f"y_{horizon}d"])


def test_builder_and_tuner_keep_unlabeled_prediction_rows(monkeypatch, small_panel):
    raw = small_panel.copy()
    columns = ["y_5d", "y_20d", "fwd_ret_5d", "fwd_ret_20d"]
    raw.loc[raw["Date"] >= "2020-01-01", columns] = np.nan
    source = raw.drop(columns=["y_5d", "y_20d"])
    mock_label_source(monkeypatch, source)
    monkeypatch.setattr(HMMDataBuilder, "_build_expert_panel", lambda self: raw[["Date", "StockID", "p_cnn", "p_rf"]].copy())
    cfg = hmm_tuner.HMMVersion1TuningConfig("", "", "", verbose=False)
    panel = hmm_tuner.build_hmm_panel(cfg)
    assert len(panel) == len(raw)
    assert panel.loc[panel["Date"] >= "2020-01-01", columns].isna().all().all()
    assert {"label_end_5d", "label_end_20d"}.issubset(panel)


def test_partial_labels_contribute_only_to_observed_emissions(small_panel, monkeypatch):
    d = small_panel.iloc[:4].copy()
    d["Date"] = pd.date_range("2019-09-01", periods=4)
    d["y_20d"] = [1, 0, np.nan, np.nan]
    d["y_5d"] = [1, np.nan, 0, np.nan]
    model = HMMMultihorizon(model_config())
    mats = model._build_matrices(d, require_labels=True)
    np.testing.assert_equal(mats["valid_5d"], [True, False, True, False])
    np.testing.assert_equal(mats["valid_20d"], [True, True, False, False])
    constants20 = [_ConstantBernoulliModel(0.7), _ConstantBernoulliModel(0.3)]
    constants5 = [_ConstantBernoulliModel(0.6), _ConstantBernoulliModel(0.4)]
    logB = model._emission_loglik(mats, constants20, constants5)
    np.testing.assert_allclose(logB, [
        [np.log(0.7) + np.log(0.6), np.log(0.3) + np.log(0.4)],
        [np.log(0.3), np.log(0.7)], [0, 0], [0, 0],
    ])
    calls = []
    def fit_logit(X, y, w, *args):
        calls.append(y.copy())
        assert np.isfinite(X).all() and np.isfinite(y).all()
        return _ConstantBernoulliModel(0.5)
    monkeypatch.setattr("multimodal_fusion.core.hmm_multihorizon._fit_weighted_logit", fit_logit)
    model._fit_state_models(mats, np.full((4, 2), 0.5))
    assert [len(y) for y in calls] == [2, 1, 2, 1]


def test_latest_unlabeled_dates_receive_forecasts_without_likelihood(small_panel):
    d = small_panel.copy()
    columns = ["y_5d", "y_20d", "fwd_ret_5d", "fwd_ret_20d"]
    d.loc[d["Date"] >= "2020-01-01", columns] = np.nan
    model = HMMMultihorizon(model_config()).fit(d)
    latest = d[d["Date"] >= "2020-01-01"]
    predictions = model.predict(latest.drop(columns=columns + ["label_end_5d", "label_end_20d"]))
    assert predictions.notna().all()
    pd.testing.assert_index_equal(predictions.index, latest.index)
    bundle = model.get_fit_bundle()
    training = d[d["Date"].isin(bundle["dates"])]
    training = mask_unrealized_hmm_labels(training, bundle["information_cutoff"])
    mats = model._build_matrices(training, require_labels=True)
    logB = model._emission_loglik(mats, bundle["models20"], bundle["models5"])
    missing_dates = mats["dates"] >= pd.Timestamp("2020-01-01")
    assert missing_dates.any()
    np.testing.assert_equal(logB[missing_dates], 0)
    assert model.fit_log_[-1]["num_observed_20d"] < model.fit_log_[-1]["num_rows"]


@pytest.fixture
def fitted_model(small_panel):
    return HMMMultihorizon(model_config()).fit(small_panel)


@pytest.mark.parametrize("ordering", ["shuffle", "reverse"])
def test_predict_preserves_original_index_and_row_identity(fitted_model, small_panel, ordering):
    rows = small_panel.merge(fitted_model.row_pred_, on=["Date", "StockID"], validate="one_to_one")
    rows.index = pd.Index(np.arange(len(rows)) * 7 + 11, name="caller_row")
    rows = rows.sample(frac=1, random_state=63) if ordering == "shuffle" else rows.iloc[::-1]
    actual = fitted_model.predict(rows)
    pd.testing.assert_index_equal(actual.index, rows.index, exact=True)
    np.testing.assert_allclose(actual, rows["up_prob"], atol=1e-12)
    assert actual.nunique() > 1


def test_predict_keeps_duplicate_indices_duplicate_keys_and_invalid_rows(fitted_model, small_panel):
    rows = small_panel[small_panel["Date"] == small_panel["Date"].max()].copy()
    rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    expected = fitted_model.predict(rows).to_numpy()
    rows.index = pd.Index(["same"] * len(rows), name="duplicate")
    rows.iloc[1, rows.columns.get_loc("p_cnn")] = np.nan
    expected[1] = np.nan
    actual = fitted_model.predict(rows)
    pd.testing.assert_index_equal(actual.index, rows.index)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    invalid = rows.assign(p_cnn=np.nan)
    pd.testing.assert_index_equal(fitted_model.predict(invalid).index, invalid.index)
    assert fitted_model.predict(invalid).isna().all()


def test_new_label_free_predictions_are_aligned_after_sorting(fitted_model, small_panel):
    latest = small_panel[small_panel["Date"] == small_panel["Date"].max()].copy()
    future = pd.concat([latest.assign(Date=latest["Date"] + pd.Timedelta(weeks=i)) for i in (1, 2)], ignore_index=True)
    future = future[["Date", "StockID", "p_cnn", "p_rf", "ctx_market"]]
    future.index = pd.Index(np.arange(len(future)) + 900, name="future")
    expected = deepcopy(fitted_model).predict(future)
    shuffled = future.sample(frac=1, random_state=41)
    actual = deepcopy(fitted_model).predict(shuffled)
    pd.testing.assert_series_equal(actual, expected.reindex(shuffled.index))


@pytest.mark.parametrize("max_iter,tol", [(0, 1e-4), (1, 1e-4), (8, 1e10)])
def test_final_filter_agrees_with_independent_forward_recursion(small_panel, max_iter, tol):
    model = HMMMultihorizon(model_config(max_iter=max_iter, tol=tol))
    cutoff = pd.Timestamp("2020-01-03")
    training = model._prepare_panel(small_panel[small_panel["Date"] < cutoff], require_labels=True)
    bundle = model._fit_hmm_block(training, information_cutoff=cutoff)
    observed = mask_unrealized_hmm_labels(training, cutoff)
    mats = model._build_matrices(observed, require_labels=True)
    # Compute emissions and a scaled forward recursion independently of the
    # production emission and forward/backward routines.
    emissions = np.zeros((mats["num_dates"], 2))
    for state in range(2):
        for target, design, models, valid in (
            ("y20", "X20", "models20", observed["y_20d"].notna().to_numpy()),
            ("y5", "X5_obs", "models5", observed[["y_5d", "y_20d"]].notna().all(axis=1).to_numpy()),
        ):
            p = np.clip(bundle[models][state].predict_proba(mats[design][valid])[:, 1], model.config.prob_clip, 1 - model.config.prob_clip)
            y = mats[target][valid]
            ll = np.log(np.where(y == 1, p, 1 - p))
            np.add.at(emissions[:, state], mats["date_ix"][valid], ll)
    q = np.clip(bundle["pi"], model.config.prob_clip, 1 - model.config.prob_clip)
    transition = np.clip(bundle["A"], model.config.prob_clip, 1 - model.config.prob_clip)
    loglik = 0.0
    for t, ll in enumerate(emissions):
        if t:
            q = q @ transition
        q = q * np.exp(ll)
        loglik += np.log(q.sum())
        q /= q.sum()
    np.testing.assert_allclose(bundle["filtered_last"], q, atol=1e-11, rtol=1e-11)
    assert bundle["final_loglik"] == pytest.approx(loglik, abs=1e-10)
    if max_iter == 8:
        assert len(bundle["ll_trace"]) == 2  # convergence exit also ends after an M-step


def test_realization_uses_independent_horizons_and_strict_year_end_cutoff():
    d = pd.DataFrame({
        "Date": pd.to_datetime(["2019-12-02", "2019-12-20", "2019-12-20", "2019-12-20"]),
        "y_5d": [1, 1, 1, np.nan], "y_20d": [0, 0, 0, 0],
        "label_end_5d": pd.to_datetime(["2019-12-09", "2019-12-30", "2020-01-01", "2019-12-30"]),
        "label_end_20d": pd.to_datetime(["2019-12-31", "2020-01-22", None, "2019-12-20"]),
    })
    out = mask_unrealized_hmm_labels(d, pd.Timestamp("2020-01-01"))
    np.testing.assert_equal(out["y_5d"].to_numpy(), [1, 1, np.nan, np.nan])
    np.testing.assert_equal(out["y_20d"].to_numpy(), [0, np.nan, np.nan, np.nan])
    # The endpoint's day has to have passed, including for labels spanning years.
    assert not realized_outcome_mask(d, 20, pd.Timestamp("2020-01-22")).iloc[1]
    assert realized_outcome_mask(d, 20, pd.Timestamp("2020-01-23")).iloc[1]


def test_unknown_endpoints_never_assumed_realized(small_panel):
    d = small_panel.drop(columns=["label_end_5d", "label_end_20d"])
    with pytest.raises(ValueError, match="label_end_5d / label_end_20d"):
        HMMMultihorizon(model_config()).fit(d)


@pytest.mark.parametrize("refit_freq", ["month", "quarter", "year"])
def test_unrealized_year_end_labels_cannot_change_january_forecasts(small_panel, refit_freq):
    cutoff = pd.Timestamp("2020-01-03")
    through_january = small_panel[small_panel["Date"] < "2020-02-01"].copy()
    changed = through_january.copy()
    for horizon in (5, 20):
        unavailable = changed[f"label_end_{horizon}d"] >= cutoff
        changed.loc[unavailable, f"y_{horizon}d"] = 1 - changed.loc[unavailable, f"y_{horizon}d"]
        changed.loc[unavailable, f"fwd_ret_{horizon}d"] *= -1
    cfg = model_config(refit_freq=refit_freq)
    before = HMMMultihorizon(cfg).fit(through_january)
    after = HMMMultihorizon(cfg).fit(changed)
    a = before.row_pred_.query("Date >= @cutoff").reset_index(drop=True)
    b = after.row_pred_.query("Date >= @cutoff").reset_index(drop=True)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)
    latest_bundle = before.get_fit_bundle()
    assert latest_bundle["information_cutoff"] == cutoff
    # December is not excluded wholesale: outcomes that ended before January
    # contribute, while late December's 20d outcomes do not.
    observed = mask_unrealized_hmm_labels(through_january, cutoff)
    assert observed.loc[observed["Date"] == "2019-12-06", "y_20d"].isna().all()
    assert observed.loc[observed["Date"] == "2019-12-20", "y_5d"].notna().all()


def test_configured_lag_remains_an_additional_embargo(small_panel):
    model = HMMMultihorizon(model_config(label_lag_periods=4))
    data = model._prepare_panel(small_panel, require_labels=True)
    dates = pd.Index(sorted(data["Date"].unique()))
    blocks = model._collect_eligible_blocks(data, dates, model._build_refit_dates(dates))
    january = next(block for block in blocks if block["pred_start"] == pd.Timestamp("2020-01-03"))
    assert january["hist_end"] == pd.Timestamp("2019-11-29")


def test_annual_tuning_masks_labels_and_passes_january_cutoff(monkeypatch, small_panel, tmp_path):
    seen = []
    def evaluate(panel, cfg, params, **kwargs):
        seen.append(panel.copy())
        assert cfg.dev_end_year == 2019
        assert panel["Date"].max() < pd.Timestamp("2020-01-01")
        assert panel.loc[panel["label_end_20d"] >= "2020-01-01", "y_20d"].isna().all()
        assert panel.loc[panel["Date"] == "2019-12-20", "y_5d"].notna().all()
        return {"params": params, "mean_score": 1.0, "std_score": 0.0, "per_year": [], "num_eval_years": 1}
    monkeypatch.setattr(hmm_tuner, "evaluate_hmm_candidate", evaluate)
    cfg = hmm_annual_tuner.HMMAnnualTuningConfig("", "", str(tmp_path), start_year=2019, end_year=2021, search_budget=1, verbose=False)
    payload = hmm_annual_tuner.tune_hmm_for_year(cfg, 2020, panel=small_panel)
    assert seen and payload["validation_years"] == [2019]


def test_tuning_scores_only_returns_realized_before_outer_cutoff(monkeypatch, small_panel):
    fitted = []
    scored = []
    def predict(panel, cfg, **kwargs):
        fitted.append(panel.copy())
        return panel[["Date", "StockID"]].assign(up_prob=0.6)
    monkeypatch.setattr(hmm_tuner, "fit_predict_validation_year", predict)
    monkeypatch.setattr(hmm_tuner, "estimate_total_hmm_block_fits", lambda *a, **kw: 0)
    monkeypatch.setattr(hmm_tuner, "score_signal_with_sharpe", lambda signal, **kw: scored.append(signal.copy()) or 1.0)
    cfg = hmm_tuner.HMMVersion1TuningConfig("", "", "", dev_start_year=2019, dev_end_year=2019, validation_years=[2019], verbose=False)
    params = {"num_states": 2, "l2": 1.0, "refit_freq": "month", "train_window_years": 2, "transition_smoothing": 1e-3}
    result = hmm_tuner.evaluate_hmm_candidate(small_panel, cfg, params)
    assert result["information_cutoff"] == "2020-01-01"
    assert fitted[0]["Date"].max() == pd.Timestamp("2019-12-27")
    assert scored[0]["Date"].max() == pd.Timestamp("2019-12-20")
    assert fitted[0].loc[fitted[0]["label_end_20d"] >= "2020-01-01", "y_20d"].isna().all()
    assert len(scored[0]) < len(fitted[0])  # Forecast coverage was retained.


def test_annual_runner_preserves_live_rows_and_core_cutoffs(monkeypatch, small_panel, tmp_path):
    d = small_panel.copy()
    d.loc[d["Date"] >= "2020-01-01", ["y_5d", "y_20d", "fwd_ret_5d", "fwd_ret_20d"]] = np.nan
    models = []
    class RecordingHMM(HMMMultihorizon):
        def fit(self, frame):
            result = super().fit(frame)
            models.append(self)
            return result
    params = {"num_states": 2, "l2": 1.0, "refit_freq": "month", "train_window_years": 2, "transition_smoothing": 1e-3}
    monkeypatch.setattr(hmm_annual_tuner, "tune_hmm_for_year", lambda *a, **kw: {"best_params": params, "manifest_json": "synthetic-tuning"})
    monkeypatch.setattr(hmm_annual_runner, "_import_hmm_runtime_stack", lambda: (RecordingHMM, hmm_tuner.make_hmm_config, lambda *a, **kw: 0.0))
    monkeypatch.setattr(hmm_annual_runner, "_import_overlap_utils", lambda: (None, lambda *a, **kw: {}))
    cfg = hmm_annual_runner.HMMAnnualRunConfig(
        "", "", str(tmp_path), start_year=2020, end_year=2020, verbose=False,
        max_iter_fixed=1, min_history_dates_fixed=4, label_lag_periods_fixed=0,
        save_state_probs=False, save_fit_log=False, save_diagnostics=False,
    )
    result = hmm_annual_runner.run_hmm_year(cfg, 2020, panel=d)
    assert len(result["pred_df"]) == len(d[d["Date"].dt.year == 2020])
    assert result["pred_df"]["up_prob"].notna().all()
    assert result["metrics"]["rows"] == 0
    assert np.isnan(result["metrics"]["accuracy"])
    assert models[0].fit_log_[-1]["information_cutoff"] == "2020-03-06"
    assert models[0].fit_log_[-1]["num_observed_20d"] < models[0].fit_log_[-1]["num_rows"]


def test_full_input_alignment_including_warmup_rows(fitted_model, small_panel):
    rows = small_panel.sample(frac=1, random_state=8)
    expected = rows.merge(fitted_model.row_pred_, on=["Date", "StockID"], how="left", sort=False)["up_prob"]
    actual = fitted_model.predict(rows)
    pd.testing.assert_index_equal(actual.index, rows.index)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual.isna().any() and actual.notna().any()


def test_compact_label_loader_preserves_endpoints(monkeypatch):
    builder = HMMDataBuilder(HMMDataConfig("", "", verbose=False))
    columns = ["Date", "StockID", "Ret_5d", "Ret_20d", "label_end_5d", "label_end_20d", "unused_large_feature"]
    monkeypatch.setattr(builder, "_parquet_columns", lambda path: columns)
    requested = []
    def read(path, columns):
        requested.extend(columns)
        return pd.DataFrame(columns=columns)
    monkeypatch.setattr(pd, "read_parquet", read)
    out = builder._read_fallback_label_subset("synthetic.parquet", need_5d=True, need_20d=True)
    assert {"label_end_5d", "label_end_20d", "fwd_ret_5d", "fwd_ret_20d"}.issubset(out)
    assert "unused_large_feature" not in requested


@pytest.mark.parametrize("freq,expected_date", [("week", "2019-12-20"), ("month", "2019-11-29")])
def test_tuning_scoring_uses_relevant_horizon(small_panel, freq, expected_date):
    cfg = hmm_tuner.HMMVersion1TuningConfig("", "", "", freq=freq)
    pred = small_panel[["Date", "StockID"]].assign(up_prob=0.5)
    out = hmm_tuner._realized_scoring_signal(pred, small_panel, cfg, pd.Timestamp("2020-01-01"))
    assert out["Date"].max() == pd.Timestamp(expected_date)


def test_delayed_tuning_scoring_requires_actual_return_endpoint(small_panel):
    cfg = hmm_tuner.HMMVersion1TuningConfig("", "", "", score_delay=1)
    pred = small_panel[["Date", "StockID"]].assign(up_prob=0.5)
    with pytest.raises(ValueError, match="actual score_end"):
        hmm_tuner._realized_scoring_signal(pred, small_panel, cfg, pd.Timestamp("2020-01-01"))


def test_public_fit_state_is_propagated_from_final_parameters(fitted_model, small_panel):
    bundle = fitted_model.get_fit_bundle()
    train = small_panel[small_panel["Date"].isin(bundle["dates"])]
    train = mask_unrealized_hmm_labels(train, bundle["information_cutoff"])
    mats = fitted_model._build_matrices(train, require_labels=True)
    logB = fitted_model._emission_loglik(mats, bundle["models20"], bundle["models5"])
    _, _, _, filtered = fitted_model._forward_backward(logB, bundle["pi"], bundle["A"])
    np.testing.assert_allclose(bundle["filtered_last"], filtered[-1], atol=1e-11)
    later_dates = sorted(small_panel.loc[small_panel["Date"] > bundle["dates"][-1], "Date"].unique())
    q = filtered[-1]
    for date in later_dates:
        q = q @ bundle["A"]
        q /= q.sum()
        state = fitted_model.state_probs_by_date_
        stored = state.loc[state["Date"] == date].sort_values("state")["prob"]
        if date >= bundle["information_cutoff"]:
            np.testing.assert_allclose(stored, q, atol=1e-11)
    np.testing.assert_allclose(fitted_model._last_q_, q, atol=1e-11)


def test_validation_and_frozen_runner_keep_unlabeled_latest_dates(monkeypatch, small_panel):
    d = small_panel.copy()
    d.loc[d["Date"] >= "2020-01-01", ["y_5d", "y_20d", "fwd_ret_5d", "fwd_ret_20d"]] = np.nan
    validation = hmm_tuner.fit_predict_validation_year(d, model_config(), target_year=2020)
    cfg = hmm_tuner.HMMVersion1TuningConfig(
        "", "", "", dev_start_year=2019, dev_end_year=2019,
        lockbox_start_year=2020, lockbox_end_year=2020, max_iter_fixed=1,
        min_history_dates_fixed=4, label_lag_periods_fixed=0, verbose=False,
    )
    params = {"num_states": 2, "l2": 1.0, "refit_freq": "month", "train_window_years": 2, "transition_smoothing": 1e-3}
    frozen = hmm_tuner.run_frozen_hmm_lockbox(cfg, params, panel=d)
    for pred in (validation, frozen):
        assert len(pred) == len(d[d["Date"].dt.year == 2020])
        assert pred["up_prob"].notna().all()
