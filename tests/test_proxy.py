"""Synthetic-data checks for the stage-1 proxy; no competition test data used."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.load_data import y_columns
from src.metrics import N_DATES
from src.proxy import ProxyPipeline, candidates, cv_splits, select_candidate, training_rows
from threadpoolctl import threadpool_limits


def synthetic():
    rng = np.random.default_rng(42)
    X = rng.uniform(0.2, 0.8, size=(30, 4))
    t = np.maximum(np.arange(N_DATES) - 3, 0)
    oil = (10 + X[:, 0:1]*2 + X[:, 1:2]) * t
    gas = oil * 0.363
    water = (2 + X[:, 2:3]) * t**2
    return X, np.concatenate([oil, gas, water], axis=1)


def test_scaling_fitted_on_fit_rows_and_unchanged_by_extreme_predictions():
    X, y = synthetic()
    p = ProxyPipeline().fit(X[:20], y[:20])
    xmin, xmax = p.xmin_.copy(), p.xmax_.copy()
    scales = [p.streams_[s]["scale"] for s in C.CUM_STREAMS]
    np.testing.assert_array_equal(xmin, X[:20].min(axis=0))
    np.testing.assert_array_equal(xmax, X[:20].max(axis=0))
    np.testing.assert_allclose(scales, [b.max() for b in np.split(y[:20], 3, axis=1)])
    p.predict(np.full((2, 4), 1000.0))
    np.testing.assert_array_equal(p.xmin_, xmin)
    np.testing.assert_array_equal(p.xmax_, xmax)
    assert scales == [p.streams_[s]["scale"] for s in C.CUM_STREAMS]


def test_mean_baseline_uses_training_rows_only():
    X, y = synthetic()
    p = ProxyPipeline(model="dumb").fit(X[:20], y[:20])
    np.testing.assert_allclose(p.predict(X[20:]), np.tile(y[:20].mean(axis=0), (10, 1)))


def test_ridge_recovers_linear_curves_in_original_units_and_output_order():
    X, y = synthetic()
    p = ProxyPipeline(alpha=1e-10).fit(X[:20], y[:20])
    np.testing.assert_allclose(p.predict(X[20:]), y[20:], atol=1e-5)
    assert p.predict(X[20:]).shape == (10, 123)


def test_named_inputs_are_reordered_and_bad_schema_rejected():
    X, y = synthetic()
    p = ProxyPipeline().fit(X, y)
    frame = pd.DataFrame(X, columns=C.PARAM_COLS)
    np.testing.assert_allclose(p.predict(frame[C.PARAM_COLS[::-1]]), p.predict(X))
    with pytest.raises(ValueError):
        p.predict(frame.rename(columns={C.PARAM_COLS[0]: "wrong"}))
    with pytest.raises(ValueError):
        p.predict(np.full((1, 4), np.nan))


def test_physical_corrections_report_changes_and_preserve_zero_prefix():
    X, y = synthetic()
    p = ProxyPipeline().fit(X, y)
    raw = y[:1].copy()
    raw[0, 0] = 100
    raw[0, 7] = -20
    fixed = p.correct_predictions(raw).reshape(1, 3, N_DATES)
    assert p.last_n_fixed > 0
    assert p.last_max_correction_scaled > 0
    assert (fixed >= 0).all() and (np.diff(fixed, axis=2) >= 0).all()
    assert (fixed[:, :, :4] == 0).all()
    assert raw[0, 0] == 100  # The audit's raw predictions remain available.


def test_cv_holds_out_each_training_case_once():
    ids = np.arange(1, 71)
    held = []
    for fit_idx, held_idx in cv_splits(ids):
        assert len(fit_idx) == 56 and len(held_idx) == 14
        assert not set(fit_idx) & set(held_idx)
        held.extend(ids[held_idx])
    assert sorted(held) == list(ids)
    with pytest.raises(ValueError):
        cv_splits(np.arange(31, 101))


def test_validation_and_test_targets_do_not_enter_training_rows():
    columns = ["case", "split"] + C.PARAM_COLS + y_columns()
    master = pd.DataFrame(1.0, index=range(100), columns=columns[2:])
    master.insert(0, "split", [C.split_of(i) for i in range(1, 101)])
    master.insert(0, "case", range(1, 101))
    first = training_rows(master)
    master.loc[master.split != "train", C.PARAM_COLS+y_columns()] = np.nan
    pd.testing.assert_frame_equal(training_rows(master), first)


@pytest.mark.parametrize("model", ["ridge", "gp", "xgb"])
def test_constant_inputs_and_streams_are_safe(model):
    X, y = synthetic()
    X[:, 0] = 1
    y[:, N_DATES:2*N_DATES] = 0
    with threadpool_limits(limits=1):
        p = ProxyPipeline(model=model, gp_restarts=0, n_estimators=20).fit(X, y)
    pred = p.predict(X)
    assert np.isfinite(pred).all()
    assert (pred[:, N_DATES:2*N_DATES] == 0).all()


@pytest.mark.parametrize("model", ["gp", "xgb"])
def test_added_models_learn_without_mutating_scales_on_heldout_inputs(model):
    X, y = synthetic()
    with threadpool_limits(limits=1):
        p = ProxyPipeline(model=model, gp_restarts=0).fit(X[:20], y[:20])
        pred = p.predict(X[20:])
    baseline = np.tile(y[:20].mean(axis=0), (10, 1))
    for true, fitted, mean in zip(np.split(y[20:], 3, axis=1), np.split(pred, 3, axis=1), np.split(baseline, 3, axis=1)):
        assert np.mean((true-fitted)**2) < np.mean((true-mean)**2)
    np.testing.assert_array_equal(p.xmin_, X[:20].min(axis=0))
    np.testing.assert_array_equal(p.xmax_, X[:20].max(axis=0))
    assert pred.shape == (10, 123) and np.isfinite(pred).all()
    assert (np.diff(pred.reshape(-1, 3, N_DATES), axis=2) >= 0).all()


def test_extra_pca_components_are_fold_local_and_capped():
    X, y = synthetic()
    auto = ProxyPipeline(max_components=2).fit(X[:20], y[:20])
    extra = ProxyPipeline(max_components=2, pca_extra=2).fit(X[:20], y[:20])
    for stream in C.CUM_STREAMS:
        assert extra.streams_[stream]["auto_components"] == auto.streams_[stream]["n_components"]
        assert extra.streams_[stream]["n_components"] == min(auto.streams_[stream]["n_components"] + 2, 2)
        assert extra.streams_[stream]["pca_error"] <= auto.streams_[stream]["pca_error"] + 1e-14


def test_selection_prefers_simpler_family_only_within_ten_percent():
    summary = pd.DataFrame([
        {"candidate": "mean", "model": "dumb", "selection_metric": 0.1},
        {"candidate": "ridge_best", "model": "ridge", "selection_metric": 0.00109},
        {"candidate": "ridge_worse", "model": "ridge", "selection_metric": 0.01},
        {"candidate": "gp_best", "model": "gp", "selection_metric": 0.001},
    ])
    assert select_candidate(summary).candidate == "ridge_best"
    summary.loc[summary.candidate == "ridge_best", "selection_metric"] = 0.00111
    assert select_candidate(summary).candidate == "gp_best"
    assert len(candidates("baseline")) == 7
    assert len(candidates("extended")) == 15
