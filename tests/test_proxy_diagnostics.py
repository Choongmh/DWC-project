"""Synthetic checks for GP uncertainty and gas diagnostics; no field test cases."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from src import config as C
from src.metrics import N_DATES
from src.proxy import GPOptimizerTrace, ProxyPipeline
from src.proxy_diagnostics import coverage_rows, gas_ratio_summary, paired_ratio_error


def toy_data():
    X = np.random.default_rng(123).uniform(0.2, 0.8, (18, 4))
    t = np.maximum(np.arange(N_DATES)-3, 0)
    oil = (10+X[:, :1])*t
    return X, np.concatenate([oil, oil*0.363, (1+X[:, 1:2])*t**2], axis=1)


def test_traced_optimizer_preserves_default_fit_and_reports_selected_attempt():
    X, y = toy_data()
    with threadpool_limits(limits=1):
        standard = ProxyPipeline(model="gp", gp_restarts=1).fit(X, y)
        traced = ProxyPipeline(model="gp", gp_restarts=1, gp_trace_optimizer=True).fit(X, y)
    np.testing.assert_allclose(standard.predict(X), traced.predict(X), rtol=1e-10, atol=1e-10)
    for state in traced.streams_.values():
        for regressor, audit in zip(state["regressor"], state["gp_optimizer_audit"]):
            assert len(audit["runs"]) == 2
            selected = audit["runs"][audit["selected_run"]]
            assert selected["objective"] == min(r["objective"] for r in audit["runs"])
            np.testing.assert_allclose(regressor.kernel_.theta, selected["theta"])


def test_uncertainty_reconstruction_matches_independent_component_variance():
    # Hand-check two independent coordinates: variances add after squared loadings.
    class ConstantRegressor:
        def __init__(self, mean, std):
            self.mean, self.std = mean, std

        def predict(self, X, return_std=False):
            mean = np.full(len(X), self.mean)
            return (mean, np.full(len(X), self.std)) if return_std else mean

    model = ProxyPipeline(model="gp")
    model.xmin_, model.xrange_ = np.zeros(4), np.ones(4)
    model.streams_ = {}
    for s in C.CUM_STREAMS:
        model.streams_[s] = {"scale": 10, "n_components": 2,
                             "pca": SimpleNamespace(components_=np.vstack([np.ones(N_DATES)*0.6, np.ones(N_DATES)*0.8]), mean_=np.ones(N_DATES)),
                             "regressor": [ConstantRegressor(2, 3), ConstantRegressor(4, 5)],
                             "zero_prefix": np.zeros(N_DATES, dtype=bool)}
    mean, sd = model.predict_raw(np.ones((2, 4)), return_std=True)
    np.testing.assert_allclose(mean, 54)
    np.testing.assert_allclose(sd, np.sqrt((3*0.6)**2 + (5*0.8)**2)*10)
    corrected, sd2 = model.predict(np.ones((2, 4)), return_std=True)
    np.testing.assert_allclose(mean, corrected)
    np.testing.assert_array_equal(sd, sd2)


def test_gp_std_units_and_point_predictions_are_consistent():
    X, y = toy_data()
    with threadpool_limits(limits=1):
        p = ProxyPipeline(model="gp", gp_restarts=0).fit(X[:12], y[:12])
    raw, sd = p.predict_raw(X[12:], return_std=True)
    np.testing.assert_allclose(raw, p.predict_raw(X[12:]), atol=1e-9)
    assert sd.shape == raw.shape == (6, 123)
    assert np.isfinite(sd).all() and (sd >= 0).all() and (sd > 0).any()
    q = deepcopy(p)
    for state in q.streams_.values():
        state["scale"] *= 100
    raw100, sd100 = q.predict_raw(X[12:], return_std=True)
    np.testing.assert_allclose(raw100, raw*100)
    np.testing.assert_allclose(sd100, sd*100)


def test_uncertainty_constant_stream_and_non_gp():
    X, y = toy_data()
    y[:, N_DATES:2*N_DATES] = 0
    with threadpool_limits(limits=1):
        p = ProxyPipeline(model="gp", gp_restarts=0).fit(X, y)
    mean, sd = p.predict(X[:2], return_std=True)
    assert (mean[:, N_DATES:2*N_DATES] == 0).all()
    assert (sd[:, N_DATES:2*N_DATES] == 0).all()
    with pytest.raises(NotImplementedError):
        ProxyPipeline(model="ridge").fit(X, y).predict(X, return_std=True)


def test_gas_check_excludes_zero_oil_and_detects_inconsistent_predictions():
    _, y = toy_data()
    r = gas_ratio_summary(y, 1e-6)
    assert r["n_excluded"] > 0
    assert r["median_ratio"] == pytest.approx(0.363)
    assert r["within_spec_0_1_percent"]
    bad = y.copy()
    bad[:, N_DATES:2*N_DATES] *= 1.1
    b = gas_ratio_summary(bad, 1e-6, reference=r["median_ratio"])
    assert not b["within_spec_0_1_percent"]
    assert paired_ratio_error(y, bad, 1e-6)["paired_mean_relative_error"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        gas_ratio_summary(np.zeros_like(y), 1e-6)


def test_coverage_separates_zero_production_and_raw_from_corrected():
    _, y = toy_data()
    raw, fixed, sd = y+100, y.copy(), np.ones_like(y)
    scales = {s: np.ones(len(y)) for s in C.CUM_STREAMS}
    rows = coverage_rows(y, raw, fixed, sd, scales)
    assert all(r["coverage_2std"] == (0 if r["center"] == "raw" else 1) for r in rows)
    for s in C.CUM_STREAMS:
        selected = [r for r in rows if r["stream"] == s and r["center"] == "raw"]
        assert selected[0]["n_points"] > selected[1]["n_points"]
