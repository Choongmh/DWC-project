"""Export and one-time evaluation safeguards, using synthetic data only."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.load_data import y_columns
from src import proxy, proxy_final


@pytest.fixture
def exported(tmp_path, monkeypatch):
    X = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [.2, .7, .4, .8]], dtype=float)
    curve = np.tile(np.arange(41, dtype=float), 3)
    y = np.array([curve, 2*curve, 1.4*curve])
    model = proxy.ProxyPipeline(model="ridge", pca_extra=2).fit(X, y)
    model.search_bounds_ = np.array([[0., 1.]]*4)
    model.input_columns_ = list(C.PARAM_COLS)
    model.output_columns_ = y_columns()
    path = tmp_path / "proxy.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(proxy, "_CACHED_PROXY", None)
    return model, path


def test_roundtrip_reordering_batching_and_bounds_copy(exported):
    model, path = exported
    loaded = proxy.load_proxy(path)
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (2051, 4))
    frame = pd.DataFrame(X, columns=C.PARAM_COLS).iloc[:, ::-1]
    np.testing.assert_allclose(proxy.predict(frame), model.predict(X), atol=1e-10)
    assert loaded is proxy._CACHED_PROXY
    bounds = proxy.param_bounds()
    bounds.iloc[0, 0] = -999
    assert proxy.param_bounds().iloc[0, 0] == 0
    assert proxy.predict(X).shape == (2051, 123)


def test_outside_bounds_warns_without_clipping_and_rejects_bad_inputs(exported):
    model, path = exported
    proxy.load_proxy(path)
    X = np.array([[2, .3, .4, .5]])
    with pytest.warns(UserWarning, match="outside the declared"):
        actual = proxy.predict(X)
    np.testing.assert_allclose(actual, model.predict(X))
    for bad in [np.ones((2, 3)), [[np.nan]*4], np.ones(4)]:
        with pytest.raises(ValueError):
            proxy.predict(bad)
    with pytest.raises(NotImplementedError):
        proxy.predict(np.ones((1, 4)), return_std=True)


def test_uncertainty_batching_and_serialization(tmp_path, monkeypatch):
    X = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [.2, .7, .4, .8]], dtype=float)
    curve = np.tile(np.arange(41, dtype=float), 3)
    model = proxy.ProxyPipeline(model="gp", gp_restarts=0, max_components=1).fit(X, np.array([curve, 2*curve, 1.4*curve]))
    model.search_bounds_ = np.array([[0., 1.]]*4)
    model.input_columns_, model.output_columns_ = list(C.PARAM_COLS), y_columns()
    path = tmp_path / "gp.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(proxy, "_CACHED_PROXY", None)
    proxy.load_proxy(path)
    queries = np.tile([[.3, .4, .5, .6]], (2050, 1))
    mean, std = proxy.predict(queries, return_std=True)
    expected, expected_std = model.predict(queries[:1], return_std=True)
    np.testing.assert_allclose(mean, np.repeat(expected, len(queries), axis=0), atol=1e-10)
    np.testing.assert_allclose(std, np.repeat(expected_std, len(queries), axis=0), atol=1e-8)
    assert (std >= 0).all()


def test_rejects_wrong_schema_without_replacing_cache(exported, tmp_path):
    model, path = exported
    cached = proxy.load_proxy(path)
    model.input_columns_ = list(reversed(C.PARAM_COLS))
    bad = tmp_path / "bad.joblib"
    joblib.dump(model, bad)
    with pytest.raises(ValueError, match="schema"):
        proxy.load_proxy(bad)
    assert proxy._CACHED_PROXY is cached


def test_fit85_does_not_use_test_targets():
    master = pd.DataFrame(1., index=range(100), columns=C.PARAM_COLS+y_columns())
    master.insert(0, "split", [C.split_of(i) for i in range(1, 101)])
    master.insert(0, "case", range(1, 101))
    master.loc[master.split == "test", y_columns()] = np.nan
    fit = proxy_final.final_fit_rows(master)
    assert fit.case.tolist() == list(range(1, 86))
    assert np.isfinite(fit[y_columns()].to_numpy()).all()


def test_durable_marker_and_repeat_guard_precede_data_access(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    proxy_final.claim_test(state, {"status": "started"})
    with pytest.raises(FileExistsError):
        proxy_final.claim_test(state, {"status": "overwritten"})
    assert json.loads(state.read_text())["status"] == "started"
    monkeypatch.setattr(proxy_final, "STATE", state)
    monkeypatch.setattr(proxy_final, "read_master", lambda: pytest.fail("Must not access data again"))
    with pytest.raises(RuntimeError, match="already started"):
        proxy_final.evaluate_once()
