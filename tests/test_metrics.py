"""Small checks that the shared scoring behaves as expected. Run: python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.metrics import N_DATES, SCORE_MASK, nrmse, proxy_report, score_vs_history, split_y


def fake_history():
    ramp = np.linspace(0, 100, N_DATES)
    data = {"date": C.HISTORY_DATES}
    for s in C.CUM_STREAMS + C.RATE_STREAMS:
        data[s] = ramp
    return pd.DataFrame(data)


def test_score_mask_skips_first_three_quarters():
    assert SCORE_MASK.sum() == N_DATES - 3
    assert not SCORE_MASK[:3].any()


def test_perfect_match_scores_zero():
    h = fake_history()
    pred = {s: h[s].to_numpy()[None, :] for s in C.CUM_STREAMS}
    assert score_vs_history(pred, h)["score"].iloc[0] == pytest.approx(0.0)


def test_constant_offset_gives_expected_nrmse():
    h = fake_history()
    pred = {s: h[s].to_numpy()[None, :] + 10 for s in C.CUM_STREAMS}   # +10 everywhere, max = 100
    res = score_vs_history(pred, h)
    assert res["nrmse_oil_cum"].iloc[0] == pytest.approx(0.10)
    assert res["score"].iloc[0] == pytest.approx(0.10)                 # weights sum to 1


def test_differences_before_score_start_are_ignored():
    true = np.linspace(0, 100, N_DATES)
    pred = true.copy()
    pred[:3] += 1000
    assert nrmse(pred, true, 100) == pytest.approx(0.0)


def test_weights_sum_to_one():
    assert sum(C.SCORE_WEIGHTS.values()) == pytest.approx(1.0)


def test_split_y_and_proxy_report_shapes():
    y = np.random.default_rng(C.SEED).random((5, 3 * N_DATES)) + 1
    parts = split_y(y)
    assert all(v.shape == (5, N_DATES) for v in parts.values())
    rep = proxy_report(y, y, fake_history())
    assert np.allclose(rep["nrmse_mean"], 0) and np.allclose(rep["r2"], 1)