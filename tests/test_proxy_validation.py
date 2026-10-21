"""Check validation calculations on synthetic data, never competition test data."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.load_data import y_columns
from src.metrics import N_DATES
from src.proxy_validation import cumulative_rates, ranking_report, stream_diagnostics, validation_rows


def test_rates_use_actual_days_and_interval_ending_convention():
    days = (C.HISTORY_DATES-C.HISTORY_DATES[0]).days.to_numpy()
    cumulative = np.array([7*days, 19*days], dtype=float)
    rates = cumulative_rates(cumulative)
    assert np.isnan(rates[:, 0]).all()
    np.testing.assert_allclose(rates[0, 1:], 7)
    np.testing.assert_allclose(rates[1, 1:], 19)
    assert len(set(np.diff(days))) > 1  # Quarters have different lengths.


def test_ranking_checks_score_error_even_when_ranks_are_perfect():
    true = np.array([.01, .02, .03, .04])
    table, result = ranking_report([71,72,73,74], true, true+.004)
    assert result['spearman_rho'] == pytest.approx(1)
    assert result['max_abs_score_error'] > .003
    assert result['true_best_case'] == result['proxy_best_case'] == 71
    assert result['top_3_overlap'] == 3
    _, reversed_result = ranking_report([71,72,73,74], true, true[::-1])
    assert reversed_result['spearman_rho'] == pytest.approx(-1)
    assert reversed_result['max_rank_displacement'] == 3


def test_late_and_endpoint_metrics_distinguish_pooled_from_mean_case():
    true = np.tile(np.arange(N_DATES, dtype=float)+1, (2,1))
    pred = true.copy()
    pred[1, -8:] += 2
    result, _, _ = stream_diagnostics(true, pred, history_scale=100)
    assert result['late_nrmse_pooled'] == pytest.approx(np.sqrt(2)/100)
    assert result['late_nrmse_mean_case'] == pytest.approx(.01)
    assert result['end_pct_err_mean'] == pytest.approx(100/41)
    assert result['end_pct_err_max'] == pytest.approx(200/41)


def test_validation_selection_ignores_test_values_and_rejects_bad_validation():
    master=pd.DataFrame(1., index=range(100), columns=C.PARAM_COLS+y_columns())
    master.insert(0, 'split', [C.split_of(i) for i in range(1,101)])
    master.insert(0, 'case', range(1,101))
    t, v = validation_rows(master)
    assert list(t.case)==list(range(1,71))
    assert list(v.case)==list(range(71,86))
    master.loc[master.split=='test', C.PARAM_COLS+y_columns()]=np.nan
    t2, v2 = validation_rows(master)
    pd.testing.assert_frame_equal(t,t2)
    pd.testing.assert_frame_equal(v,v2)
    master.loc[master['case']==71, C.PARAM_COLS[0]]=np.nan
    with pytest.raises(ValueError):
        validation_rows(master)
