"""
Shared scoring functions. EVERY track uses these, so all numbers are comparable.

Two kinds of comparison in this project:
  Line A (history match):  curves  vs  REAL history (RAMP_History)    -> score_vs_history()
  Line B (proxy accuracy): proxy predictions  vs  simulated curves    -> proxy_report()

Conventions (team decision, 10 Oct, set in config.py):
  - curves are on the 41 quarterly HISTORY_DATES (columns t01..t41)
  - only dates >= SCORE_START (1998-10-01) are scored (the first 3 quarters are 0 for everyone)
  - NRMSE = RMSE / (max of the REAL history for that stream)  -> unit-free, comparable across streams
  - combined score = weighted sum of the stream NRMSEs (weights in config.SCORE_WEIGHTS)
"""
import numpy as np
import pandas as pd

from src import config as C

N_DATES = len(C.HISTORY_DATES)
SCORE_MASK = np.asarray(C.HISTORY_DATES >= C.SCORE_START)   # 38 of the 41 dates


# ------------------------------------------------------------ reshaping helpers
def curves_to_arrays(curves: pd.DataFrame, streams=C.CUM_STREAMS) -> tuple[np.ndarray, dict]:
    """curves_long table -> (case numbers, {stream: array (n_cases, 41)})."""
    curves = curves.sort_values(["case", "date"])
    cases = curves["case"].unique()
    arrays = {s: curves[s].to_numpy().reshape(len(cases), N_DATES) for s in streams}
    return cases, arrays


def split_y(y: np.ndarray, streams=C.CUM_STREAMS) -> dict:
    """A y matrix in master-table order (n, 123) -> {stream: array (n, 41)}.
    Use this on proxy predictions before scoring them."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    assert y.shape[1] == N_DATES * len(streams), f"expected {N_DATES * len(streams)} columns, got {y.shape[1]}"
    return {s: y[:, i * N_DATES:(i + 1) * N_DATES] for i, s in enumerate(streams)}


def history_arrays(history: pd.DataFrame, streams=C.CUM_STREAMS) -> dict:
    """history table -> {stream: array (41,)} on HISTORY_DATES."""
    h = history.set_index("date").reindex(C.HISTORY_DATES)
    return {s: h[s].to_numpy(dtype=float) for s in streams}


# ------------------------------------------------------------ core metric
def nrmse(pred: np.ndarray, true: np.ndarray, scale: float) -> np.ndarray:
    """RMSE over the scored dates, divided by `scale`.
    pred: (n, 41) or (41,);  true: (41,) or same shape as pred.  Returns (n,) or a scalar."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    err = (pred - true)[..., SCORE_MASK]
    return np.sqrt(np.mean(err ** 2, axis=-1)) / scale


# ------------------------------------------------------------ Line A: match against REAL history
def score_vs_history(pred: dict, history: pd.DataFrame, streams=None, weights=None) -> pd.DataFrame:
    """Score curves against the real field history.

    pred    : {stream: array (n, 41)}  e.g. from curves_to_arrays() or split_y(proxy.predict(X))
    returns : DataFrame (n rows) with one nrmse_<stream> column per stream + 'score' (lower = better)
    """
    streams = streams or C.SCORE_STREAMS
    weights = weights or C.SCORE_WEIGHTS
    hist = history_arrays(history, streams)
    out = {}
    for s in streams:
        scale = np.max(hist[s][SCORE_MASK])
        out[f"nrmse_{s}"] = np.atleast_1d(nrmse(pred[s], hist[s], scale))
    df = pd.DataFrame(out)
    df["score"] = sum(weights.get(s, 0.0) * df[f"nrmse_{s}"] for s in streams)
    return df


# ------------------------------------------------------------ Line B: proxy vs simulator
def proxy_report(y_true: np.ndarray, y_pred: np.ndarray, history: pd.DataFrame,
                 streams=C.CUM_STREAMS) -> pd.DataFrame:
    """How well a proxy reproduces the SIMULATED curves (use on validation / test cases).

    y_true, y_pred : (n, 123) in master-table column order
    returns one row per stream:
      nrmse_mean     mean NRMSE over cases (same scale as the history score)
      r2             R^2 over all scored points
      end_pct_err    mean |error| of the 2008 cumulative value, in % of the true value
    """
    t, p = split_y(y_true, streams), split_y(y_pred, streams)
    hist = history_arrays(history, streams)
    rows = []
    for s in streams:
        scale = np.max(hist[s][SCORE_MASK])
        tt, pp = t[s][:, SCORE_MASK], p[s][:, SCORE_MASK]
        ss_res = np.sum((tt - pp) ** 2)
        ss_tot = np.sum((tt - tt.mean()) ** 2)
        end_err = np.abs(p[s][:, -1] - t[s][:, -1]) / np.abs(t[s][:, -1]) * 100
        rows.append({"stream": s,
                     "nrmse_mean": float(np.mean(nrmse(p[s], t[s], scale))),
                     "r2": float(1 - ss_res / ss_tot),
                     "end_pct_err": float(np.mean(end_err))})
    return pd.DataFrame(rows).set_index("stream")