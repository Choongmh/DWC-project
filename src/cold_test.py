"""T3 Task E: rate check (Part 1) and time-direction cold test (Part 2, oil only).

Part 1 (read-only): compare last observed / proxy / first-forecast oil rates.
Part 2: fit Arps on proxy rates up to a cutoff, forecast forward to 2008-01-01,
        compare with the same proxy curve and with observed history.

NOTE ON INTERPRETATION:
    The cold test extrapolates 2, 3, or 4 years.  This is an OPTIMISTIC (EASIER)
    test than the real forecast, which extrapolates 20 years.  These short-horizon errors do not establish 20-year accuracy;
    longer-term errors may be larger.

Run: .venv/bin/python -m src.cold_test
"""
from __future__ import annotations

import warnings
from itertools import product as iproduct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src import config as C
from src.load_data import read_history

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FIT_START = pd.Timestamp(C.T3_ARPS_FIT_START)   # 2003-01-01
_COLD_CUTOFFS = {
    "2004-01-01": pd.Timestamp("2004-01-01"),
    "2005-01-01": pd.Timestamp("2005-01-01"),
    "2006-01-01": pd.Timestamp("2006-01-01"),
}
_FORECAST_END = pd.Timestamp("2008-01-01")

# Minimum quarterly points required to fit the 3-parameter (free-b) model via AICc
_MIN_PTS_FREE_B = 8

# Late window starts for main forecast sensitivity
_LATE_WINDOW_STARTS = ["2005-01-01", "2006-01-01"]

# Optimisation bounds (scaled rates, Di in 1/year)
_BND_EXP    = ([0.3, 1e-3],       [3.0, 20.0])            # b=0 fixed: qi_s, Di
_BND_HAR    = ([0.3, 1e-3],       [3.0, 20.0])            # b=1 fixed: qi_s, Di
_BND_FREE   = ([0.3, 1e-3, 0.0],  [3.0, 20.0, C.T3_ARPS_B_MAX])  # free b

_P0_2 = [(q, d) for q, d in iproduct((0.8, 1.0, 1.2), (0.05, 0.15, 0.4, 1.0, 3.0))]
_P0_3 = [(q, d, b) for q, d, b in iproduct(
    (0.8, 1.0, 1.2), (0.05, 0.15, 0.5, 1.5), (0.0, 0.3, 0.6, 1.0)
)]

# ---------------------------------------------------------------------------
# Arps functions (scaled rates, t in years from t0)
# ---------------------------------------------------------------------------

def _hyp(t: np.ndarray, qi_s: float, Di: float, b: float) -> np.ndarray:
    """Hyperbolic: qi_s / (1 + b*Di*t)^(1/b).  b < 1e-8 -> exponential fallback."""
    t = np.asarray(t, float)
    if b < 1e-8:
        return qi_s * np.exp(-Di * t)
    with np.errstate(invalid="ignore", over="ignore"):
        base = np.maximum(1.0 + b * Di * t, 1e-300)
        return np.clip(qi_s / np.power(base, 1.0 / b), 0.0, None)


def _exp(t: np.ndarray, qi_s: float, Di: float) -> np.ndarray:
    """Exponential: qi_s * exp(-Di*t), b=0 fixed."""
    return qi_s * np.exp(-np.asarray(t, float) * Di)


def _har(t: np.ndarray, qi_s: float, Di: float) -> np.ndarray:
    """Harmonic: qi_s / (1 + Di*t), b=1 fixed."""
    return np.clip(qi_s / np.maximum(1.0 + Di * np.asarray(t, float), 1e-300), 0.0, None)


def _aicc(sse: float, n: int, k: int) -> float:
    if not np.isfinite(sse) or sse <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _best_fit(func, p0_list, bounds, t, q):
    """Try every starting guess; return (best_popt, best_sse) or (None, inf)."""
    best_sse, best_popt = np.inf, None
    for p0 in p0_list:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(func, t, q, p0=p0, bounds=bounds,
                                    maxfev=30_000, method="trf")
            res = q - func(t, *popt)
            sse = float(np.dot(res, res))
            if sse < best_sse:
                best_sse, best_popt = sse, popt
        except Exception:
            pass
    return best_popt, best_sse


# ---------------------------------------------------------------------------
# Fit three methods for one window
# ---------------------------------------------------------------------------

def _fit_three(t_yr: np.ndarray, q_sc: np.ndarray, n: int) -> dict:
    """Return {'exp', 'har', 'free'} fits.  'free' is omitted when n < _MIN_PTS_FREE_B.

    Each entry: {'params': (qi_s, Di, b), 'sse', 'aicc', 'success'}
    """
    out: dict = {}

    popt, sse = _best_fit(_exp, _P0_2, _BND_EXP, t_yr, q_sc)
    out["exp"] = ({"params": (popt[0], popt[1], 0.0), "sse": sse,
                   "aicc": _aicc(sse, n, 2), "success": True}
                  if popt is not None
                  else {"params": None, "sse": np.inf, "aicc": np.inf, "success": False})

    popt, sse = _best_fit(_har, _P0_2, _BND_HAR, t_yr, q_sc)
    out["har"] = ({"params": (popt[0], popt[1], 1.0), "sse": sse,
                   "aicc": _aicc(sse, n, 2), "success": True}
                  if popt is not None
                  else {"params": None, "sse": np.inf, "aicc": np.inf, "success": False})

    if n >= _MIN_PTS_FREE_B:
        popt, sse = _best_fit(_hyp, _P0_3, _BND_FREE, t_yr, q_sc)
        out["free"] = ({"params": tuple(popt), "sse": sse,
                        "aicc": _aicc(sse, n, 3), "success": True}
                       if popt is not None
                       else {"params": None, "sse": np.inf, "aicc": np.inf, "success": False})

    return out


# ---------------------------------------------------------------------------
# Arps projection and cumulative integration
# ---------------------------------------------------------------------------

def _project(params: tuple, scale: float, t0: pd.Timestamp,
             dates: pd.DatetimeIndex) -> np.ndarray:
    """Evaluate Arps at each date (t in years relative to t0) and un-scale."""
    qi_s, Di, b = params
    t = np.array([(d - t0).days / 365.25 for d in dates])
    return _hyp(t, qi_s, Di, b) * scale


def _integrate(rates: np.ndarray, dates: pd.DatetimeIndex,
               cum_start: float, start_date: pd.Timestamp) -> np.ndarray:
    """Quarter-by-quarter integration from start_date with cum_start."""
    prev, cum, cums = start_date, cum_start, []
    for d, r in zip(dates, rates):
        cum += r * (d - prev).days
        cums.append(cum)
        prev = d
    return np.array(cums)


# ---------------------------------------------------------------------------
# Part 1: rate check table
# ---------------------------------------------------------------------------

def print_rate_check(curves: pd.DataFrame, history: pd.DataFrame,
                     forecast: pd.DataFrame) -> None:
    """Print observed / proxy / first-forecast oil rates at the history/forecast boundary."""
    h = history.set_index("date")
    obs_rate = float(h.loc["2008-01-01", "oil_rate"])
    ids = sorted(curves["id"].unique())

    print("\n=== Part 1: Oil rate check at the history/forecast boundary ===")
    print("Proxy rates are proxy-estimated (not simulator-verified).")
    print("  All rates in STB/d.  % = (value - observed) / observed * 100.")
    print(f"\n{'id':>3}  {'obs (2008-01-01)':>18}  {'proxy (2008-01-01)':>20}  "
          f"{'proxy vs obs':>13}  {'1st fc (2008-04-01)':>21}  {'1st fc vs obs':>14}")
    for id_ in ids:
        g = curves[curves["id"] == id_].set_index("date")
        proxy_rate = float(g.loc["2008-01-01", "oil_rate"])
        fc = forecast[forecast["id"] == id_].set_index("date")
        fc_rate = float(fc.loc["2008-04-01", "oil_rate"])
        proxy_pct = (proxy_rate - obs_rate) / obs_rate * 100
        fc_pct = (fc_rate - obs_rate) / obs_rate * 100
        print(f"{id_:>3}  {obs_rate:>18.1f}  {proxy_rate:>20.1f}  {proxy_pct:>+12.1f}%"
              f"  {fc_rate:>21.1f}  {fc_pct:>+13.1f}%")
    print("\n  Note: first forecast (2008-04-01) is one quarter after the last observed (2008-01-01).")
    print("  The decline between the two dates is expected; + means over-estimate vs observed.")


# ---------------------------------------------------------------------------
# Part 2: cold test for one cutoff
# ---------------------------------------------------------------------------

def _run_one_cutoff(cutoff: pd.Timestamp, curves: pd.DataFrame,
                    history: pd.DataFrame) -> dict:
    """Fit Arps up to cutoff and project forward to 2008-01-01.

    Returns dict keyed by id, each containing fit results and error metrics.
    """
    h = history.set_index("date")
    ids = sorted(curves["id"].unique())

    fc_dates = pd.date_range(
        cutoff + pd.DateOffset(months=3), "2008-01-01", freq="QS"
    )

    obs_cum_at_cutoff = float(h.loc[cutoff, "oil_cum"])
    obs_cum_at_end    = float(h.loc["2008-01-01", "oil_cum"])
    obs_rate_at_end   = float(h.loc["2008-01-01", "oil_rate"])
    obs_cum_gained    = obs_cum_at_end - obs_cum_at_cutoff

    results: dict = {}
    for id_ in ids:
        g = curves[curves["id"] == id_].set_index("date")

        proxy_cum_at_cutoff = float(g.loc[cutoff, "oil_cum"])
        proxy_cum_at_end    = float(g.loc["2008-01-01", "oil_cum"])
        proxy_rate_at_end   = float(g.loc["2008-01-01", "oil_rate"])
        proxy_cum_gained    = proxy_cum_at_end - proxy_cum_at_cutoff

        # Fit window: _FIT_START to cutoff
        fit_rates = g.loc[_FIT_START:cutoff, "oil_rate"]
        n = len(fit_rates)
        t0 = _FIT_START
        t_yr = np.array([(d - t0).days / 365.25 for d in fit_rates.index])
        scale = float(fit_rates.iloc[0])
        q_sc = fit_rates.to_numpy() / scale

        fits = _fit_three(t_yr, q_sc, n)

        id_res: dict = {
            "n": n,
            "fc_dates": fc_dates,
            "scale": scale,
            "t0": t0,
            "proxy_rate_at_end": proxy_rate_at_end,
            "proxy_cum_gained": proxy_cum_gained,
            "obs_rate_at_end": obs_rate_at_end,
            "obs_cum_gained": obs_cum_gained,
            "proxy_cum_at_cutoff": proxy_cum_at_cutoff,
            "obs_cum_at_cutoff": obs_cum_at_cutoff,
            "methods": {},
        }

        for mname, fit in fits.items():
            if not fit["success"]:
                id_res["methods"][mname] = None
                continue

            rates = _project(fit["params"], scale, t0, fc_dates)
            # Cumulative starting from proxy cum at cutoff
            cums_p = _integrate(rates, fc_dates, proxy_cum_at_cutoff, cutoff)
            # Cumulative starting from observed cum at cutoff
            cums_o = _integrate(rates, fc_dates, obs_cum_at_cutoff, cutoff)

            fc_rate_at_end = float(rates[-1])
            fc_gained_p    = float(cums_p[-1]) - proxy_cum_at_cutoff
            fc_gained_o    = float(cums_o[-1]) - obs_cum_at_cutoff

            id_res["methods"][mname] = {
                "params":          fit["params"],
                "rates_fc":        rates,
                "cums_p":          cums_p,
                "fc_rate_at_end":  fc_rate_at_end,
                "rate_err_proxy":  (fc_rate_at_end - proxy_rate_at_end) / proxy_rate_at_end * 100,
                "rate_err_obs":    (fc_rate_at_end - obs_rate_at_end)   / obs_rate_at_end   * 100,
                "cum_err_proxy":   (fc_gained_p - proxy_cum_gained) / proxy_cum_gained * 100,
                "cum_err_obs":     (fc_gained_o - obs_cum_gained)   / obs_cum_gained   * 100,
            }

        results[id_] = id_res
    return results


def _print_cutoff_table(cutoff_str: str, results: dict) -> dict:
    """Print error table for one cutoff.  Return {method: {metric: [values]}} for summary."""
    cutoff   = pd.Timestamp(cutoff_str)
    horizon  = round((pd.Timestamp("2008-01-01") - cutoff).days / 365.25)
    ids      = sorted(results.keys())
    n_pts    = results[ids[0]]["n"]
    methods  = list(results[ids[0]]["methods"].keys())

    print(f"\n=== Cold test: fit {C.T3_ARPS_FIT_START} to {cutoff_str}, "
          f"forecast to 2008-01-01 "
          f"(horizon = {horizon} yr, n_fit = {n_pts} pts) ===")
    print("NOTE: This is an OPTIMISTIC test -- the real forecast extrapolates 20 years (much harder).")

    if n_pts < _MIN_PTS_FREE_B:
        print(f"  Free-b fit SKIPPED: only {n_pts} fit points "
              f"(need >= {_MIN_PTS_FREE_B} for a reliable 3-parameter AICc comparison).")

    errors: dict = {m: {"rate_proxy": [], "rate_obs": [],
                        "cum_proxy":  [], "cum_obs":  []} for m in methods}

    for mname in methods:
        print(f"\n  Method: {mname}   (b = {'0 fixed' if mname == 'exp' else '1 fixed' if mname == 'har' else 'free [0,1]'})")
        hdr = (f"  {'id':>3}  {'b':>5}  "
               f"{'rate err% vs proxy':>20}  {'rate err% vs obs':>18}  "
               f"{'cum gained err% vs proxy':>26}  {'cum gained err% vs obs':>24}")
        print(hdr)
        for id_ in ids:
            mr = results[id_]["methods"].get(mname)
            if mr is None:
                print(f"  {id_:>3}  -- fit failed")
                continue
            b  = mr["params"][2]
            rp = mr["rate_err_proxy"]
            ro = mr["rate_err_obs"]
            cp = mr["cum_err_proxy"]
            co = mr["cum_err_obs"]
            print(f"  {id_:>3}  {b:>5.2f}  "
                  f"{rp:>+19.1f}%  {ro:>+17.1f}%  "
                  f"{cp:>+25.1f}%  {co:>+23.1f}%")
            errors[mname]["rate_proxy"].append(rp)
            errors[mname]["rate_obs"].append(ro)
            errors[mname]["cum_proxy"].append(cp)
            errors[mname]["cum_obs"].append(co)

        v = errors[mname]
        if v["rate_obs"]:
            mean_rp = np.mean(v["rate_proxy"]);  worst_rp = max(v["rate_proxy"], key=abs)
            mean_ro = np.mean(v["rate_obs"]);    worst_ro = max(v["rate_obs"],   key=abs)
            mean_cp = np.mean(v["cum_proxy"]);   worst_cp = max(v["cum_proxy"],  key=abs)
            mean_co = np.mean(v["cum_obs"]);     worst_co = max(v["cum_obs"],    key=abs)
            print(f"  {'mean':>3}  {'':>5}  "
                  f"{mean_rp:>+19.1f}%  {mean_ro:>+17.1f}%  "
                  f"{mean_cp:>+25.1f}%  {mean_co:>+23.1f}%")
            print(f"  {'worst':>3}  {'':>5}  "
                  f"{worst_rp:>+19.1f}%  {worst_ro:>+17.1f}%  "
                  f"{worst_cp:>+25.1f}%  {worst_co:>+23.1f}%")

    return errors


# ---------------------------------------------------------------------------
# Main forecast sensitivity: later fit window starts
# ---------------------------------------------------------------------------

def _run_late_window(win_start_str: str, curves: pd.DataFrame,
                     obs_cum_2008: float) -> tuple[dict, int]:
    """Refit main Arps with a later window start.  Return (results per id per method, n_pts)."""
    win_start = pd.Timestamp(win_start_str)
    win_end   = pd.Timestamp("2008-01-01")
    fc_dates  = C.FORECAST_DATES
    ids       = sorted(curves["id"].unique())

    n_pts_out = 0
    result: dict = {}
    for id_ in ids:
        g        = curves[curves["id"] == id_].set_index("date")
        fit_data = g.loc[win_start:win_end, "oil_rate"]
        n        = len(fit_data)
        if id_ == ids[0]:
            n_pts_out = n
        t0    = win_start
        t_yr  = np.array([(d - t0).days / 365.25 for d in fit_data.index])
        scale = float(fit_data.iloc[0])
        q_sc  = fit_data.to_numpy() / scale

        fits     = _fit_three(t_yr, q_sc, n)
        id_res   = {}
        for mname, fit in fits.items():
            if not fit["success"]:
                id_res[mname] = {"b": np.nan, "cum_2028": np.nan}
                continue
            rates = _project(fit["params"], scale, t0, fc_dates)
            cums  = _integrate(rates, fc_dates, obs_cum_2008, pd.Timestamp("2008-01-01"))
            id_res[mname] = {"b": fit["params"][2], "cum_2028": float(cums[-1])}
        result[id_] = id_res
    return result, n_pts_out


# ---------------------------------------------------------------------------
# Horizon summary table
# ---------------------------------------------------------------------------

def _print_horizon_summary(all_errors: dict) -> None:
    """Print mean |error vs observed| per method per horizon."""
    horizons = sorted(all_errors.keys())
    h_yrs    = {c: round((pd.Timestamp("2008-01-01") - pd.Timestamp(c)).days / 365.25)
                for c in horizons}
    all_methods = []
    for errs in all_errors.values():
        for m in errs:
            if m not in all_methods:
                all_methods.append(m)

    print("\n=== Summary: mean |error vs observed| by cold-test horizon ===")
    print("Even the 4-year horizon is much shorter than the real 20-year forecast.")
    print("Errors here are OPTIMISTIC -- real forecast errors are likely larger.")
    print()
    header = f"  {'method':>6}  "
    for c in horizons:
        header += f"  {h_yrs[c]:>2}yr: rate%  cum%"
    print(header)
    for m in all_methods:
        row = f"  {m:>6}  "
        for c in horizons:
            v = all_errors[c].get(m, {})
            ro = v.get("rate_obs", [])
            co = v.get("cum_obs",  [])
            if ro:
                row += f"  {np.mean(np.abs(ro)):>11.1f}  {np.mean(np.abs(co)):>3.1f}%"
            else:
                row += f"  {'n/a':>11}  {'n/a':>4}"
        print(row)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _plot_cold_test(curves: pd.DataFrame, history: pd.DataFrame,
                    results_2005: dict, id_: int = 1) -> None:
    """Oil rate figure for id_: observed + proxy + three cold-test Arps forecasts from 2005-04-01."""
    cutoff   = pd.Timestamp("2005-01-01")
    fc_dates = pd.date_range("2005-04-01", "2008-01-01", freq="QS")

    g = curves[curves["id"] == id_].set_index("date")
    h = history.set_index("date")
    zoom_start = pd.Timestamp("2003-01-01")

    g_z = g[g.index >= zoom_start]
    h_z = h[h.index >= zoom_start]

    col_m   = {"exp": "tab:blue", "har": "tab:orange", "free": "tab:green"}
    label_m = {"exp": "Exponential (b=0)", "har": "Harmonic (b=1)",
               "free": f"Free b [0,{C.T3_ARPS_B_MAX}]"}

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(g_z.index, g_z["oil_rate"], color="tab:purple", lw=1.6, linestyle="--",
            label="Proxy rate (proxy-estimated)", zorder=4)
    ax.plot(h_z.index, h_z["oil_rate"], color="black", lw=2.2,
            label="Observed rate", zorder=5)
    ax.axvline(cutoff, color="gray", lw=1.2, linestyle=":",
               label="Cold-test cutoff (2005-01-01)")

    res_id = results_2005[id_]
    for mname, mr in res_id["methods"].items():
        if mr is None:
            continue
        ax.plot(fc_dates, mr["rates_fc"],
                color=col_m.get(mname, "gray"), lw=1.6,
                label=f"Cold-test forecast: {label_m.get(mname, mname)}")

    ax.set_title(
        f"T3 cold test id={id_}: oil rate 2003-2008 (proxy-estimated data, three Arps methods)\n"
        "Fit: 2003-01-01 to 2005-01-01.  Forecast: 2005-04-01 to 2008-01-01 (3-year horizon).\n"
        "NOTE: 3-year horizon is OPTIMISTIC -- real forecast is 20 years."
    )
    ax.set_ylabel("Oil rate [STB/d]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = C.FIGURES_DIR / "t3_cold_test_oil.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("Loading data...", flush=True)
    history  = read_history()
    curves   = pd.read_parquet(C.DATA_PROCESSED / "t3_top10_curves.parquet")
    forecast = pd.read_csv(C.OUTPUTS_DIR / "t3_forecast_oil.csv", parse_dates=["date"])

    h_idx        = history.set_index("date")
    obs_cum_2008 = float(h_idx.loc["2008-01-01", "oil_cum"])

    # ---------------------------------------------------------------- Part 1
    print_rate_check(curves, history, forecast)

    # ---------------------------------------------------------------- Part 2: cold tests
    all_cutoff_results: dict = {}
    all_errors: dict = {}

    for cutoff_str, cutoff in _COLD_CUTOFFS.items():
        res = _run_one_cutoff(cutoff, curves, history)
        all_cutoff_results[cutoff_str] = res
        errs = _print_cutoff_table(cutoff_str, res)
        all_errors[cutoff_str] = errs

    # ---------------------------------------------------------------- Late window sensitivity
    print(f"\n=== Main forecast sensitivity: later fit window starts ===")
    print(f"(Reference: main fit window is {C.T3_ARPS_FIT_START} to 2008-01-01, "
          f"b_max={C.T3_ARPS_B_MAX})")
    for win_str in _LATE_WINDOW_STARTS:
        lw, n_pts = _run_late_window(win_str, curves, obs_cum_2008)
        ids        = sorted(lw.keys())
        methods    = list(lw[ids[0]].keys())
        print(f"\n  Window start: {win_str}  (n = {n_pts} pts, "
              f"{'free-b included' if n_pts >= _MIN_PTS_FREE_B else 'free-b SKIPPED'})")
        # header
        hdr = f"  {'id':>3}"
        for m in methods:
            hdr += f"  {'b(' + m + ')':>9}  {'2028 cum(' + m + ')  STB':>22}"
        print(hdr)
        for id_ in ids:
            row = f"  {id_:>3}"
            for m in methods:
                b   = lw[id_][m]["b"]
                c28 = lw[id_][m]["cum_2028"]
                row += f"  {b:>9.3f}  {c28:>25,.0f}"
            print(row)

    # ---------------------------------------------------------------- Horizon summary
    _print_horizon_summary(all_errors)

    # ---------------------------------------------------------------- Figure
    _plot_cold_test(curves, history, all_cutoff_results["2005-01-01"])

    # ---------------------------------------------------------------- Summary
    print("\n=== Summary (proxy-estimated results) ===")
    print("Cold-test horizons: 4 yr (2004), 3 yr (2005), 2 yr (2006).")
    print("All three horizons are MUCH shorter than the real 20-year forecast.")
    print("Short-horizon checks do not establish 20-year accuracy; longer-term errors may be larger.")
    print("Key question: does b stay at the upper bound with later fit windows?")
    print("  (See the late-window sensitivity table above.)")


if __name__ == "__main__":
    main()
