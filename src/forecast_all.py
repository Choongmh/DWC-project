"""T3 Task F/G: gas + water forecast 2008-2028, plus calibrated and uncapped scenarios.

Three scenarios per stream:
  'main'            -- harmonic Arps (oil/gas), best-WOR method (water); Option B start.
                       Total liquid capped at max observed rate over last 9 history quarters.
  'rate_calibrated' -- every stream's rate scaled by obs_rate_2008 / proxy_rate_2008.
                       Water: WOR is shifted so starting water rate = observed water rate.
                       Liquid cap also applied.  All values are proxy-estimated, extrapolated.
  'uncapped'        -- same oil as 'main', WOR water WITHOUT liquid cap.
                       For comparison only -- do NOT use in the template.

Units
  Oil, water  : rates in STB/d, cumulative in STB.
  Gas         : rates in MSCF/d, cumulative in MSCF.

Run: .venv/bin/python -m src.forecast_all
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
_FIT_START    = pd.Timestamp(C.T3_ARPS_FIT_START)   # 2003-01-01
_FIT_END      = pd.Timestamp("2008-01-01")
_FC_DATES     = C.FORECAST_DATES                      # 2008-04-01 ... 2028-01-01
_COLD_CUTOFF  = pd.Timestamp("2005-01-01")
_WATER_T0     = _FIT_START                            # reference date for time-based water models
_FW_MAX       = 0.99                                  # water cut hard cap
_WOR_EPS      = 1e-6                                  # guard log(0) in WOR fits

# Harmonic Arps bounds / starting guesses (b=1 fixed, fit qi_s and Di only)
_BND_HAR = ([0.3, 1e-3], [3.0, 20.0])
_P0_HAR  = [(q, d) for q, d in iproduct((0.8, 1.0, 1.2), (0.05, 0.15, 0.5, 1.5))]

# ---------------------------------------------------------------------------
# Harmonic Arps helpers
# ---------------------------------------------------------------------------

def _har(t: np.ndarray, qi_s: float, Di: float) -> np.ndarray:
    """Harmonic decline: qi_s / (1 + Di*t), t in years."""
    return np.clip(qi_s / np.maximum(1.0 + Di * np.asarray(t, float), 1e-300), 0.0, None)


def _fit_harmonic(rate_series: pd.Series) -> tuple[float, float, float, pd.Timestamp]:
    """Fit harmonic Arps.  Returns (qi_s, Di, scale, t0).

    scale: rate at first fit date (un-scales qi_s back to original units).
    t0: first fit date (time origin).
    """
    t0    = rate_series.index[0]
    t_yr  = np.array([(d - t0).days / 365.25 for d in rate_series.index])
    scale = float(rate_series.iloc[0])
    q_sc  = rate_series.to_numpy() / scale

    best_sse, best_p = np.inf, None
    for p0 in _P0_HAR:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(_har, t_yr, q_sc, p0=p0, bounds=_BND_HAR,
                                    maxfev=30_000, method="trf")
            res = q_sc - _har(t_yr, *popt)
            sse = float(np.dot(res, res))
            if sse < best_sse:
                best_sse, best_p = sse, popt
        except Exception:
            pass
    if best_p is None:
        raise RuntimeError(f"Harmonic Arps fit failed (t0={t0})")
    return float(best_p[0]), float(best_p[1]), scale, t0


def _project_har(qi_s: float, Di: float, scale: float, t0: pd.Timestamp,
                 dates: pd.DatetimeIndex) -> np.ndarray:
    """Evaluate harmonic Arps at given dates and un-scale to original units."""
    t = np.array([(d - t0).days / 365.25 for d in dates])
    return _har(t, qi_s, Di) * scale


def _integrate(rates: np.ndarray, dates: pd.DatetimeIndex,
               cum_start: float, start_date: pd.Timestamp) -> np.ndarray:
    """Quarter-by-quarter cumulative starting from cum_start at start_date."""
    prev, cum, cums = start_date, cum_start, []
    for d, r in zip(dates, rates):
        cum += r * (d - prev).days
        cums.append(cum)
        prev = d
    return np.array(cums)


# ---------------------------------------------------------------------------
# Water WOR / fw methods
# ---------------------------------------------------------------------------

def _wor(water_rates: np.ndarray, oil_rates: np.ndarray) -> np.ndarray:
    return np.maximum(water_rates, 0.0) / np.maximum(oil_rates, 1e-9)


def _fw_from_wor(wor: np.ndarray) -> np.ndarray:
    return wor / (1.0 + np.maximum(wor, 0.0))


def _cap_wor_by_fw(wor: np.ndarray) -> np.ndarray:
    """Cap WOR so that fw = wor/(1+wor) <= _FW_MAX."""
    fw_cap_wor = _FW_MAX / (1.0 - _FW_MAX)   # WOR corresponding to fw = 0.99
    return np.minimum(wor, fw_cap_wor)


def _fit_water(method: str, wor_arr: np.ndarray, dates: pd.DatetimeIndex,
               cum_oil_arr: np.ndarray) -> dict:
    """Fit a water model.

    method 'wor_time'  : log(WOR) = a + b*t  (t in years from _WATER_T0)
    method 'wor_cum'   : log(WOR) = a + b*cum_oil  (cum_oil in STB)
    method 'logistic'  : logit(fw) = a + b*t

    Returns params dict.
    """
    log_wor = np.log(np.maximum(wor_arr, _WOR_EPS))
    t_yr    = np.array([(d - _WATER_T0).days / 365.25 for d in dates])

    if method == "wor_time":
        b, a = np.polyfit(t_yr, log_wor, 1)
        return {"a": float(a), "b": float(b), "method": method}

    elif method == "wor_cum":
        b, a = np.polyfit(cum_oil_arr, log_wor, 1)
        return {"a": float(a), "b": float(b), "method": method}

    elif method == "logistic":
        fw        = _fw_from_wor(wor_arr)
        fw_clip   = np.clip(fw, _WOR_EPS, 1.0 - _WOR_EPS)
        logit_fw  = np.log(fw_clip / (1.0 - fw_clip))
        b, a      = np.polyfit(t_yr, logit_fw, 1)
        return {"a": float(a), "b": float(b), "method": method}

    else:
        raise ValueError(f"Unknown water method: {method}")


def _forecast_water(method: str, params: dict, oil_rates_fc: np.ndarray,
                    oil_cums_fc: np.ndarray, fc_dates: pd.DatetimeIndex,
                    wor_scale: float = 1.0) -> np.ndarray:
    """Forecast water rates (STB/d).

    wor_scale: multiplicative shift applied to WOR after forecasting
               (used for rate_calibrated scenario to match observed starting water rate).

    Returns water rates with fw capped at _FW_MAX.
    """
    a, b  = params["a"], params["b"]
    t_yr  = np.array([(d - _WATER_T0).days / 365.25 for d in fc_dates])

    if method == "wor_time":
        wor = np.exp(a + b * t_yr)
    elif method == "wor_cum":
        wor = np.exp(a + b * oil_cums_fc)
    elif method == "logistic":
        fw  = 1.0 / (1.0 + np.exp(-(a + b * t_yr)))
        wor = fw / np.maximum(1.0 - fw, 1e-9)
    else:
        raise ValueError(f"Unknown water method: {method}")

    wor   = _cap_wor_by_fw(wor * wor_scale)
    return np.maximum(oil_rates_fc * wor, 0.0)


# ---------------------------------------------------------------------------
# Water cold test (cutoff 2005-01-01, use proxy oil rates to isolate water error)
# ---------------------------------------------------------------------------

def _cold_test_water(curves: pd.DataFrame, history: pd.DataFrame) -> tuple[str, dict]:
    """Fit three water methods on proxy data up to _COLD_CUTOFF; forecast to 2008-01-01.

    Uses proxy oil rates (not Arps forecast oil) to isolate the water-model error.
    This is an OPTIMISTIC 3-year test -- the real forecast is 20 years.
    Returns (best_method, results_dict).
    """
    cutoff   = _COLD_CUTOFF
    fc_dates = pd.date_range(cutoff + pd.DateOffset(months=3), "2008-01-01", freq="QS")
    ids      = sorted(curves["id"].unique())
    h        = history.set_index("date")

    obs_water_rate_end  = float(h.loc["2008-01-01", "water_rate"])
    obs_water_at_cutoff = float(h.loc[cutoff,        "water_cum"])
    obs_water_end       = float(h.loc["2008-01-01",  "water_cum"])
    obs_water_gained    = obs_water_end - obs_water_at_cutoff

    methods     = ("wor_time", "wor_cum", "logistic")
    all_results = {}

    for id_ in ids:
        g = curves[curves["id"] == id_].set_index("date")

        # Fit-window proxy data
        fit_w   = g.loc[_FIT_START:cutoff, "water_rate"].to_numpy()
        fit_o   = g.loc[_FIT_START:cutoff, "oil_rate"].to_numpy()
        fit_co  = g.loc[_FIT_START:cutoff, "oil_cum"].to_numpy()
        fit_idx = g.loc[_FIT_START:cutoff].index
        wor_fit = _wor(fit_w, fit_o)

        # Proxy values for forecast dates (use proxy oil to isolate water error)
        fc_o_rates = g.loc[fc_dates, "oil_rate"].to_numpy()
        fc_o_cums  = g.loc[fc_dates, "oil_cum"].to_numpy()

        proxy_water_at_cutoff = float(g.loc[cutoff,       "water_cum"])
        proxy_water_end       = float(g.loc["2008-01-01", "water_cum"])
        proxy_water_rate_end  = float(g.loc["2008-01-01", "water_rate"])
        proxy_water_gained    = proxy_water_end - proxy_water_at_cutoff

        id_res = {
            "proxy_water_rate_end":  proxy_water_rate_end,
            "obs_water_rate_end":    obs_water_rate_end,
            "proxy_water_gained":    proxy_water_gained,
            "obs_water_gained":      obs_water_gained,
            "proxy_water_at_cutoff": proxy_water_at_cutoff,
            "obs_water_at_cutoff":   obs_water_at_cutoff,
            "methods": {},
        }

        for m in methods:
            try:
                params     = _fit_water(m, wor_fit, fit_idx, fit_co)
                w_fc       = _forecast_water(m, params, fc_o_rates, fc_o_cums, fc_dates)
                cums_p     = _integrate(w_fc, fc_dates, proxy_water_at_cutoff, cutoff)
                cums_o     = _integrate(w_fc, fc_dates, obs_water_at_cutoff,   cutoff)
                fc_rate_end= float(w_fc[-1])
                gained_p   = float(cums_p[-1]) - proxy_water_at_cutoff
                gained_o   = float(cums_o[-1]) - obs_water_at_cutoff

                denom_rate_p = max(proxy_water_rate_end, 1.0)
                denom_rate_o = max(obs_water_rate_end,   1.0)
                denom_cum_p  = max(proxy_water_gained,   1.0)
                denom_cum_o  = max(obs_water_gained,     1.0)

                id_res["methods"][m] = {
                    "rate_err_proxy": (fc_rate_end - proxy_water_rate_end) / denom_rate_p * 100,
                    "rate_err_obs":   (fc_rate_end - obs_water_rate_end)   / denom_rate_o * 100,
                    "cum_err_proxy":  (gained_p - proxy_water_gained)      / denom_cum_p  * 100,
                    "cum_err_obs":    (gained_o - obs_water_gained)        / denom_cum_o  * 100,
                    "w_fc":           w_fc,
                }
            except Exception as e:
                id_res["methods"][m] = None

        all_results[id_] = id_res

    # Print cold-test table
    print(f"\n=== Water cold test: fit 2003-01-01 to 2005-01-01, "
          f"forecast to 2008-01-01 (3-yr horizon) ===")
    print("NOTE: 3-year cold test is OPTIMISTIC -- real forecast is 20 years.")
    print("      Proxy oil rates are used to isolate the water-model error.")

    method_abs_cum_proxy: dict = {m: [] for m in methods}

    for m in methods:
        m_label = {"wor_time": "wor_time (log(WOR) vs t)",
                   "wor_cum":  "wor_cum  (log(WOR) vs cum_oil)",
                   "logistic": "logistic (logit(fw) vs t)"}[m]
        print(f"\n  Method: {m_label}")
        print(f"  {'id':>3}  {'rate%vsProxy':>14}  {'rate%vsObs':>12}  "
              f"{'cum%vsProxy':>13}  {'cum%vsObs':>11}")
        for id_ in sorted(all_results.keys()):
            mr = all_results[id_]["methods"].get(m)
            if mr is None:
                print(f"  {id_:>3}  fit failed"); continue
            print(f"  {id_:>3}  {mr['rate_err_proxy']:>+13.1f}%  {mr['rate_err_obs']:>+11.1f}%"
                  f"  {mr['cum_err_proxy']:>+12.1f}%  {mr['cum_err_obs']:>+10.1f}%")
            method_abs_cum_proxy[m].append(abs(mr["cum_err_proxy"]))

        errs_p = [all_results[id_]["methods"][m]["cum_err_proxy"]
                  for id_ in sorted(all_results) if all_results[id_]["methods"].get(m)]
        errs_o = [all_results[id_]["methods"][m]["cum_err_obs"]
                  for id_ in sorted(all_results) if all_results[id_]["methods"].get(m)]
        if errs_p:
            print(f"  mean   {np.mean(errs_p):>+13.1f}%  {np.mean(errs_o):>+11.1f}%  "
                  f"(cum vs proxy / obs)")
            worst_p = max(errs_p, key=abs); worst_o = max(errs_o, key=abs)
            print(f"  worst  {worst_p:>+13.1f}%  {worst_o:>+11.1f}%")

    # Choose method: smallest mean |cum_err vs proxy|
    best_m  = min(methods, key=lambda m: np.mean(method_abs_cum_proxy[m])
                  if method_abs_cum_proxy[m] else np.inf)
    best_mean = np.mean(method_abs_cum_proxy[best_m])
    print(f"\n  --> Chosen water method: {best_m}  "
          f"(mean |cum err vs proxy| = {best_mean:.1f}%)")
    print("      This 3-year cold test is optimistic; longer-term errors may be larger.")

    return best_m, all_results


# ---------------------------------------------------------------------------
# Build forecast for all ids, both scenarios
# ---------------------------------------------------------------------------

def _build_forecast(curves: pd.DataFrame, history: pd.DataFrame,
                    water_method: str) -> pd.DataFrame:
    """Fit harmonic Arps (oil, gas) and water model; return long-format forecast table."""
    h       = history.set_index("date")
    ids     = sorted(curves["id"].unique())

    # Observed cumulatives at 2008-01-01 (Option B start)
    obs_oil_cum_2008   = float(h.loc["2008-01-01", "oil_cum"])
    obs_gas_cum_2008   = float(h.loc["2008-01-01", "gas_cum"])
    obs_water_cum_2008 = float(h.loc["2008-01-01", "water_cum"])

    # Observed rates at 2008-01-01 (for rate_calibrated scale factors)
    obs_oil_rate_2008   = float(h.loc["2008-01-01", "oil_rate"])
    obs_gas_rate_2008   = float(h.loc["2008-01-01", "gas_rate"])
    obs_water_rate_2008 = float(h.loc["2008-01-01", "water_rate"])

    # Total liquid rate cap: max observed total liquid rate over last 9 quarters of history.
    # Modelling assumption: forecast total liquid (oil+water) is capped at this level.
    # Water rate = min(WOR * oil_rate,  cap - oil_rate).
    hist_window = h.loc["2006-01-01":"2008-01-01"]
    liq_cap = float((hist_window["oil_rate"] + hist_window["water_rate"]).max())
    print(f"\n  Total liquid rate cap (max observed 2006-01-01 to 2008-01-01): {liq_cap:.1f} STB/d")
    print("  Modelling assumption: forecast total liquid is capped at this level.")
    print("  Scenario 'uncapped' is kept for comparison; it does NOT go into the template.")

    print("\n=== Proxy vs observed rate at 2008-01-01 (scale factors for rate_calibrated) ===")
    print("  Stream: oil [STB/d], gas [MSCF/d], water [STB/d]")
    print(f"  {'id':>3}  {'obs oil':>10}  {'proxy oil':>11}  {'oil sf':>8}"
          f"  {'obs gas':>10}  {'proxy gas':>11}  {'gas sf':>8}"
          f"  {'obs water':>11}  {'proxy water':>13}  {'water sf':>9}")

    rows = []

    for id_ in ids:
        g = curves[curves["id"] == id_].set_index("date")

        # Scale factors (own-stream calibration)
        proxy_oil_rate_2008   = float(g.loc["2008-01-01", "oil_rate"])
        proxy_gas_rate_2008   = float(g.loc["2008-01-01", "gas_rate"])
        proxy_water_rate_2008 = float(g.loc["2008-01-01", "water_rate"])
        oil_sf   = obs_oil_rate_2008   / proxy_oil_rate_2008
        gas_sf   = obs_gas_rate_2008   / proxy_gas_rate_2008
        water_sf = obs_water_rate_2008 / proxy_water_rate_2008

        print(f"  {id_:>3}  {obs_oil_rate_2008:>10.1f}  {proxy_oil_rate_2008:>11.1f}  {oil_sf:>8.4f}"
              f"  {obs_gas_rate_2008:>10.2f}  {proxy_gas_rate_2008:>11.2f}  {gas_sf:>8.4f}"
              f"  {obs_water_rate_2008:>11.1f}  {proxy_water_rate_2008:>13.1f}  {water_sf:>9.4f}")

        # Fit harmonic Arps: oil
        oil_qi, oil_Di, oil_sc, oil_t0 = _fit_harmonic(
            g.loc[_FIT_START:_FIT_END, "oil_rate"])

        # Fit harmonic Arps: gas (MSCF/d)
        gas_qi, gas_Di, gas_sc, gas_t0 = _fit_harmonic(
            g.loc[_FIT_START:_FIT_END, "gas_rate"])

        # Fit water model on full history window (2003-01-01 to 2008-01-01)
        fit_w    = g.loc[_FIT_START:_FIT_END, "water_rate"].to_numpy()
        fit_o    = g.loc[_FIT_START:_FIT_END, "oil_rate"].to_numpy()
        fit_co   = g.loc[_FIT_START:_FIT_END, "oil_cum"].to_numpy()
        fit_idx  = g.loc[_FIT_START:_FIT_END].index
        wor_fit  = _wor(fit_w, fit_o)
        water_params = _fit_water(water_method, wor_fit, fit_idx, fit_co)

        # Main forecast rates
        oil_rates_main    = _project_har(oil_qi, oil_Di, oil_sc, oil_t0, _FC_DATES)
        gas_rates_main    = _project_har(gas_qi, gas_Di, gas_sc, gas_t0, _FC_DATES)
        oil_cums_main     = _integrate(oil_rates_main, _FC_DATES, obs_oil_cum_2008, _FIT_END)

        # WOR-model water before any liquid cap (used for 'uncapped' scenario and as base for cal)
        water_rates_uncap = _forecast_water(water_method, water_params,
                                            oil_rates_main, oil_cums_main, _FC_DATES)

        # Apply liquid cap: water = min(WOR_water, liq_cap - oil), keeping total <= liq_cap.
        water_rates_main  = np.minimum(water_rates_uncap,
                                       np.maximum(liq_cap - oil_rates_main, 0.0))

        gas_cums_main      = _integrate(gas_rates_main,    _FC_DATES, obs_gas_cum_2008,   _FIT_END)
        water_cums_main    = _integrate(water_rates_main,  _FC_DATES, obs_water_cum_2008, _FIT_END)
        water_cums_uncap   = _integrate(water_rates_uncap, _FC_DATES, obs_water_cum_2008, _FIT_END)

        # Rate-calibrated: scale from uncapped WOR water (preserves original wor_scale logic)
        oil_rates_cal   = oil_rates_main    * oil_sf
        gas_rates_cal   = gas_rates_main    * gas_sf
        water_rates_cal = water_rates_uncap * water_sf

        # Enforce fw cap on water_calibrated
        total_liq_cal = oil_rates_cal + water_rates_cal
        fw_cal        = np.where(total_liq_cal > 0, water_rates_cal / total_liq_cal, 0.0)
        exceed        = fw_cal > _FW_MAX
        if exceed.any():
            water_rates_cal = np.where(exceed,
                                       oil_rates_cal * _FW_MAX / (1.0 - _FW_MAX),
                                       water_rates_cal)

        # Apply liquid cap to rate_calibrated as well
        water_rates_cal = np.minimum(water_rates_cal,
                                     np.maximum(liq_cap - oil_rates_cal, 0.0))

        oil_cums_cal   = _integrate(oil_rates_cal,   _FC_DATES, obs_oil_cum_2008,   _FIT_END)
        gas_cums_cal   = _integrate(gas_rates_cal,   _FC_DATES, obs_gas_cum_2008,   _FIT_END)
        water_cums_cal = _integrate(water_rates_cal, _FC_DATES, obs_water_cum_2008, _FIT_END)

        # Assemble rows (starting row at 2008-01-01 + 80 forecast dates)
        # 'uncapped' uses same oil as 'main' but WOR water WITHOUT liquid cap (comparison only).
        for scenario, oil_r, gas_r, wat_r, oil_c, gas_c, wat_c in [
            ("main",
             np.concatenate([[proxy_oil_rate_2008],   oil_rates_main]),
             np.concatenate([[proxy_gas_rate_2008],   gas_rates_main]),
             np.concatenate([[proxy_water_rate_2008], water_rates_main]),
             np.concatenate([[obs_oil_cum_2008],   oil_cums_main]),
             np.concatenate([[obs_gas_cum_2008],   gas_cums_main]),
             np.concatenate([[obs_water_cum_2008], water_cums_main])),
            ("rate_calibrated",
             np.concatenate([[proxy_oil_rate_2008   * oil_sf],   oil_rates_cal]),
             np.concatenate([[proxy_gas_rate_2008   * gas_sf],   gas_rates_cal]),
             np.concatenate([[proxy_water_rate_2008 * water_sf], water_rates_cal]),
             np.concatenate([[obs_oil_cum_2008],   oil_cums_cal]),
             np.concatenate([[obs_gas_cum_2008],   gas_cums_cal]),
             np.concatenate([[obs_water_cum_2008], water_cums_cal])),
            ("uncapped",
             np.concatenate([[proxy_oil_rate_2008],   oil_rates_main]),
             np.concatenate([[proxy_gas_rate_2008],   gas_rates_main]),
             np.concatenate([[proxy_water_rate_2008], water_rates_uncap]),
             np.concatenate([[obs_oil_cum_2008],   oil_cums_main]),
             np.concatenate([[obs_gas_cum_2008],   gas_cums_main]),
             np.concatenate([[obs_water_cum_2008], water_cums_uncap])),
        ]:
            all_dates = pd.DatetimeIndex([_FIT_END]).append(_FC_DATES)
            for date, or_, gr_, wr_, oc_, gc_, wc_ in zip(
                all_dates, oil_r, gas_r, wat_r, oil_c, gas_c, wat_c
            ):
                rows.append({
                    "id": id_, "scenario": scenario, "date": date,
                    "oil_rate": float(or_),   "gas_rate": float(gr_),
                    "water_rate": float(wr_),
                    "oil_cum": float(oc_),    "gas_cum": float(gc_),
                    "water_cum": float(wc_),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_gas_params(curves: pd.DataFrame, history: pd.DataFrame,
                      fc_df: pd.DataFrame) -> None:
    """Print gas harmonic Arps params and GOR sanity ratios."""
    h   = history.set_index("date")
    ids = sorted(curves["id"].unique())

    print(f"\n=== Gas harmonic Arps fit parameters "
          f"(fit {C.T3_ARPS_FIT_START} to 2008-01-01, proxy-estimated) ===")
    print("  Gas rates in MSCF/d, gas cumulative in MSCF.")
    print(f"  {'id':>3}  {'qi_s':>6}  {'qi [MSCF/d]':>13}  {'Di [1/yr]':>10}  "
          f"{'2028 gas cum [MSCF]':>22}  {'GOR@2008-04 [MSCF/STB]':>24}  "
          f"{'GOR@2028-01 [MSCF/STB]':>24}")

    main_rows = fc_df[fc_df["scenario"] == "main"]
    for id_ in ids:
        g   = curves[curves["id"] == id_].set_index("date")
        qi_s, Di, sc, t0 = _fit_harmonic(g.loc[_FIT_START:_FIT_END, "gas_rate"])
        qi_abs = qi_s * sc   # MSCF/d

        id_rows = main_rows[main_rows["id"] == id_].set_index("date")
        c28     = float(id_rows.loc["2028-01-01", "gas_cum"])

        gor_start = float(id_rows.loc["2008-04-01", "gas_rate"]) / \
                    float(id_rows.loc["2008-04-01", "oil_rate"])
        gor_end   = float(id_rows.loc["2028-01-01", "gas_rate"]) / \
                    float(id_rows.loc["2028-01-01", "oil_rate"])

        print(f"  {id_:>3}  {qi_s:>6.3f}  {qi_abs:>13.2f}  {Di:>10.4f}  "
              f"{c28:>22,.0f}  {gor_start:>24.4f}  {gor_end:>24.4f}")
    print("  GOR stable <=> gas and oil have similar decline rates (Di_gas ~ Di_oil).")


def _print_2028_summary(fc_df: pd.DataFrame) -> None:
    """Print 2028-01-01 cumulative for all ids, all scenarios, plus water spread stats."""
    end_date = pd.Timestamp("2028-01-01")
    end_rows = fc_df[fc_df["date"] == end_date]

    print(f"\n=== 2028-01-01 cumulative (proxy-estimated, extrapolated) ===")
    print("  Oil and water in STB; gas in MSCF.")
    print("  NOTE: 'uncapped' is for comparison only -- do NOT use in the template.")
    print(f"  {'id':>3}  {'scenario':>18}  {'oil_cum STB':>20}  "
          f"{'gas_cum MSCF':>20}  {'water_cum STB':>20}")

    for scenario in ("main", "rate_calibrated", "uncapped"):
        sc_rows = end_rows[end_rows["scenario"] == scenario].sort_values("id")
        for _, r in sc_rows.iterrows():
            print(f"  {int(r['id']):>3}  {scenario:>18}  {r['oil_cum']:>20,.0f}  "
                  f"{r['gas_cum']:>20,.0f}  {r['water_cum']:>20,.0f}")

        print(f"\n  {'':>3}  {scenario + ' stats':>18}:")
        for col, unit in [("oil_cum", "STB"), ("gas_cum", "MSCF"), ("water_cum", "STB")]:
            vals = sc_rows[col].to_numpy()
            print(f"  {'':>3}  {'':>18}  {col:>12} [{unit}]  "
                  f"min={vals.min():>14,.0f}  "
                  f"median={np.median(vals):>14,.0f}  "
                  f"max={vals.max():>14,.0f}")
        print()

    # Dedicated water cum comparison across scenarios
    print("=== 2028-01-01 water cumulative [STB]: min / median / max by scenario ===")
    for scenario in ("main", "rate_calibrated", "uncapped"):
        vals = end_rows[end_rows["scenario"] == scenario]["water_cum"].to_numpy()
        tag  = "  <-- NOT for template" if scenario == "uncapped" else ""
        print(f"  {scenario:>18}:  min={vals.min():>14,.0f}  "
              f"median={np.median(vals):>14,.0f}  max={vals.max():>14,.0f}{tag}")


# ---------------------------------------------------------------------------
# Liquid cap verification table
# ---------------------------------------------------------------------------

def _print_liquid_check(fc_df: pd.DataFrame) -> None:
    """Print oil/water/total-liquid/fw at 5 dates for ids 1, 5, 9 (main scenario).

    Used to verify the liquid cap is working correctly.
    """
    check_ids   = [1, 5, 9]
    check_dates = pd.to_datetime(
        ["2008-01-01", "2013-01-01", "2018-01-01", "2023-01-01", "2028-01-01"])

    print("\n=== Liquid cap verification: main scenario, ids 1 / 5 / 9 ===")
    print("  Units: rates STB/d, fw dimensionless.")
    print(f"  {'id':>3}  {'date':>12}  {'oil_rate':>10}  {'water_rate':>11}  "
          f"{'total_liq':>10}  {'fw':>6}")

    main = fc_df[fc_df["scenario"] == "main"]
    for id_ in check_ids:
        g = main[main["id"] == id_].set_index("date")
        for d in check_dates:
            if d not in g.index:
                continue
            oil_r   = float(g.loc[d, "oil_rate"])
            wat_r   = float(g.loc[d, "water_rate"])
            tot_liq = oil_r + wat_r
            fw      = wat_r / tot_liq if tot_liq > 0 else 0.0
            print(f"  {id_:>3}  {str(d.date()):>12}  {oil_r:>10.1f}  {wat_r:>11.1f}  "
                  f"{tot_liq:>10.1f}  {fw:>6.3f}")
        print()


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _sanity_checks(fc_df: pd.DataFrame) -> None:
    """PASS/FAIL checks on the full forecast table."""
    print("\n=== Sanity checks ===")
    ids = sorted(fc_df["id"].unique())

    # 1. Rates never negative
    neg = (fc_df[["oil_rate", "gas_rate", "water_rate"]] < -1e-6).any().any()
    print(f"  Rates never negative       : {'PASS' if not neg else 'FAIL'}")

    # 2. Cumulative non-decreasing (skip starting row at 2008-01-01)
    fc_only = fc_df[fc_df["date"] > pd.Timestamp("2008-01-01")]
    dec = False
    for (id_, sc), grp in fc_only.groupby(["id", "scenario"]):
        for col in ("oil_cum", "gas_cum", "water_cum"):
            if (np.diff(grp.sort_values("date")[col].to_numpy()) < -1.0).any():
                dec = True
    print(f"  Cumulative non-decreasing  : {'PASS' if not dec else 'FAIL'}")

    # 3. Water cut fw <= FW_MAX in all rows
    total_liq = fc_df["oil_rate"] + fc_df["water_rate"]
    fw_vals   = np.where(total_liq > 0, fc_df["water_rate"] / total_liq, 0.0)
    fw_ok     = (fw_vals <= _FW_MAX + 1e-6).all()
    max_fw    = float(fw_vals.max())
    print(f"  Water cut fw <= {_FW_MAX:.2f}       : {'PASS' if fw_ok else 'FAIL'}"
          f"  (max fw = {max_fw:.4f})")

    # 4. No cumulative jump at 2008-01-01 boundary
    start_rows = fc_df[fc_df["date"] == pd.Timestamp("2008-01-01")]
    next_rows  = fc_df[fc_df["date"] == pd.Timestamp("2008-04-01")]
    jump_ok    = True
    for (id_, sc) in start_rows.groupby(["id", "scenario"]).groups:
        s0 = start_rows[(start_rows["id"] == id_) & (start_rows["scenario"] == sc)]
        s1 = next_rows[ (next_rows["id"]  == id_) & (next_rows["scenario"]  == sc)]
        if s0.empty or s1.empty:
            continue
        days = 91   # 2008-01-01 to 2008-04-01
        for col_c, col_r in [("oil_cum", "oil_rate"), ("gas_cum", "gas_rate"),
                              ("water_cum", "water_rate")]:
            expected = float(s0[col_c].iloc[0]) + float(s1[col_r].iloc[0]) * days
            actual   = float(s1[col_c].iloc[0])
            if abs(actual - expected) > 2.0:
                jump_ok = False
    print(f"  No cum jump at 2008-01-01  : {'PASS' if jump_ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _plot(fc_df: pd.DataFrame, history: pd.DataFrame) -> None:
    """Three-panel figure: oil, gas, water cumulative (main scenario, 10 ids + history)."""
    h      = history.set_index("date")
    ids    = sorted(fc_df["id"].unique())
    colors = plt.cm.tab10.colors
    main   = fc_df[fc_df["scenario"] == "main"]

    stream_info = [
        ("oil_cum",   "Oil cumulative [STB]"),
        ("gas_cum",   "Gas cumulative [MSCF]"),
        ("water_cum", "Water cumulative [STB]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (col, label) in zip(axes, stream_info):
        for i, id_ in enumerate(ids):
            g = main[main["id"] == id_].sort_values("date")
            ax.plot(g["date"], g[col], color=colors[i % 10], lw=1.0,
                    alpha=0.85, label=f"id {id_}")
        ax.plot(h.index, h[col], color="black", lw=2.2, zorder=5, label="Observed history")
        ax.axvline(pd.Timestamp("2008-01-01"), color="gray", lw=1.0, linestyle="--", alpha=0.7)
        ax.set_title(label)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "T3 proxy-estimated, extrapolated: oil / gas / water cumulative 1998-2028\n"
        "('main' scenario, harmonic Arps oil+gas, WOR method water; "
        "dashed = 2008-01-01 boundary)"
    )
    fig.tight_layout()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = C.FIGURES_DIR / "t3_forecast_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("Loading data...", flush=True)
    history = read_history()
    curves  = pd.read_parquet(C.DATA_PROCESSED / "t3_top10_curves.parquet")

    # Step 1: water cold test to choose best method
    best_water, cold_test_res = _cold_test_water(curves, history)

    # Step 2: build full forecast (also prints scale-factor table)
    print("\nBuilding forecast for all streams, both scenarios...", flush=True)
    fc_df = _build_forecast(curves, history, best_water)

    # Step 3: liquid cap verification table (ids 1, 5, 9)
    _print_liquid_check(fc_df)

    # Step 4: print gas parameters and GOR ratios
    _print_gas_params(curves, history, fc_df)

    # Step 5: print 2028 summary (all 3 scenarios incl. uncapped)
    _print_2028_summary(fc_df)

    # Step 6: save (3 scenarios x 10 ids x 81 dates = 2430 rows)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    fc_df.to_parquet(C.DATA_PROCESSED / "t3_forecast_all.parquet", index=False)
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fc_df.to_csv(C.OUTPUTS_DIR / "t3_forecast_all.csv", index=False)
    print(f"\nSaved t3_forecast_all.parquet ({len(fc_df):,} rows) and t3_forecast_all.csv",
          flush=True)

    # Step 7: sanity checks (all scenarios)
    _sanity_checks(fc_df)

    # Step 7: figure
    _plot(fc_df, history)


if __name__ == "__main__":
    main()
