"""T3 Task D: Arps decline curve analysis -- oil forecast 2008-2028 (oil only).

Run: .venv/bin/python -m src.forecast

Fit window: T3_ARPS_FIT_START (2003-01-01) to 2008-01-01.  A 2004-01-01 comparison is also printed.
Both exponential (b=0) and hyperbolic (0 < b <= T3_ARPS_B_MAX) are fitted per id.
Model selection: AICc (penalises the extra parameter of the hyperbolic).
Numerics: time in years, rates scaled by the first rate in the window.
Start point: Option B -- observed oil cumulative at 2008-01-01 (read_history()).

Outputs:
  data/processed/t3_forecast_oil.parquet
  outputs/t3_forecast_oil.csv
  outputs/figures/t3_forecast_oil.png
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

# Two fit windows; PRIMARY is used for saved output and figures
_WIN_PRIMARY = "2003-01"
_WIN_ALT = "2004-01"
_FIT_WINDOWS = {
    _WIN_PRIMARY: (pd.Timestamp(C.T3_ARPS_FIT_START), pd.Timestamp("2008-01-01")),
    _WIN_ALT:     (pd.Timestamp("2004-01-01"),          pd.Timestamp("2008-01-01")),
}

# Optimisation bounds (scaled rates, Di in 1/year)
_BND_EXP = ([0.3, 1e-3], [3.0, 20.0])          # qi_s, Di
_BND_HYP = ([0.3, 1e-3, 0.0], [3.0, 20.0, C.T3_ARPS_B_MAX])  # qi_s, Di, b

# Starting guesses -- multiple to avoid local minima
_P0_EXP = [(q, d) for q, d in iproduct((0.8, 1.0, 1.2), (0.05, 0.15, 0.4, 1.0, 3.0))]
_P0_HYP = [(q, d, b) for q, d, b in iproduct(
    (0.8, 1.0, 1.2), (0.05, 0.15, 0.5, 1.5), (0.0, 0.3, 0.6, 1.0)
)]
_ATOL_BOUND = 0.02   # flag a parameter as "bounds hit" if within 2 % of its bound


# ---------------------------------------------------------------------------
# Arps functions (scaled rates, t in years)
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
    """Exponential: qi_s * exp(-Di*t)."""
    return qi_s * np.exp(-np.asarray(t, float) * Di)


# ---------------------------------------------------------------------------
# AICc
# ---------------------------------------------------------------------------

def _aicc(sse: float, n: int, k: int) -> float:
    if not np.isfinite(sse) or sse <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _fit_model(func, p0_list, bounds, t, q):
    """Try every starting guess; return (best_popt, best_sse) or (None, inf)."""
    best_sse, best_popt = np.inf, None
    for p0 in p0_list:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(func, t, q, p0=p0, bounds=bounds,
                                    maxfev=30000, method="trf")
            res = q - func(t, *popt)
            sse = float(np.dot(res, res))
            if sse < best_sse:
                best_sse, best_popt = sse, popt
        except Exception:
            pass
    return best_popt, best_sse


def _bounds_hit(popt, bounds) -> bool:
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    for p, l, h in zip(popt, lo, hi):
        span = h - l
        if span > 0 and ((p - l) / span < _ATOL_BOUND or (h - p) / span < _ATOL_BOUND):
            return True
    return False


def fit_one(t_yr: np.ndarray, q_sc: np.ndarray, n: int) -> dict:
    """Return {'exp': {...}, 'hyp': {...}} for one id."""
    def pack(popt, sse, bounds, k, fill_b=None):
        if popt is None:
            return {"params": None, "sse": np.inf, "aicc": np.inf,
                    "bounds_hit": False, "success": False}
        p = tuple(popt) if fill_b is None else (popt[0], popt[1], fill_b)
        return {"params": p, "sse": float(sse), "aicc": _aicc(sse, n, k),
                "bounds_hit": _bounds_hit(popt, bounds), "success": True}

    popt_e, sse_e = _fit_model(_exp, _P0_EXP, _BND_EXP, t_yr, q_sc)
    popt_h, sse_h = _fit_model(_hyp, _P0_HYP, _BND_HYP, t_yr, q_sc)

    return {
        "exp": pack(popt_e, sse_e, _BND_EXP, k=2, fill_b=0.0),
        "hyp": pack(popt_h, sse_h, _BND_HYP, k=3),
    }


# ---------------------------------------------------------------------------
# Projection and integration
# ---------------------------------------------------------------------------

def project_rates(params_sc: tuple, scale: float, t0: pd.Timestamp,
                  dates: pd.DatetimeIndex) -> np.ndarray:
    """Evaluate Arps at each date (t relative to t0) and un-scale."""
    qi_s, Di, b = params_sc
    t = np.array([(d - t0).days / 365.25 for d in dates])
    return _hyp(t, qi_s, Di, b) * scale


def integrate_cum(rates: np.ndarray, dates: pd.DatetimeIndex, cum_start: float) -> np.ndarray:
    """Quarter-by-quarter cumulative starting from cum_start at 2008-01-01."""
    prev = pd.Timestamp("2008-01-01")
    cums, cum = [], cum_start
    for d, r in zip(dates, rates):
        cum += r * (d - prev).days
        cums.append(cum)
        prev = d
    return np.array(cums)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    print("Loading curves and history...", flush=True)
    history = read_history()
    curves = pd.read_parquet(C.DATA_PROCESSED / "t3_top10_curves.parquet")
    top10 = pd.read_csv(C.OUTPUTS_DIR / "t3_top10_params.csv")

    h_idx = history.set_index("date")
    obs_cum_2008 = float(h_idx.loc["2008-01-01", "oil_cum"])
    print(f"Observed oil cumulative at 2008-01-01 (Option B start): {obs_cum_2008:,.0f} STB")

    forecast_dates = C.FORECAST_DATES
    ids = sorted(top10["id"].tolist())

    # ------------------------------------------------------------------ fit
    all_fits: dict = {}
    for wname, (wstart, wend) in _FIT_WINDOWS.items():
        wfits: dict = {}
        for id_ in ids:
            g = curves[curves["id"] == id_].set_index("date")
            fit_q = g.loc[wstart:wend, "oil_rate"].to_numpy()
            fit_dates = g.loc[wstart:wend].index
            t0 = wstart
            t_yr = np.array([(d - t0).days / 365.25 for d in fit_dates])
            scale = float(fit_q[0])
            q_sc = fit_q / scale
            wfits[id_] = {
                "fit": fit_one(t_yr, q_sc, len(t_yr)),
                "scale": scale, "t0": t0, "n": len(t_yr),
                "t_yr": t_yr, "q_orig": fit_q,
            }
        all_fits[wname] = wfits

    # AICc winner per id (primary window)
    winners: dict = {}
    for id_ in ids:
        fi = all_fits[_WIN_PRIMARY][id_]["fit"]
        winners[id_] = "hyp" if fi["hyp"]["aicc"] < fi["exp"]["aicc"] else "exp"

    # ------------------------------------------------------------------ forecast (primary, Option B)
    fc: dict = {}
    for id_ in ids:
        fdata = all_fits[_WIN_PRIMARY][id_]
        fc[id_] = {}
        for model in ("exp", "hyp"):
            fi = fdata["fit"][model]
            if not fi["success"]:
                fc[id_][model] = (np.full(len(forecast_dates), np.nan),
                                  np.full(len(forecast_dates), np.nan))
                continue
            rates = project_rates(fi["params"], fdata["scale"], fdata["t0"], forecast_dates)
            cums = integrate_cum(rates, forecast_dates, obs_cum_2008)
            fc[id_][model] = (rates, cums)

    # ------------------------------------------------------------------ Option A (proxy start)
    fc_A: dict = {}
    for id_ in ids:
        fdata = all_fits[_WIN_PRIMARY][id_]
        fi = fdata["fit"][winners[id_]]
        if not fi["success"]:
            fc_A[id_] = np.full(len(forecast_dates), np.nan)
            continue
        g = curves[curves["id"] == id_].set_index("date")
        proxy_start = float(g.loc["2008-01-01", "oil_cum"])
        rates = project_rates(fi["params"], fdata["scale"], fdata["t0"], forecast_dates)
        fc_A[id_] = integrate_cum(rates, forecast_dates, proxy_start)

    # ------------------------------------------------------------------ last proxy rates
    last_proxy = {id_: float(curves[curves["id"] == id_].set_index("date").loc["2008-01-01", "oil_rate"])
                  for id_ in ids}

    # ------------------------------------------------------------------ print table 1: fit quality
    print(f"\n=== Arps fit: window {C.T3_ARPS_FIT_START}–2008-01-01  (proxy-estimated, extrapolated) ===")
    hdr = (f"{'id':>3}  {'qi STB/d':>10}  {'Di 1/yr':>8}  {'b':>6}  "
           f"{'RMSE fit':>9}  {'SSE exp':>9}  {'SSE hyp':>9}  {'AICc win':>8}  "
           f"{'cum 2028 B':>14}  {'rate ratio':>11}  {'bnd?':>5}")
    print(hdr)
    for id_ in ids:
        fd = all_fits[_WIN_PRIMARY][id_]
        fi_e = fd["fit"]["exp"]
        fi_h = fd["fit"]["hyp"]
        w = winners[id_]
        fi_w = fd["fit"][w]
        qi_s, Di, b = fi_w["params"]
        qi_orig = qi_s * fd["scale"]
        rmse = float(np.sqrt(fi_w["sse"] / fd["n"])) * fd["scale"]
        c28 = fc[id_][w][1][-1]
        first_fc = fc[id_][w][0][0]
        ratio = first_fc / last_proxy[id_]
        bnd = "YES" if fi_w["bounds_hit"] else "no"
        print(f"{id_:>3}  {qi_orig:>10.1f}  {Di:>8.4f}  {b:>6.3f}  "
              f"{rmse:>9.1f}  {fi_e['sse']:>9.5f}  {fi_h['sse']:>9.5f}  {w:>8}  "
              f"{c28:>14,.0f}  {ratio:>11.4f}  {bnd:>5}")

    # ------------------------------------------------------------------ print table 2: hyp vs exp 2028 comparison
    n_hyp = sum(1 for id_ in ids if winners[id_] == "hyp")
    print(f"\n=== Hyperbolic vs exponential 2028-01-01 cumulative comparison ===")
    print(f"{'id':>3}  {'b':>6}  {'2028 hyp':>14}  {'2028 exp':>14}  {'diff %':>8}  AICc winner")
    diffs_pct = []
    for id_ in ids:
        c_h = fc[id_]["hyp"][1][-1]
        c_e = fc[id_]["exp"][1][-1]
        if np.isfinite(c_h) and np.isfinite(c_e) and c_e > 0:
            d = (c_h - c_e) / c_e * 100
            diffs_pct.append(abs(d))
        else:
            d = np.nan
        b_hyp = all_fits[_WIN_PRIMARY][id_]["fit"]["hyp"]["params"][2]
        print(f"{id_:>3}  {b_hyp:>6.3f}  {c_h:>14,.0f}  {c_e:>14,.0f}  {d:>+7.1f}%  {winners[id_]}")
    print(f"\nAICc: hyperbolic wins {n_hyp}/{len(ids)}, exponential {len(ids)-n_hyp}/{len(ids)}")
    if diffs_pct:
        print(f"2028 cumulative |hyp-exp|/exp: mean {np.mean(diffs_pct):.1f}%, max {np.max(diffs_pct):.1f}%")

    # ------------------------------------------------------------------ spread of winner forecasts
    w_c28 = [fc[id_][winners[id_]][1][-1] for id_ in ids]
    w_c28 = [c for c in w_c28 if np.isfinite(c)]
    c_min, c_med, c_max = np.min(w_c28), np.median(w_c28), np.max(w_c28)
    spread = (c_max - c_min) / c_med * 100
    print(f"\n2028-01-01 oil cumulative spread (AICc-selected, Option B, proxy-estimated, extrapolated):")
    print(f"  min    : {c_min:>16,.0f} STB")
    print(f"  median : {c_med:>16,.0f} STB")
    print(f"  max    : {c_max:>16,.0f} STB")
    print(f"  (max-min)/median : {spread:.1f}%")

    # ------------------------------------------------------------------ Option A check
    print(f"\n=== Option A vs Option B: start-point difference at 2028-01-01 ===")
    print(f"{'id':>3}  {'Option B (obs start)':>22}  {'Option A (proxy start)':>23}  {'diff M STB':>11}")
    for id_ in ids:
        c_B = fc[id_][winners[id_]][1][-1]
        c_A = fc_A[id_][-1]
        diff_M = (c_A - c_B) / 1e6
        print(f"{id_:>3}  {c_B:>22,.0f}  {c_A:>23,.0f}  {diff_M:>+11.2f}")

    # ------------------------------------------------------------------ window comparison
    print(f"\n=== Window comparison: {C.T3_ARPS_FIT_START} vs 2004-01-01 start ===")
    print(f"{'id':>3}  {'b (2003)':>9}  {'b (2004)':>9}  {'2028 (2003)':>14}  {'2028 (2004)':>14}  {'diff %':>8}")
    for id_ in ids:
        fd03 = all_fits[_WIN_PRIMARY][id_]
        fd04 = all_fits[_WIN_ALT][id_]
        w03 = winners[id_]
        w04 = "hyp" if fd04["fit"]["hyp"]["aicc"] < fd04["fit"]["exp"]["aicc"] else "exp"
        b03 = fd03["fit"][w03]["params"][2]
        b04 = fd04["fit"][w04]["params"][2]
        fi04 = fd04["fit"][w04]
        rates04 = project_rates(fi04["params"], fd04["scale"], fd04["t0"], forecast_dates)
        cums04 = integrate_cum(rates04, forecast_dates, obs_cum_2008)
        c28_03 = fc[id_][w03][1][-1]
        c28_04 = cums04[-1]
        d = (c28_04 - c28_03) / c28_03 * 100
        print(f"{id_:>3}  {b03:>9.3f}  {b04:>9.3f}  {c28_03:>14,.0f}  {c28_04:>14,.0f}  {d:>+8.1f}%")

    # ------------------------------------------------------------------ save
    rows = []
    for id_ in ids:
        w = winners[id_]
        rates, cums = fc[id_][w]
        for d, r, c in zip(forecast_dates, rates, cums):
            rows.append({"id": id_, "date": d, "oil_rate": float(r), "oil_cum": float(c)})
    fc_df = pd.DataFrame(rows)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    fc_df.to_parquet(C.DATA_PROCESSED / "t3_forecast_oil.parquet", index=False)
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fc_df.to_csv(C.OUTPUTS_DIR / "t3_forecast_oil.csv", index=False)
    print(f"\nSaved t3_forecast_oil.parquet and t3_forecast_oil.csv", flush=True)

    # ------------------------------------------------------------------ sanity checks
    print("\n=== Sanity checks ===")
    neg = (fc_df["oil_rate"] < 0).any()
    print(f"Rates never negative      : {'PASS' if not neg else 'FAIL'}")
    dec = any(
        (np.diff(fc_df[fc_df["id"] == id_]["oil_cum"].to_numpy()) < -1.0).any()
        for id_ in ids
    )
    print(f"Cumulative non-decreasing : {'PASS' if not dec else 'FAIL'}")
    inc = any(
        (np.diff(fc_df[fc_df["id"] == id_]["oil_rate"].to_numpy()) > 1.0).any()
        for id_ in ids
    )
    print(f"Rates decline after 2008  : {'PASS' if not inc else 'FAIL'}")

    # ------------------------------------------------------------------ figure
    _plot(curves, history, fc_df, ids, winners)


def _plot(curves: pd.DataFrame, history: pd.DataFrame,
          fc_df: pd.DataFrame, ids: list, winners: dict) -> None:
    colors = plt.cm.tab10.colors
    h = history.set_index("date")
    boundary = pd.Timestamp("2008-01-01")

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(16, 6))
    for i, id_ in enumerate(ids):
        col = colors[i % 10]
        g = curves[curves["id"] == id_].set_index("date")
        fd = fc_df[fc_df["id"] == id_]
        # rates
        ax_r.plot(g.index, g["oil_rate"], color=col, lw=0.9, alpha=0.7)
        ax_r.plot(fd["date"], fd["oil_rate"], color=col, lw=1.4, label=f"id {id_}")
        # cumulative
        dates_all = list(g.index) + list(fd["date"])
        cums_all = list(g["oil_cum"]) + list(fd["oil_cum"])
        ax_c.plot(dates_all, cums_all, color=col, lw=1.0, label=f"id {id_}")

    ax_r.plot(h.index, h["oil_rate"], color="black", lw=2.2, zorder=5, label="Observed")
    ax_c.plot(h.index, h["oil_cum"], color="black", lw=2.2, zorder=5, label="Observed history")

    for ax in (ax_r, ax_c):
        ax.axvline(boundary, color="gray", lw=1.0, linestyle="--", alpha=0.7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    ax_r.set_title("Oil rate [STB/d]")
    ax_c.set_title("Oil cumulative [STB]")
    fig.suptitle(
        "T3 proxy-estimated, extrapolated: oil rate and cumulative 1998–2028\n"
        "(dashed = 2008-01-01 history/forecast boundary; AICc-selected Arps model per id)"
    )
    fig.tight_layout()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = C.FIGURES_DIR / "t3_forecast_oil.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
