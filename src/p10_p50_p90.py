"""T3 Task G Part 3: P10 / P50 / P90 fan statistics + scenario range comparison.

Convention (industry standard, high-case = P10):
  P90 = 10th percentile  -- low / conservative estimate
  P50 = 50th percentile  -- median
  P10 = 90th percentile  -- high / optimistic estimate

Two distinct uncertainty sources shown:
  (a) Parameter uncertainty: P10-P90 spread across 10 diverse parameter sets
      (all using the same 'main' method assumptions).
  (b) Method / assumption uncertainty: min-max of the median across scenarios
      (main, rate_calibrated; water also includes uncapped).
      Labelled "method / assumption scenarios -- NOT a probability range".

Outputs:
  outputs/t3_p10_p50_p90.csv              -- long-format with P90/P50/P10 + scenario bounds
  outputs/figures/t3_fan_chart.png        -- 3-panel fan chart showing both bands
  outputs/figures/t3_uncertainty_compare.png -- % deviation from main P50 for both bands

Run: .venv/bin/python -m src.p10_p50_p90
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.load_data import read_history

_STREAMS = [
    ("oil_cum",   "Oil cumulative [STB]"),
    ("gas_cum",   "Gas cumulative [MSCF]"),
    ("water_cum", "Water cumulative [STB]"),
]

# Scenarios included in the scenario range per stream.
# Water gets 'uncapped' too so the full modelling assumption spread is shown.
_SCENARIO_STREAMS: dict[str, list[str]] = {
    "oil_cum":   ["main", "rate_calibrated"],
    "gas_cum":   ["main", "rate_calibrated"],
    "water_cum": ["main", "rate_calibrated", "uncapped"],
}


def _compute_fan(fc: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Compute P10 / P50 / P90 across ids at each date for a given scenario."""
    sc   = fc[fc["scenario"] == scenario].copy()
    rows = []
    for date, grp in sc.groupby("date"):
        for col, _ in _STREAMS:
            vals = grp[col].to_numpy()
            rows.append({
                "date":   date,
                "stream": col,
                "P90":    float(np.percentile(vals, 10)),   # low
                "P50":    float(np.percentile(vals, 50)),   # median
                "P10":    float(np.percentile(vals, 90)),   # high
            })
    return pd.DataFrame(rows).sort_values(["stream", "date"]).reset_index(drop=True)


def _compute_scenario_range(fc: pd.DataFrame) -> pd.DataFrame:
    """For each stream + date, compute min and max of P50 across applicable scenarios.

    Returns DataFrame with columns: date, stream, scenario_low, scenario_high.
    """
    # Build per-scenario P50 table
    sc_p50: dict[str, pd.DataFrame] = {}
    for sc in ("main", "rate_calibrated", "uncapped"):
        fan = _compute_fan(fc, sc)
        sc_p50[sc] = fan[["date", "stream", "P50"]].rename(columns={"P50": f"P50_{sc}"})

    # Merge all scenarios on date + stream
    merged = sc_p50["main"].merge(sc_p50["rate_calibrated"], on=["date", "stream"])
    merged = merged.merge(sc_p50["uncapped"], on=["date", "stream"])

    rows = []
    for _, row in merged.iterrows():
        stream    = row["stream"]
        scenarios = _SCENARIO_STREAMS[stream]
        vals      = [row[f"P50_{s}"] for s in scenarios]
        rows.append({
            "date":          row["date"],
            "stream":        stream,
            "scenario_low":  float(min(vals)),
            "scenario_high": float(max(vals)),
        })
    return pd.DataFrame(rows).sort_values(["stream", "date"]).reset_index(drop=True)


def _print_2028_summary(fan_df: pd.DataFrame, sc_range: pd.DataFrame) -> None:
    end_date = pd.Timestamp("2028-01-01")
    print(f"\n=== 2028-01-01 uncertainty summary (proxy-estimated, extrapolated) ===")
    print("  P90 = low (10th pct), P10 = high (90th pct).")
    print("  fan_width%   = (P10 - P90) / P50 x 100   [parameter uncertainty]")
    print("  scenario_%   = (sc_high - sc_low) / P50 x 100  [method/assumption uncertainty]")
    print(f"  {'stream':>12}  {'P90':>15}  {'P50':>15}  {'P10':>15}  "
          f"{'fan%':>7}  {'sc_low':>15}  {'sc_high':>15}  {'sc%':>7}")

    end_fan = fan_df[fan_df["date"] == end_date]
    end_sc  = sc_range[sc_range["date"] == end_date]

    for col, _ in _STREAMS:
        rf = end_fan[end_fan["stream"] == col].iloc[0]
        rs = end_sc[end_sc["stream"]  == col].iloc[0]
        unit      = "MSCF" if "gas" in col else "STB"
        fan_pct   = (rf["P10"] - rf["P90"]) / rf["P50"] * 100
        sc_pct    = (rs["scenario_high"] - rs["scenario_low"]) / rf["P50"] * 100
        print(f"  {col:>12}  {rf['P90']:>15,.0f}  {rf['P50']:>15,.0f}  {rf['P10']:>15,.0f}  "
              f"{fan_pct:>6.2f}%  {rs['scenario_low']:>15,.0f}  "
              f"{rs['scenario_high']:>15,.0f}  {sc_pct:>6.2f}%")
        print(f"  {'':>12}  (units: {unit})")


def _plot_fan_chart(fan_df: pd.DataFrame, sc_range: pd.DataFrame,
                    cal_med: pd.DataFrame, h: pd.DataFrame) -> None:
    """Fan chart showing both the parameter band and the scenario band."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (col, ylabel) in zip(axes, _STREAMS):
        f  = fan_df[fan_df["stream"]  == col].sort_values("date")
        sr = sc_range[sc_range["stream"] == col].sort_values("date")
        c  = cal_med[cal_med["stream"]   == col].sort_values("date")

        # History
        ax.plot(h.index, h[col], color="black", lw=2.2, zorder=5,
                label="Observed history")

        # Scenario band (wider; draw first so parameter band is on top)
        ax.fill_between(sr["date"], sr["scenario_low"], sr["scenario_high"],
                        color="sandybrown", alpha=0.30, zorder=2,
                        label="Method / assumption scenarios\n(not a probability range)")

        # Parameter band: P90-P10
        ax.fill_between(f["date"], f["P90"], f["P10"],
                        color="steelblue", alpha=0.40, zorder=3,
                        label="10 parameter sets (P10–P90)")

        # P50 solid
        ax.plot(f["date"], f["P50"], color="steelblue", lw=1.8, zorder=4,
                label="P50 (main)")

        # Calibrated P50 dashed
        ax.plot(c["date"], c["cal_P50"], color="darkorange", lw=1.4,
                linestyle="--", zorder=4, label="P50 calibrated")

        ax.axvline(pd.Timestamp("2008-01-01"), color="gray", lw=1.0,
                   linestyle="--", alpha=0.6)
        ax.set_title(ylabel)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "T3 proxy-estimated, extrapolated: uncertainty fan chart 1998–2028\n"
        "Blue = 10 parameter sets (P10–P90);  "
        "Tan = method/assumption scenarios;  "
        "dashed = calibrated P50"
    )
    fig.tight_layout()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = C.FIGURES_DIR / "t3_fan_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def _plot_uncertainty_compare(fan_df: pd.DataFrame,
                               sc_range: pd.DataFrame) -> None:
    """% deviation from main P50 over time for both uncertainty bands."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (col, ylabel) in zip(axes, _STREAMS):
        f  = fan_df[fan_df["stream"]  == col].sort_values("date")
        sr = sc_range[sc_range["stream"] == col].sort_values("date")
        p50 = f["P50"].to_numpy()

        # Parameter band as % of P50
        p10_pct = (f["P10"].to_numpy() - p50) / p50 * 100
        p90_pct = (f["P90"].to_numpy() - p50) / p50 * 100

        # Scenario band as % of P50
        sc_hi_pct = (sr["scenario_high"].to_numpy() - p50) / p50 * 100
        sc_lo_pct = (sr["scenario_low"].to_numpy()  - p50) / p50 * 100

        dates = f["date"].to_numpy()

        # Scenario band (tan, wider)
        ax.fill_between(dates, sc_lo_pct, sc_hi_pct,
                        color="sandybrown", alpha=0.40,
                        label="Method / assumption scenarios")

        # Parameter band (blue, narrower)
        ax.fill_between(dates, p90_pct, p10_pct,
                        color="steelblue", alpha=0.50,
                        label="10 parameter sets (P10–P90)")

        ax.axhline(0, color="black", lw=1.2, zorder=5)
        ax.axvline(pd.Timestamp("2008-01-01"), color="gray", lw=1.0,
                   linestyle="--", alpha=0.6)
        ax.set_title(ylabel)
        ax.set_ylabel("% difference from main P50")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        "T3 proxy-estimated: uncertainty size comparison\n"
        "Y-axis = % deviation from main P50;  "
        "blue = parameter uncertainty;  tan = method/assumption uncertainty"
    )
    fig.tight_layout()
    out = C.FIGURES_DIR / "t3_uncertainty_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def main() -> None:
    print("Loading data...", flush=True)
    fc      = pd.read_csv(C.OUTPUTS_DIR / "t3_forecast_all.csv", parse_dates=["date"])
    history = read_history()
    h       = history.set_index("date")

    # Exclude 2008-01-01 starting row (observed, same for all ids -- zero fan width)
    fc_only = fc[fc["date"] > pd.Timestamp("2008-01-01")].copy()

    # Parameter uncertainty: P10/P50/P90 across 10 ids in 'main' scenario
    fan_df = _compute_fan(fc_only, "main")

    # Calibrated median for fan chart dashed reference line
    cal_df  = _compute_fan(fc_only, "rate_calibrated")
    cal_med = cal_df[["date", "stream", "P50"]].rename(columns={"P50": "cal_P50"})

    # Method / assumption uncertainty: min-max of median across scenarios
    sc_range = _compute_scenario_range(fc_only)

    # Merge scenario bounds into fan_df for CSV output
    out_df = fan_df.merge(sc_range, on=["date", "stream"], how="left")

    # Save CSV
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = C.OUTPUTS_DIR / "t3_p10_p50_p90.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}  ({len(out_df):,} rows, "
          f"columns: {list(out_df.columns)})", flush=True)

    # Print 2028 summary
    _print_2028_summary(fan_df, sc_range)

    # Figures
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_fan_chart(fan_df, sc_range, cal_med, h)
    _plot_uncertainty_compare(fan_df, sc_range)


if __name__ == "__main__":
    main()
