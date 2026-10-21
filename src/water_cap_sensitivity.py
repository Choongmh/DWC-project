"""T3 Task J: sensitivity of 2028 water forecast to the total-liquid rate cap.

Four cap levels (all read from read_history(), never hardcoded):
  (a) last_quarter -- 2008-01-01 total liquid rate (last observed quarterly rate)
  (b) last_9q_max  -- max observed 2006-01-01 to 2008-01-01 (current main-scenario cap)
  (c) hist_max     -- maximum total liquid rate observed anywhere in the history
  (d) uncapped     -- no cap

Integration reuses _integrate() from forecast_all; no model re-fitting is done.

Reads:
  outputs/t3_forecast_all.csv  (main oil rates + uncapped water rates)
  history via read_history()

Writes:
  outputs/t3_water_cap_sensitivity.csv
  outputs/figures/t3_water_cap_sensitivity.png

Run: .venv/bin/python -m src.water_cap_sensitivity
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.load_data import read_history
from src.forecast_all import _integrate   # reuse; do not rewrite

_FIT_END  = pd.Timestamp("2008-01-01")
_FC_DATES = C.FORECAST_DATES              # DatetimeIndex 2008-04-01 to 2028-01-01
_END_DATE = pd.Timestamp("2028-01-01")


def _first_binding_year(oil_rates: np.ndarray, water_uncap: np.ndarray,
                        cap: float, fc_dates: pd.DatetimeIndex) -> int | None:
    """Return the calendar year of the first forecast quarter where the cap is active."""
    for date, oil_r, wat_r in zip(fc_dates, oil_rates, water_uncap):
        if wat_r > cap - oil_r:
            return date.year
    return None


def main() -> None:  # noqa: C901
    print("Loading data...", flush=True)
    history = read_history()
    h       = history.set_index("date")
    fc      = pd.read_csv(C.OUTPUTS_DIR / "t3_forecast_all.csv", parse_dates=["date"])

    fc_only = fc[fc["date"] > _FIT_END].copy()

    obs_water_cum_2008 = float(h.loc["2008-01-01", "water_cum"])
    ids = sorted(fc["id"].unique())

    # ── Cap levels (all from read_history) ──────────────────────────────────
    # (a) last observed quarterly rate = 2008-01-01 row (rate = Q4-2007 avg daily prod)
    cap_a = float(h.loc["2008-01-01", "oil_rate"]) + float(h.loc["2008-01-01", "water_rate"])

    # (b) max total liquid over last 9 quarters (same window as forecast_all._build_forecast)
    hist_window = h.loc["2006-01-01":"2008-01-01"]
    cap_b = float((hist_window["oil_rate"] + hist_window["water_rate"]).max())

    # (c) max total liquid anywhere in the full history
    cap_c = float((h["oil_rate"] + h["water_rate"]).max())

    # (d) no cap
    cap_d = float("inf")

    caps = [
        ("last_quarter", cap_a),
        ("last_9q_max",  cap_b),
        ("hist_max",     cap_c),
        ("uncapped",     cap_d),
    ]

    print("\n=== Cap levels ===")
    desc_map = {
        "last_quarter": f"(a) last_quarter : {cap_a:.1f} STB/d  [2008-01-01 observed total liquid]",
        "last_9q_max":  f"(b) last_9q_max  : {cap_b:.1f} STB/d  [current main-scenario cap]",
        "hist_max":     f"(c) hist_max     : {cap_c:.1f} STB/d  [full-history maximum]",
        "uncapped":     "(d) uncapped     : no cap",
    }
    for label, _ in caps:
        print(f"  {desc_map[label]}")

    # ── Per-cap integration ──────────────────────────────────────────────────
    main_fc  = fc_only[fc_only["scenario"] == "main"]
    uncap_fc = fc_only[fc_only["scenario"] == "uncapped"]

    rows       = []
    bind_years = {label: [] for label, _ in caps}

    for label, cap_val in caps:
        for id_ in ids:
            oil_r = (main_fc[main_fc["id"]  == id_]
                     .sort_values("date")["oil_rate"].to_numpy())
            wat_u = (uncap_fc[uncap_fc["id"] == id_]
                     .sort_values("date")["water_rate"].to_numpy())

            if np.isinf(cap_val):
                water_rates = wat_u
            else:
                water_rates = np.minimum(wat_u, np.maximum(cap_val - oil_r, 0.0))

            water_cums = _integrate(water_rates, _FC_DATES,
                                    obs_water_cum_2008, _FIT_END)

            by = _first_binding_year(oil_r, wat_u, cap_val, _FC_DATES)
            bind_years[label].append(by)

            for date, wc in zip(_FC_DATES, water_cums):
                rows.append({
                    "cap_label": label,
                    "cap_stbd":  None if np.isinf(cap_val) else round(cap_val, 1),
                    "id":        id_,
                    "date":      date,
                    "water_cum": float(wc),
                })

    df = pd.DataFrame(rows)

    # ── Verification checks ──────────────────────────────────────────────────
    print("\n=== Verification checks ===")
    main_end  = fc_only[(fc_only["scenario"] == "main")     & (fc_only["date"] == _END_DATE)]
    uncap_end = fc_only[(fc_only["scenario"] == "uncapped") & (fc_only["date"] == _END_DATE)]
    b_end     = df[(df["cap_label"] == "last_9q_max") & (df["date"] == _END_DATE)]
    d_end     = df[(df["cap_label"] == "uncapped")    & (df["date"] == _END_DATE)]

    b_ok = d_ok = True
    for id_ in ids:
        b_val = float(b_end[b_end["id"]   == id_]["water_cum"].iloc[0])
        m_val = float(main_end[main_end["id"] == id_]["water_cum"].iloc[0])
        if abs(b_val - m_val) > 1.0:
            print(f"  FAIL (b) id={id_}: reintegrated={b_val:.0f}  main_csv={m_val:.0f}")
            b_ok = False

        d_val = float(d_end[d_end["id"]     == id_]["water_cum"].iloc[0])
        u_val = float(uncap_end[uncap_end["id"] == id_]["water_cum"].iloc[0])
        if abs(d_val - u_val) > 1.0:
            print(f"  FAIL (d) id={id_}: reintegrated={d_val:.0f}  uncapped_csv={u_val:.0f}")
            d_ok = False

    print(f"  (b) re-integration == main scenario (tolerance 1 STB): "
          f"{'PASS' if b_ok else 'FAIL'}")
    print(f"  (d) re-integration == uncapped scenario (tolerance 1 STB): "
          f"{'PASS' if d_ok else 'FAIL'}")

    if not (b_ok and d_ok):
        raise RuntimeError("Verification failed.")

    # ── 2028 summary table ───────────────────────────────────────────────────
    print(f"\n=== 2028-01-01 water cumulative [STB] by cap level ===")
    print(f"  {'label':>14}  {'cap STB/d':>12}  {'min':>15}  {'median':>15}  "
          f"{'max':>15}  {'binding from (median id)'}")
    end_df = df[df["date"] == _END_DATE]
    for label, cap_val in caps:
        vals    = end_df[end_df["cap_label"] == label]["water_cum"].to_numpy()
        bys     = [y for y in bind_years[label] if y is not None]
        bstr    = (str(int(np.median(bys))) + " (median year)") if bys else "never binding"
        cap_str = f"{cap_val:.1f}" if not np.isinf(cap_val) else "none"
        print(f"  {label:>14}  {cap_str:>12}  "
              f"{vals.min():>15,.0f}  {np.median(vals):>15,.0f}  "
              f"{vals.max():>15,.0f}  {bstr}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = C.OUTPUTS_DIR / "t3_water_cap_sensitivity.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}  ({len(df):,} rows)", flush=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    label_legend = {
        "last_quarter": f"(a) Last quarter cap: {cap_a:.0f} STB/d",
        "last_9q_max":  f"(b) Last-9q max cap: {cap_b:.0f} STB/d  [main]",
        "hist_max":     f"(c) History max cap: {cap_c:.0f} STB/d",
        "uncapped":     "(d) No cap",
    }
    colors     = {"last_quarter": "steelblue", "last_9q_max": "darkorange",
                  "hist_max": "seagreen",      "uncapped":    "firebrick"}
    linestyls  = {"last_quarter": "-",         "last_9q_max": "--",
                  "hist_max":     "-.",         "uncapped":    ":"}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(h.index, h["water_cum"], color="black", lw=2.2, zorder=5,
            label="Observed history")
    ax.axvline(_FIT_END, color="gray", lw=1.0, linestyle="--", alpha=0.6)

    for label, _ in caps:
        med = (df[df["cap_label"] == label]
               .groupby("date")["water_cum"].median().reset_index())
        ax.plot(med["date"], med["water_cum"],
                color=colors[label], lw=1.8, linestyle=linestyls[label],
                label=label_legend[label])

    ax.set_title(
        "T3 proxy-estimated, extrapolated: "
        "water cumulative sensitivity to total-liquid cap assumption\n"
        "Lines = median across 10 parameter sets"
    )
    ax.set_ylabel("Water cumulative [STB]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = C.FIGURES_DIR / "t3_water_cap_sensitivity.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Saved {out_fig}", flush=True)


if __name__ == "__main__":
    main()
