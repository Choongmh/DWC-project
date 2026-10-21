"""T3 Task C: build and check the 10 selected proxy curves (1998-2008 only).

Run: .venv/bin/python -m src.t3_curves

Outputs:
  data/processed/t3_top10_curves.parquet
  outputs/t3_top10_curves.csv
  outputs/figures/t3_top10_curves.png
  outputs/figures/t3_top10_closeup.png

Rate note: add_rates() assigns 0 to the first date (1998-01-01) because the cumulative
difference is 0 there (prepend trick). No NaN is produced.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.load_data import add_rates, read_history
from src.metrics import split_y
from src.proxy import load_proxy, predict


def build_curves(top10: pd.DataFrame) -> pd.DataFrame:
    """Predict proxy curves for the 10 picks and return a long-format table with rates.

    Rates are computed by add_rates() from src/load_data.py:
    rate(t) = (cum(t) - cum(t-1)) / days_in_quarter, assigned to the interval-ending date.
    The first date (1998-01-01) gets rate = 0 (no prior quarter).
    """
    X = top10[C.PARAM_COLS].to_numpy(dtype=float)
    preds = predict(X)           # (10, 123)
    streams = split_y(preds)     # {stream: (10, 41)}

    n = len(top10)
    rows: dict = {
        "id": np.repeat(top10["id"].to_numpy(), len(C.HISTORY_DATES)),
        "date": np.tile(C.HISTORY_DATES, n),
    }
    for s in C.CUM_STREAMS:
        rows[s] = streams[s].reshape(-1)

    df = pd.DataFrame(rows).sort_values(["id", "date"]).reset_index(drop=True)
    df = add_rates(df)
    return df[["id", "date"] + C.CUM_STREAMS + C.RATE_STREAMS]


def print_end_table(curves: pd.DataFrame, history: pd.DataFrame) -> None:
    """Print proxy vs observed cumulative values at 2008-01-01 with % gap."""
    end_date = C.HISTORY_DATES[-1]
    hist_end = history.set_index("date").loc[end_date]
    end = curves[curves["date"] == end_date].set_index("id")

    stream_labels = {"oil_cum": "Oil [STB]", "gas_cum": "Gas [MSCF]", "water_cum": "Water [STB]"}
    print("\n=== Proxy-estimated cumulative at 2008-01-01 vs observed history (proxy-estimated) ===")
    print("    % gap = (proxy − observed at 2008-01-01) / (observed at 2008-01-01) × 100")
    header = f"{'id':>3}  " + "  ".join(f"{lbl:>28}" for lbl in stream_labels.values())
    print(header)
    for id_ in sorted(end.index):
        row_str = f"{id_:>3}  "
        for s, lbl in stream_labels.items():
            proxy_val = float(end.loc[id_, s])
            hist_val = float(hist_end[s])
            pct = (proxy_val - hist_val) / hist_val * 100
            row_str += f"  {proxy_val:>14,.0f} ({pct:+6.1f}%)"
        print(row_str)
    print("% gap = (proxy − observed) / observed × 100; + means proxy too high")


def print_score_comparison(top10: pd.DataFrame) -> None:
    """Compare our 10 picks' mean proxy score to the baseline top-10 mean simulator score.

    Baseline scores are read from outputs/baseline_ranking.csv, column 'score',
    sorted ascending to select the 10 lowest-scoring (best-matching) cases.
    """
    baseline = pd.read_csv(C.OUTPUTS_DIR / "baseline_ranking.csv")
    baseline_top10_mean = float(baseline.nsmallest(10, "score")["score"].mean())
    our_mean = float(top10["score"].mean())
    print(f"\nBaseline top-10 mean score : {baseline_top10_mean:.6f}"
          "  (simulator curves, from baseline_ranking.csv column 'score')")
    print(f"Our 10 picks mean score    : {our_mean:.6f}  (proxy-estimated)")


def plot_curves(curves: pd.DataFrame, history: pd.DataFrame) -> None:
    """Figure 1: 3 panels of cumulative curves, 10 proxy curves + thick observed history line."""
    labels = {"oil_cum": "Oil cumulative [STB]",
              "gas_cum": "Gas cumulative [MSCF]",
              "water_cum": "Water cumulative [STB]"}
    colors = plt.cm.tab10.colors
    ids = sorted(curves["id"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, s in zip(axes, C.CUM_STREAMS):
        for i, id_ in enumerate(ids):
            g = curves[curves["id"] == id_]
            ax.plot(g["date"], g[s], color=colors[i % 10], lw=1.0, alpha=0.85, label=f"id {id_}")
        ax.plot(history["date"], history[s], color="black", lw=2.2, zorder=5, label="Observed history")
        ax.set_title(labels[s])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("T3 proxy-estimated top-10 curves vs observed history (1998–2008)")
    fig.tight_layout()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = C.FIGURES_DIR / "t3_top10_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def plot_closeup(curves: pd.DataFrame, history: pd.DataFrame,
                 zoom_from: str = "2005-01-01") -> None:
    """Figure 2: 2-row × 3-col close-up (2005-2008).

    Row 1: zoomed proxy curves vs observed history.
    Row 2: error = (proxy - observed) / max(history) * 100, with a zero line.
    Denominator matches src/baseline.py plot_top_only() and the official NRMSE scale,
    so this figure is directly comparable with baseline_top10_closeup.png.
    Style follows src/baseline.py plot_top_only().
    """
    labels = {"oil_cum": "Oil cumulative [STB]",
              "gas_cum": "Gas cumulative [MSCF]",
              "water_cum": "Water cumulative [STB]"}
    colors = plt.cm.tab10.colors
    h = history.set_index("date")
    ids = sorted(curves["id"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for j, s in enumerate(C.CUM_STREAMS):
        ax_c, ax_e = axes[0, j], axes[1, j]
        scale = float(h[s].max())
        for i, id_ in enumerate(ids):
            g = curves[curves["id"] == id_].set_index("date")
            late = g[g.index >= zoom_from]
            ax_c.plot(late.index, late[s], color=colors[i % 10], lw=1.6, label=f"id {id_}")
            err = (g[s] - h[s]) / scale * 100
            err_late = err[err.index >= zoom_from]
            ax_e.plot(err_late.index, err_late, color=colors[i % 10], lw=1.6, label=f"id {id_}")
        hl = h[h.index >= zoom_from]
        ax_c.scatter(hl.index, hl[s], color="black", s=22, zorder=3, label="Observed history")
        ax_e.axhline(0, color="black", lw=1.2)
        ax_c.set_title(f"{labels[s]}  (from {zoom_from[:4]})")
        ax_e.set_title(f"{labels[s].split(' [')[0]}: error [% of history max]")
        ax_e.set_ylabel("(proxy − observed) / max(history)  [%]")
        for ax in (ax_c, ax_e):
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "T3 proxy-estimated close-up (2005–2008). "
        "Top: proxy curves vs history (black). "
        "Bottom: (proxy − observed) / max(history) [%]  — same denominator as NRMSE and baseline_top10_closeup.png"
    )
    fig.tight_layout()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = C.FIGURES_DIR / "t3_top10_closeup.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def main() -> None:
    print("Loading proxy model and history...", flush=True)
    load_proxy()
    history = read_history()
    top10 = pd.read_csv(C.OUTPUTS_DIR / "t3_top10_params.csv")

    print("Predicting curves for 10 parameter sets...", flush=True)
    curves = build_curves(top10)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    curves.to_parquet(C.DATA_PROCESSED / "t3_top10_curves.parquet", index=False)
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    curves.to_csv(C.OUTPUTS_DIR / "t3_top10_curves.csv", index=False)
    print("Saved t3_top10_curves.parquet and t3_top10_curves.csv", flush=True)

    print_end_table(curves, history)
    print_score_comparison(top10)
    plot_curves(curves, history)
    plot_closeup(curves, history)


if __name__ == "__main__":
    main()
