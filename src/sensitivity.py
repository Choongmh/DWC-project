"""
Sensitivity analysis (Track 1, guide section 7a): which parameter affects production most?

For each case we extract a few KEY OUTPUTS from its curves, then measure how strongly each of the
4 parameters drives each output (Spearman rank correlation: -1 .. +1, 0 = no effect).

Only train + validation cases (1-85) are used: the test cases stay untouched until the final proxy score.

Usage (from the repo root, after `python -m src.load_data`):
    python -m src.sensitivity

Outputs:
    outputs/sensitivity_key_outputs.csv     one row per case: parameters + key outputs
    outputs/sensitivity_spearman.csv        correlation table (parameters x key outputs)
    outputs/figures/sensitivity_tornado.png ranked bars: effect of each parameter on each key output
    outputs/figures/sensitivity_scatter.png scatter plots behind the numbers
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.load_data import read_curves, read_history, read_params
from src.metrics import curves_to_arrays, score_vs_history

WATER_CUT_LIMIT = 0.05   # water breakthrough = water cut (water / (oil + water)) first above 5 %
PLATEAU_FRACTION = 0.95  # plateau ends when the oil rate first drops below 95 % of its maximum

KEY_OUTPUTS = {
    "oil_cum_2008": "Cumulative oil 2008 [STB]",
    "water_cum_2008": "Cumulative water 2008 [STB]",
    "gas_cum_2008": "Cumulative gas 2008 [MSCF]",
    "breakthrough_yrs": "Water breakthrough [years after start]",
    "plateau_end_yrs": "Oil plateau ends [years after start]",
    "match_score": "History-match score (lower = better)",
}


def _first_crossing_years(dates: pd.Series, values: np.ndarray, limit: float, above: bool,
                          producing: np.ndarray) -> float:
    """Years after production start when `values` first crosses `limit` (linear interpolation
    between quarters), only looking at quarters where the field is producing. NaN if it never crosses."""
    t = (dates - C.SCORE_START).dt.days.to_numpy() / 365.25
    hit = values > limit if above else values < limit
    hit &= np.cumsum(producing) > 0            # ignore the quarters before production starts
    if not hit.any():
        return np.nan
    i = int(np.argmax(hit))
    if i == 0:
        return t[0]
    v0, v1 = values[i - 1], values[i]
    frac = (limit - v0) / (v1 - v0) if v1 != v0 else 0.0
    return t[i - 1] + frac * (t[i] - t[i - 1])


def key_outputs(cases_subset=None) -> pd.DataFrame:
    params, curves, history = read_params(), read_curves(), read_history()
    if cases_subset is not None:
        params = params[params.case.isin(cases_subset)]
        curves = curves[curves.case.isin(cases_subset)]

    rows = []
    for case, g in curves.groupby("case"):
        g = g.sort_values("date")
        oil, water = g["oil_rate"].to_numpy(), g["water_rate"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            wcut = np.nan_to_num(water / (oil + water))
        rows.append({
            "case": case,
            "oil_cum_2008": g["oil_cum"].iloc[-1],
            "water_cum_2008": g["water_cum"].iloc[-1],
            "gas_cum_2008": g["gas_cum"].iloc[-1],
            "breakthrough_yrs": _first_crossing_years(g["date"], wcut, WATER_CUT_LIMIT, True, oil > 0),
            "plateau_end_yrs": _first_crossing_years(g["date"], oil, PLATEAU_FRACTION * oil.max(), False, oil > 0),
        })
    out = params.merge(pd.DataFrame(rows), on="case")

    cases, cum = curves_to_arrays(curves, C.CUM_STREAMS)
    score = pd.DataFrame({"case": cases, "match_score": score_vs_history(cum, history)["score"]})
    return out.merge(score, on="case")


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = parameters, columns = key outputs, values = Spearman rank correlation."""
    corr = df[C.PARAM_COLS + list(KEY_OUTPUTS)].corr(method="spearman")
    return corr.loc[C.PARAM_COLS, list(KEY_OUTPUTS)]


def real_field_reference() -> dict:
    """The same key outputs for the REAL field, to mark on the scatter plots."""
    h = read_history().sort_values("date")
    oil, water = h["oil_rate"].to_numpy(), h["water_rate"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        wcut = np.nan_to_num(water / (oil + water))
    return {
        "oil_cum_2008": h["oil_cum"].iloc[-1],
        "water_cum_2008": h["water_cum"].iloc[-1],
        "gas_cum_2008": h["gas_cum"].iloc[-1],
        "breakthrough_yrs": _first_crossing_years(h["date"], wcut, WATER_CUT_LIMIT, True, oil > 0),
        "plateau_end_yrs": _first_crossing_years(h["date"], oil, PLATEAU_FRACTION * oil.max(), False, oil > 0),
    }


def plot_tornado(corr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(KEY_OUTPUTS), figsize=(22, 4.2), sharex=True)
    for ax, (key, label) in zip(axes, KEY_OUTPUTS.items()):
        s = corr[key].reindex(corr[key].abs().sort_values().index)   # biggest effect on top
        colors = ["#c0392b" if v < 0 else "#2471a3" for v in s]
        ax.barh(s.index, s.values, color=colors)
        for y, v in enumerate(s.values):
            ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-1.5, 1.5)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Sensitivity: Spearman correlation of each parameter with each key output "
                 "(blue = output goes UP when the parameter goes up, red = goes DOWN; |value| near 1 = strong)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "sensitivity_tornado.png", dpi=150)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    keys = ["oil_cum_2008", "water_cum_2008", "breakthrough_yrs", "match_score"]
    ref = real_field_reference()
    fig, axes = plt.subplots(len(keys), len(C.PARAM_COLS), figsize=(18, 14))
    for i, key in enumerate(keys):
        for j, p in enumerate(C.PARAM_COLS):
            ax = axes[i, j]
            ax.scatter(df[p], df[key], s=14, alpha=0.75, color="#2471a3")
            if key in ref:
                ax.axhline(ref[key], color="black", ls="--", lw=1, label="Real field")
            ax.set_title(f"rho = {corr.loc[p, key]:+.2f}", fontsize=9)
            if i == len(keys) - 1:
                ax.set_xlabel(p)
            if j == 0:
                ax.set_ylabel(KEY_OUTPUTS[key], fontsize=9)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Parameters (x) vs key outputs (y), train + validation cases. Dashed line = real field value")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "sensitivity_scatter.png", dpi=130)
    plt.close(fig)


def main() -> None:
    cases = list(C.SPLIT["train"]) + list(C.SPLIT["validation"])   # test cases stay unseen
    df = key_outputs(cases)
    corr = spearman_table(df)

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OUTPUTS_DIR / "sensitivity_key_outputs.csv", index=False)
    corr.to_csv(C.OUTPUTS_DIR / "sensitivity_spearman.csv")
    plot_tornado(corr)
    plot_scatter(df, corr)

    pd.set_option("display.width", 200)
    print(f"\n=== Spearman correlation, {len(df)} cases (train + validation) ===")
    print(corr.round(2).to_string())
    print("\nStrongest driver of each key output:")
    for key in KEY_OUTPUTS:
        p = corr[key].abs().idxmax()
        print(f"  {key:18s} <- {p:12s} (rho = {corr.loc[p, key]:+.2f})")
    ref = real_field_reference()
    print(f"\nReal field: water breakthrough {ref['breakthrough_yrs']:.2f} yrs, "
          f"plateau ends {ref['plateau_end_yrs']:.2f} yrs after start")
    print("\nSaved sensitivity_key_outputs.csv, sensitivity_spearman.csv, "
          "figures/sensitivity_tornado.png, figures/sensitivity_scatter.png")


if __name__ == "__main__":
    main()