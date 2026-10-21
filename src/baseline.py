"""
Step 1: Baseline (Track 1). No ML.

Ranks the 100 EXISTING simulated cases by how well they match the real field history.
The best score here is RMSE_baseline, which the ML-optimised result (Step 3) must beat.

Usage (from the repo root, after `python -m src.load_data`):
    python -m src.baseline

Outputs:
    outputs/baseline_ranking.csv        all 100 cases, ranked (rank 1 = best match)
    outputs/figures/baseline_top10.png  top 10 vs real history
"""
import matplotlib

matplotlib.use("Agg")   # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C
from src.load_data import read_curves, read_history, read_params
from src.metrics import curves_to_arrays, score_vs_history

EQUAL_RATE_WEIGHTS = {s: 1 / len(C.RATE_STREAMS) for s in C.RATE_STREAMS}


def rank_cases() -> pd.DataFrame:
    params, curves, history = read_params(), read_curves(), read_history()

    cases, cum = curves_to_arrays(curves, C.CUM_STREAMS)
    _, rate = curves_to_arrays(curves, C.RATE_STREAMS)

    main = score_vs_history(cum, history)                                    # main score (cumulative)
    second = score_vs_history(rate, history, C.RATE_STREAMS, EQUAL_RATE_WEIGHTS)   # cross-check (rates)
    second = second.rename(columns={"score": "score_rate"})

    df = pd.DataFrame({"case": cases}).join(main).join(second)
    df = params.merge(df, on="case")
    df["rank"] = df["score"].rank(method="first").astype(int)
    df["rank_rate"] = df["score_rate"].rank(method="first").astype(int)
    return df.sort_values("rank").reset_index(drop=True)


def plot_top(ranking: pd.DataFrame, k: int = C.TOP_K) -> None:
    curves, history = read_curves(), read_history()
    top = ranking.head(k)["case"].tolist()
    labels = {"oil_cum": "Oil cumulative [STB]", "gas_cum": "Gas cumulative [MSCF]",
              "water_cum": "Water cumulative [STB]"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, s in zip(axes, C.CUM_STREAMS):
        for case, g in curves.groupby("case"):
            if case not in top:
                ax.plot(g["date"], g[s], color="#cccccc", lw=0.6, zorder=1)
        for case in top:
            g = curves[curves.case == case]
            ax.plot(g["date"], g[s], lw=1.4, zorder=2, label=f"Case {case}")
        ax.scatter(history["date"], history[s], color="black", s=14, zorder=3, label="Real history")
        ax.set_title(labels[s])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Baseline: top {k} of the 100 simulated cases (colour) vs real history (black); "
                 f"grey = other cases")
    fig.tight_layout()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / "baseline_top10.png", dpi=150)
    plt.close(fig)

def plot_top_only(ranking: pd.DataFrame, k: int = C.TOP_K, zoom_from: str = "2003-01-01") -> None:
    """Close-up of the top k only (no other cases).
    Row 1: top k vs real history, zoomed to the late period where the curves separate.
    Row 2: error over time = (case - history) / max(history), in %, so small differences are visible."""
    curves, history = read_curves(), read_history()
    top = ranking.head(k)["case"].tolist()
    labels = {"oil_cum": "Oil cumulative [STB]", "gas_cum": "Gas cumulative [MSCF]",
              "water_cum": "Water cumulative [STB]"}
    h = history.set_index("date")
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for j, s in enumerate(C.CUM_STREAMS):
        ax_c, ax_e = axes[0, j], axes[1, j]
        scale = h[s].max()
        for i, case in enumerate(top):
            g = curves[curves.case == case].set_index("date")
            late = g[g.index >= zoom_from]
            lab = f"#{i + 1} Case {case}"
            ax_c.plot(late.index, late[s], color=colors[i % 10], lw=1.6, label=lab)
            err = (g[s] - h[s]) / scale * 100
            ax_e.plot(err.index, err, color=colors[i % 10], lw=1.6, label=lab)
        hl = h[h.index >= zoom_from]
        ax_c.scatter(hl.index, hl[s], color="black", s=22, zorder=3, label="Real history")
        ax_e.axhline(0, color="black", lw=1.2)
        ax_c.set_title(f"{labels[s]}  (from {zoom_from[:4]})")
        ax_e.set_title(f"{labels[s].split(' [')[0]}: error vs history [% of field max]")
        ax_e.set_ylabel("case − history  [%]")
        for ax in (ax_c, ax_e):
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Baseline close-up: top {k} cases only. Top: curves vs real history (black). "
                 f"Bottom: error over time (0 = perfect match; above 0 = case produces too much)")
    fig.tight_layout()
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / "baseline_top10_closeup.png", dpi=150)
    plt.close(fig)

def main() -> None:
    ranking = rank_cases()
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(C.OUTPUTS_DIR / "baseline_ranking.csv", index=False)
    plot_top(ranking)
    plot_top_only(ranking)

    top = ranking.head(C.TOP_K)
    cols = ["rank", "case", "split"] + C.PARAM_COLS + \
           [f"nrmse_{s}" for s in C.CUM_STREAMS] + ["score", "rank_rate"]
    pd.set_option("display.width", 200)
    print(f"\n=== Baseline top {C.TOP_K} (score = weighted NRMSE on cumulative, lower is better) ===")
    print(top[cols].round(4).to_string(index=False))
    print(f"\nRMSE_baseline (best case)       : {top['score'].iloc[0]:.4f}  (case {top['case'].iloc[0]})")
    print(f"RMSE_baseline (mean of top {C.TOP_K})  : {top['score'].mean():.4f}")
    overlap = len(set(top["case"]) & set(ranking.nsmallest(C.TOP_K, "score_rate")["case"]))
    print(f"Top-{C.TOP_K} overlap with the rate-based ranking: {overlap}/{C.TOP_K}")
    print(f"\nSaved {C.OUTPUTS_DIR / 'baseline_ranking.csv'}, figures/baseline_top10.png and figures/baseline_top10_closeup.png")

if __name__ == "__main__":
    main()