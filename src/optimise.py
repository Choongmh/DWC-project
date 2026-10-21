"""T3 Task A: random-sample 50,000 parameter sets and score them against the real history.

Run: .venv/bin/python -m src.optimise

Outputs:
  data/processed/t3_candidates_all.parquet  -- all 50,000 candidates, sorted by score
  outputs/t3_candidates_top500.csv          -- best 500 candidates
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.load_data import read_history, y_columns
from src.metrics import score_vs_history, split_y
from src.proxy import load_proxy, param_bounds, predict


def sample_candidates(bounds: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Draw n parameter sets uniformly at random inside [min, max] for each column."""
    bounds = bounds.loc[C.PARAM_COLS]
    rng = np.random.default_rng(seed)
    lo = bounds["min"].to_numpy()
    hi = bounds["max"].to_numpy()
    return lo + rng.random((n, len(lo))) * (hi - lo)


def score_candidates(X: np.ndarray, history: pd.DataFrame) -> pd.Series:
    """Predict curves for X and score each row against the real history. Returns a Series of scores."""
    preds = predict(X)
    split = split_y(preds)
    scores_df = score_vs_history(split, history)
    return scores_df["score"]


def case6_score(master: pd.DataFrame, history: pd.DataFrame) -> float:
    """Compute the score for Case 6 using score_vs_history() on the master table."""
    row = master.loc[master["case"] == 6, y_columns()].to_numpy(dtype=float)
    split = split_y(row)
    result = score_vs_history(split, history)
    return float(result["score"].iloc[0])


DISTANCE_COLS = ["perm_mult", "aquifer_pv", "fault_trans"]


def scale_for_distance(candidates: pd.DataFrame, bounds: pd.DataFrame) -> np.ndarray:
    """Scale DISTANCE_COLS to [0, 1] using param_bounds() min/max. poro_mult is excluded."""
    bounds = bounds.loc[DISTANCE_COLS]
    lo = bounds["min"].to_numpy()
    hi = bounds["max"].to_numpy()
    raw = candidates[DISTANCE_COLS].to_numpy()
    return (raw - lo) / (hi - lo)


def select_diverse(pool: pd.DataFrame, bounds: pd.DataFrame, n: int) -> pd.DataFrame:
    """Greedy farthest-point selection from pool.

    Picks the best-scoring candidate first, then repeatedly picks the candidate
    whose distance to its nearest already-picked candidate is largest, until n
    candidates are selected. Stops early if the pool is smaller than n.
    """
    if len(pool) < n:
        print(f"WARNING: near-optimal pool has only {len(pool)} candidates, fewer than {n}. "
              "Stopping without widening the band.")
        n = len(pool)

    scaled = scale_for_distance(pool, bounds)
    selected_idx = [0]  # best-scoring candidate is index 0 (pool is sorted by score)

    # min distance from each candidate to the nearest already-selected candidate
    min_dist = np.full(len(pool), np.inf)
    min_dist[0] = 0.0
    # initialise distances to the first selected point
    diff = scaled - scaled[0]
    min_dist = np.minimum(min_dist, np.sqrt((diff ** 2).sum(axis=1)))

    for _ in range(n - 1):
        next_idx = int(np.argmax(min_dist))
        selected_idx.append(next_idx)
        min_dist[next_idx] = 0.0
        diff = scaled - scaled[next_idx]
        min_dist = np.minimum(min_dist, np.sqrt((diff ** 2).sum(axis=1)))

    return pool.iloc[selected_idx].reset_index(drop=True)


def main() -> None:
    print("Loading proxy model and history...", flush=True)
    load_proxy()
    bounds = param_bounds()
    history = read_history()
    case6_df = pd.read_parquet(
        C.DATA_PROCESSED / "master_table.parquet",
        filters=[("case", "=", 6)],
        columns=["case"] + y_columns(),
    )

    print(f"Sampling {C.T3_N_SAMPLES:,} candidates (seed={C.T3_SEED})...", flush=True)
    X = sample_candidates(bounds, C.T3_N_SAMPLES, C.T3_SEED)

    print("Predicting and scoring...", flush=True)
    scores = score_candidates(X, history)

    param_df = pd.DataFrame(X, columns=C.PARAM_COLS)
    candidates = param_df.copy()
    candidates["score"] = scores.to_numpy()
    candidates = candidates.sort_values("score", kind="stable").reset_index(drop=True)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    all_path = C.DATA_PROCESSED / "t3_candidates_all.parquet"
    candidates.to_parquet(all_path, index=False)
    print(f"Saved {len(candidates):,} candidates -> {all_path}", flush=True)

    top = candidates.head(C.T3_TOP_K)
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    top_path = C.OUTPUTS_DIR / "t3_candidates_top500.csv"
    top.to_csv(top_path, index=False)
    print(f"Saved top {C.T3_TOP_K} candidates -> {top_path}", flush=True)

    best_score = float(candidates["score"].iloc[0])
    near_optimal = int((candidates["score"] <= best_score + C.T3_TIE_BAND).sum())
    c6_score = case6_score(case6_df, history)

    print(f"\nBest candidate score : {best_score:.6f}  (proxy-estimated)")
    print(f"Case 6 score         : {c6_score:.6f}  (simulator result, from master table)")
    print(f"Candidates within {C.T3_TIE_BAND} of best: {near_optimal:,}")

    # --- Task B: diverse selection from near-optimal pool ---
    print("\n--- Task B: greedy farthest-point selection ---", flush=True)
    pool = candidates.loc[candidates["score"] <= best_score + C.T3_TIE_BAND].reset_index(drop=True)
    picks = select_diverse(pool, bounds, C.T3_N_SELECT)

    out = picks[C.PARAM_COLS + ["score"]].copy()
    out.insert(0, "id", range(1, len(out) + 1))
    top10_path = C.OUTPUTS_DIR / "t3_top10_params.csv"
    out.to_csv(top10_path, index=False)
    print(f"Saved {len(out)} picks -> {top10_path}", flush=True)

    print(f"\n{'id':>3}  {'fault_trans':>12}  {'poro_mult':>10}  {'perm_mult':>10}  {'aquifer_pv':>11}  {'score':>10}")
    for _, row in out.iterrows():
        print(f"{int(row['id']):>3}  {row['fault_trans']:>12.6f}  {row['poro_mult']:>10.6f}  "
              f"{row['perm_mult']:>10.6f}  {row['aquifer_pv']:>11.6f}  {row['score']:>10.6f}")

    score_min = float(out["score"].min())
    score_max = float(out["score"].max())
    print(f"\nScore range among picks: {score_min:.6f} – {score_max:.6f}  (proxy-estimated)")

    scaled_picks = scale_for_distance(picks, bounds)
    n_picks = len(scaled_picks)
    min_pair_dist = np.inf
    for i in range(n_picks):
        for j in range(i + 1, n_picks):
            d = float(np.sqrt(((scaled_picks[i] - scaled_picks[j]) ** 2).sum()))
            if d < min_pair_dist:
                min_pair_dist = d
    print(f"Smallest pairwise scaled distance: {min_pair_dist:.6f}")


if __name__ == "__main__":
    main()
