"""Train-only GP bounds, optimizer, uncertainty and gas/oil diagnostics.

Run: python -m src.proxy_diagnostics
Compares the existing winning configuration under original and wider GP bounds.
No validation/test evaluation, model export, automatic tuning or uncertainty inflation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import config as C
from src.load_data import read_history, read_master, y_columns
from src.metrics import N_DATES, SCORE_MASK, history_arrays, nrmse, proxy_report, split_y
from src.proxy import ProxyPipeline, cv_splits, training_rows


def gas_ratio_summary(y, oil_floor, reference=None):
    """Ratios in MSCF/STB, using scored dates with finite gas and oil > floor."""
    parts = split_y(y)
    oil, gas = parts["oil_cum"][:, SCORE_MASK], parts["gas_cum"][:, SCORE_MASK]
    valid = np.isfinite(oil) & np.isfinite(gas) & (oil > oil_floor)
    ratios = gas[valid] / oil[valid]
    if not len(ratios):
        raise ValueError("No positive-oil points available for gas/oil check")
    median = float(np.median(ratios))
    ref = median if reference is None else float(reference)
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError("Gas/oil reference ratio must be positive and finite")
    deviations = np.abs(ratios / ref - 1)
    return {"oil_floor": float(oil_floor), "reference_ratio": ref,
            "n_used": int(valid.sum()), "n_excluded": int(valid.size-valid.sum()),
            "median_ratio": median, "min_ratio": float(ratios.min()), "max_ratio": float(ratios.max()),
            "mean_relative_deviation": float(deviations.mean()),
            "max_relative_deviation": float(deviations.max()),
            "within_spec_0_1_percent": bool(deviations.max() < 0.001)}


def paired_ratio_error(true, pred, oil_floor):
    """Compare predicted gas/oil with the same case/date simulator ratio."""
    t, p = split_y(true), split_y(pred)
    to, tg = t["oil_cum"][:, SCORE_MASK], t["gas_cum"][:, SCORE_MASK]
    po, pg = p["oil_cum"][:, SCORE_MASK], p["gas_cum"][:, SCORE_MASK]
    valid = (to > oil_floor) & (po > oil_floor) & (tg > 0)
    valid &= np.isfinite(to) & np.isfinite(tg) & np.isfinite(po) & np.isfinite(pg)
    if not valid.any():
        raise ValueError("No paired positive-production points")
    err = np.abs((pg[valid]/po[valid]) / (tg[valid]/to[valid]) - 1)
    return {"paired_n_used": int(valid.sum()), "paired_mean_relative_error": float(err.mean()),
            "paired_max_relative_error": float(err.max())}


def coverage_rows(true, raw, corrected, std, scales):
    """Report nominal +/-2 std coverage, without using truth to tune the bands."""
    if not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("Uncertainty must be finite and nonnegative")
    rows = []
    for i, stream in enumerate(C.CUM_STREAMS):
        block = slice(i*N_DATES, (i+1)*N_DATES)
        t, r, p, sd = [x[:, block][:, SCORE_MASK] for x in (true, raw, corrected, std)]
        positive = t > np.asarray(scales[stream])[:, None] * 1e-8
        for scope, mask in (("all_scored", np.ones(t.shape, dtype=bool)), ("positive_production", positive)):
            for center, mean in (("raw", r), ("corrected", p)):
                rows.append({"stream": stream, "scope": scope, "center": center,
                             "n_points": int(mask.sum()),
                             "coverage_2std": float(np.mean((np.abs(t-mean) <= 2*sd)[mask])) if mask.any() else None,
                             "mean_std_over_fit_scale": float(np.mean((sd/np.asarray(scales[stream])[:, None])[mask])) if mask.any() else None})
    return rows


def plot_diagnostics(summary, uncertainty):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["Oil", "Gas", "Water"]
    x = np.arange(3)
    for j, row in summary.iterrows():
        axes[0].bar(x + (j-0.5)*0.34, [row[f"{s}_nrmse_mean"] for s in C.CUM_STREAMS],
                    width=0.34, label=row.variant)
        u = uncertainty.loc[(uncertainty.variant == row.variant) &
                            (uncertainty.scope == "positive_production") & (uncertainty.center == "raw")]
        axes[1].bar(x + (j-0.5)*0.34, u.set_index("stream").loc[C.CUM_STREAMS, "coverage_2std"] * 100,
                    width=0.34, label=row.variant)
    axes[0].set_ylabel("Mean case NRMSE")
    axes[0].set_title("Prediction accuracy: same five training folds")
    axes[1].axhline(95.45, color="gray", linestyle="--", label="Nominal Gaussian +/-2 std")
    axes[1].set_ylim(0, 103)
    axes[1].set_ylabel("Points inside raw prediction +/-2 std (%)")
    axes[1].set_title("Uncertainty coverage: positive production only")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.text(0.02, 0.01, "Development diagnostics on cases 1-70. Validation/test not evaluated. Bands omit discarded-PC and model-selection uncertainty.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(C.FIGURES_DIR / "proxy_gp_diagnostics.png", dpi=160)
    plt.close(fig)


def run_diagnostics(train, history):
    ids = train.case.to_numpy()
    folds = cv_splits(ids)
    X, y = train[C.PARAM_COLS].to_numpy(float), train[y_columns()].to_numpy(float)
    hist = history_arrays(history)
    specs = [("original_bounds", 1e3, 1e-10), ("wider_bounds", 1e5, 1e-12)]
    summary, fold_rows, case_rows, uncertainty_rows, gas_rows, audit = [], [], [], [], [], []
    oil_floor_all = float(split_y(y)["oil_cum"].max()) * 1e-8
    gas_rows.append({"variant": "descriptive_only", "fold": 0, "source": "all_training_simulator",
                     **gas_ratio_summary(y, oil_floor_all)})
    original_oof = None
    for variant, upper, lower in specs:
        oof, raw_oof, std_oof = [np.full_like(y, np.nan) for _ in range(3)]
        scales = {s: np.zeros(len(y)) for s in C.CUM_STREAMS}
        variant_folds = []
        for fold, (fit, held) in enumerate(folds, 1):
            print(f"{variant}: fold {fold}/5", flush=True)
            model = ProxyPipeline(model="gp", nu=2.5, pca_extra=2, gp_restarts=5,
                                  gp_length_upper=upper, gp_noise_lower=lower,
                                  gp_alpha=1e-10, gp_trace_optimizer=True)
            start = perf_counter()
            model.fit(X[fit], y[fit])
            fit_seconds = perf_counter()-start
            raw, std = model.predict_raw(X[held], return_std=True)
            pred = model.correct_predictions(raw)
            if not np.isfinite(pred).all() or not np.isfinite(std).all() or (std < 0).any():
                raise RuntimeError("Invalid predictions or uncertainty")
            oof[held], raw_oof[held], std_oof[held] = pred, raw, std
            report, raw_report = proxy_report(y[held], pred, history), proxy_report(y[held], raw, history)
            row = {"variant": variant, "fold": fold, "fit_seconds": fit_seconds,
                   "fit_warning_count": 0, "optimizer_attempts": 0, "failed_attempts": 0,
                   "selected_failed_attempts": 0, "n_values_corrected": model.last_n_fixed,
                   "max_correction_scaled": model.last_max_correction_scaled}
            states = {}
            for s, state in model.streams_.items():
                scales[s][held] = state["scale"]
                states[s] = {k: v for k, v in state.items() if k in {
                    "scale", "auto_components", "n_components", "pca_error", "pca_tolerance_met",
                    "fit_warnings", "gp_kernels", "gp_optimizer_audit"}}
                for optim in state.get("gp_optimizer_audit", []):
                    row["optimizer_attempts"] += len(optim["runs"])
                    row["failed_attempts"] += sum(not r["success"] for r in optim["runs"])
                    row["selected_failed_attempts"] += int(not optim["selected_success"])
                row["fit_warning_count"] += sum(state.get("fit_warnings", {}).values())
                row[f"{s}_nrmse"] = report.loc[s, "nrmse_mean"]
                row[f"{s}_raw_nrmse"] = raw_report.loc[s, "nrmse_mean"]
            row["selection_metric"] = (row["oil_cum_nrmse"] + row["water_cum_nrmse"]) / 2
            fold_rows.append(row)
            variant_folds.append(row)
            audit.append({"variant": variant, "fold": fold, "fit_case_ids": ids[fit].tolist(),
                          "heldout_case_ids": ids[held].tolist(), "xmin": model.xmin_.tolist(),
                          "xmax": model.xmax_.tolist(), "streams": states})
            floor = model.streams_["oil_cum"]["scale"] * 1e-8
            reference = gas_ratio_summary(y[fit], floor)
            gas_rows.append({"variant": variant, "fold": fold, "source": "fold_fit_simulator", **reference})
            for source, values in (("heldout_simulator", y[held]), ("heldout_proxy", pred)):
                gas_rows.append({"variant": variant, "fold": fold, "source": source,
                                 **gas_ratio_summary(values, floor, reference["median_ratio"]),
                                 **(paired_ratio_error(y[held], pred, floor) if source == "heldout_proxy" else {})})
        if not all(np.isfinite(v).all() for v in (oof, raw_oof, std_oof)):
            raise RuntimeError("Incomplete out-of-fold predictions")
        report = proxy_report(y, oof, history)
        f = pd.DataFrame(variant_folds)
        row = {"variant": variant, "length_upper": upper, "noise_lower": lower, "alpha": 1e-10,
               "selection_metric": float(f.selection_metric.mean()),
               **{key: float(f[key].sum()) for key in ["fit_seconds", "fit_warning_count", "optimizer_attempts", "failed_attempts", "selected_failed_attempts"]},
               "max_correction_scaled": float(f.max_correction_scaled.max())}
        for s in C.CUM_STREAMS:
            row[f"{s}_nrmse_mean"] = float(report.loc[s, "nrmse_mean"])
            row[f"{s}_end_pct_err"] = float(report.loc[s, "end_pct_err"])
            errors = nrmse(split_y(oof)[s], split_y(y)[s], hist[s][SCORE_MASK].max())
            for i, case in enumerate(ids):
                case_rows.append({"variant": variant, "case": int(case), "stream": s, "nrmse": float(errors[i])})
        uncertainty_rows.extend({"variant": variant, **r} for r in coverage_rows(y, raw_oof, oof, std_oof, scales))
        if original_oof is None:
            original_oof = oof.copy()
        else:
            for s in C.CUM_STREAMS:
                delta = split_y(oof)[s] - split_y(original_oof)[s]
                scale = hist[s][SCORE_MASK].max()
                row[f"{s}_mean_prediction_change_nrmse"] = float(nrmse(split_y(oof)[s], split_y(original_oof)[s], scale).mean())
                row[f"{s}_max_point_change_over_history_max"] = float(np.max(np.abs(delta[:, SCORE_MASK])) / scale)
        summary.append(row)
        print(f"{variant}: oil/water NRMSE {row['selection_metric']:.9f}; "
              f"selected nonconverged fits {int(row['selected_failed_attempts'])}", flush=True)
    return (pd.DataFrame(summary), pd.DataFrame(fold_rows), pd.DataFrame(case_rows),
            pd.DataFrame(uncertainty_rows), pd.DataFrame(gas_rows), audit)


def main():
    train, history = training_rows(read_master()), read_history()
    print("GP diagnostics: cases 1-70 only. Original and wider bounds; no automatic promotion.", flush=True)
    with threadpool_limits(limits=1):
        summary, folds, cases, uncertainty, gas, audit = run_diagnostics(train, history)
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for frame, name in [(summary, "proxy_gp_bounds_comparison"), (folds, "proxy_gp_diagnostic_folds"),
                        (cases, "proxy_gp_diagnostic_case_errors"), (uncertainty, "proxy_gp_uncertainty"),
                        (gas, "proxy_gas_check")]:
        frame.to_csv(C.OUTPUTS_DIR / f"{name}.csv", index=False)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": C.SEED,
                "validation_evaluated": False, "test_evaluated": False, "gas_from_oil": False,
                "configuration": {"nu": 2.5, "pca_extra": 2, "gp_restarts": 5, "gp_alpha": 1e-10},
                "optimizer": "same scipy L-BFGS-B call as sklearn default; record all six attempts per component",
                "uncertainty": "pointwise retained-PC independent GP std including WhiteKernel; excludes discarded-PC error and parameter-estimation uncertainty; not transformed by physical corrections",
                "constant_stream_std": "zero by convention, not a guarantee of certainty outside fitting data",
                "coverage": "raw and corrected centers, all scored points and positive production separately; no calibration applied",
                "gas_check": "MSCF/STB, oil > fold training maximum * 1e-8; reference ratio fitted only on each fold's fitting cases; all-training row is descriptive only",
                "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]},
                "folds": audit}
    (C.OUTPUTS_DIR / "proxy_gp_diagnostic_audit.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    plot_diagnostics(summary, uncertainty)
    print(summary.to_string(index=False))
    print("Saved diagnostic reports. Existing comparison reports unchanged; validation/test not evaluated.")


if __name__ == "__main__":
    main()
