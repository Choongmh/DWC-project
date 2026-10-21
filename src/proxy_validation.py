"""Fixed T2 validation: fit cases 1-70, evaluate 71-85, never evaluate 86-100.

Run: python -m src.proxy_validation
No automatic tuning, uncertainty calibration, final refit or model export.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from src import config as C
from src.load_data import read_history, read_master, y_columns
from src.metrics import N_DATES, SCORE_MASK, history_arrays, nrmse, proxy_report, score_vs_history, split_y
from src.proxy import ProxyPipeline, training_rows
from src.proxy_diagnostics import coverage_rows


# Fixed before inspecting validation. This is the original-bounds CV winner.
GP_SETTINGS = dict(model="gp", nu=2.5, pca_extra=2, gp_restarts=5,
                   gp_length_upper=1e3, gp_noise_lower=1e-10, gp_alpha=1e-10,
                   gp_trace_optimizer=True)
CV_CANDIDATE = "gp_nu2.5_extra2"
COLORS = ["#2471a3", "#d9822b", "#28946c"]
LABELS = ["Oil", "Gas", "Water"]
UNITS = ["STB", "MSCF", "STB"]


def validation_rows(master):
    train = training_rows(master)
    val = master.loc[master.split == "validation"].sort_values("case").copy()
    if not np.isfinite(val[C.PARAM_COLS+y_columns()].to_numpy(float)).all():
        raise ValueError("Non-finite validation values")
    return train, val


def cumulative_rates(curves):
    """Average daily rate for the interval ENDING at each history date.

    The first date has no preceding interval and is explicitly undefined (NaN).
    """
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.shape[1] != N_DATES or not np.isfinite(curves).all():
        raise ValueError("Expected finite (n, 41) cumulative curves")
    days = np.diff(C.HISTORY_DATES.to_numpy()) / np.timedelta64(1, "D")
    return np.column_stack([np.full(len(curves), np.nan), np.diff(curves, axis=1)/days])


def ranking_report(case_ids, score_true, score_proxy):
    table = pd.DataFrame({"case": case_ids, "score_true": score_true, "score_proxy": score_proxy})
    table["signed_score_error"] = table.score_proxy-table.score_true
    table["abs_score_error"] = table.signed_score_error.abs()
    table["true_rank"] = table.score_true.rank(method="average")
    table["proxy_rank"] = table.score_proxy.rank(method="average")
    rho = float(spearmanr(table.score_true, table.score_proxy).statistic)
    by_true = table.sort_values(["score_true", "case"])
    by_proxy = table.sort_values(["score_proxy", "case"])
    result = {"spearman_rho": rho, "mean_abs_score_error": float(table.abs_score_error.mean()),
              "max_abs_score_error": float(table.abs_score_error.max()),
              "true_best_case": int(by_true.iloc[0]["case"]), "proxy_best_case": int(by_proxy.iloc[0]["case"]),
              "max_rank_displacement": float((table.true_rank-table.proxy_rank).abs().max())}
    for k in (3, 10):
        result[f"top_{k}_overlap"] = len(set(by_true.head(k).case) & set(by_proxy.head(k).case))
    return table, result


def stream_diagnostics(true, pred, history_scale):
    late_errors = (pred[:, -8:]-true[:, -8:])/history_scale
    rt, rp = cumulative_rates(true), cumulative_rates(pred)
    rate_scale = float(rt[:, SCORE_MASK].max())
    if rate_scale <= 0:
        raise ValueError("Positive true validation rate required for normalization")
    endpoint = np.abs(pred[:, -1]-true[:, -1])/np.abs(true[:, -1])*100
    return {"late_nrmse_pooled": float(np.sqrt(np.mean(late_errors**2))),
            "late_nrmse_mean_case": float(np.sqrt(np.mean(late_errors**2, axis=1)).mean()),
            "end_pct_err_mean": float(endpoint.mean()), "end_pct_err_max": float(endpoint.max()),
            "rate_nrmse_mean_case": float(nrmse(rp, rt, rate_scale).mean()),
            "rate_nrmse_pooled": float(np.sqrt(np.mean(((rp-rt)[:, SCORE_MASK]/rate_scale)**2))),
            "rate_scale_true_validation_max": rate_scale}, rt, rp


def finish_figure(fig, filename, note):
    fig.text(0.02, 0.012, note, fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(C.FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def plot_curves_and_rates(cases, y, pred, history, selected):
    true, fitted, hist = split_y(y), split_y(pred), history_arrays(history)
    for kind, filename in [("cumulative", "proxy_val_curves.png"),
                           ("error", "proxy_val_curve_errors.png"), ("rate", "proxy_val_rates.png")]:
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        for r, (label, idx) in enumerate(selected):
            for j, s in enumerate(C.CUM_STREAMS):
                ax = axes[r, j]
                t, p = true[s][idx], fitted[s][idx]
                if kind == "rate":
                    t, p = cumulative_rates(t[None])[0], cumulative_rates(p[None])[0]
                if kind == "error":
                    ax.plot(C.HISTORY_DATES, (p-t)/hist[s][SCORE_MASK].max()*100, color=COLORS[j])
                    ax.axhline(0, color="black", linewidth=0.6)
                    ax.set_ylabel("Error / history maximum (%)")
                else:
                    ax.plot(C.HISTORY_DATES, t, color="#30343b", linewidth=2, label="Simulator")
                    ax.plot(C.HISTORY_DATES, p, color=COLORS[j], linestyle="--", linewidth=1.5, label="Proxy")
                    ax.set_ylabel(UNITS[j] + (" / day" if kind == "rate" else " cumulative"))
                    if r == 0 and j == 0:
                        ax.legend(fontsize=9)
                ax.set_title(f"{label}: Case {cases[idx]} — {LABELS[j]}", fontsize=11)
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.tick_params(axis="x", labelsize=9)
                ax.grid(alpha=0.2)
        fig.suptitle(f"Validation cases 71–85: {kind} comparison", fontsize=15)
        finish_figure(fig, filename, "Best / median / worst selected by mean oil, gas and water case NRMSE. Fixed GP trained on cases 1–70.")


def plot_ranking(table, rho):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(table.score_true, table.score_proxy, color=COLORS[0])
    low, high = min(table.score_true.min(), table.score_proxy.min()), max(table.score_true.max(), table.score_proxy.max())
    ax.plot([low, high], [low, high], color="gray", linestyle="--", label="Exact agreement")
    for i, row in enumerate(table.sort_values("score_true").itertuples()):
        dx = -9 if i % 2 == 0 else 9
        ax.annotate(str(row.case), (row.score_true, row.score_proxy), xytext=(dx, 6),
                    textcoords="offset points", ha="right" if dx < 0 else "left", fontsize=8)
    ax.set(xlabel="Simulator history-match score", ylabel="Proxy history-match score", title=f"Ranking agreement: Spearman = {rho:.4f}")
    ax.legend(fontsize=9)
    ordered = table.sort_values("score_true")
    axes[1].bar(np.arange(len(ordered)), ordered.signed_score_error, color=COLORS[2])
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_xticks(np.arange(len(ordered)), ordered.case)
    axes[1].set(xlabel="Validation case (best simulator match first)", ylabel="Proxy score minus simulator score", title="Score differences, not only rank correlation")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    finish_figure(fig, "proxy_ranking.png", "Both scores compare curves with RAMP history using unchanged T1 weights. This checks score/rank fidelity, not improvement over Case 6.")


def plot_late(cases, true, pred, hist):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, s in enumerate(C.CUM_STREAMS):
        late = np.sqrt(np.mean((pred[s][:, -8:]-true[s][:, -8:])**2, axis=1))/hist[s][SCORE_MASK].max()
        end = np.abs(pred[s][:, -1]-true[s][:, -1])/np.abs(true[s][:, -1])*100
        for r, values, target, title in [(0, late, 0.002, "Last 8 quarters: case NRMSE"), (1, end, 0.5, "January 2008: absolute error (%)")]:
            ax=axes[r, j]
            ax.bar(np.arange(len(cases)), values, color=COLORS[j])
            ax.axhline(target, color="gray", linestyle="--", label=f"Working target {target}")
            ax.set_xticks(np.arange(len(cases)), cases, rotation=60)
            ax.set_title(f"{LABELS[j]} — {title}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    finish_figure(fig, "proxy_val_late_time.png", "Per-case results shown to expose outliers; formal late-time check uses pooled RMSE and endpoint check uses mean percentage error.")


def plot_parameter_errors(case_table, train):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, param in zip(axes, C.PARAM_COLS):
        ax.scatter(case_table[param], case_table.mean_nrmse, color=COLORS[0])
        for i, row in enumerate(case_table.sort_values(param).itertuples()):
            dx, dy = (-5, -9) if i % 2 == 0 else (4, 5)
            ax.annotate(str(row.case), (getattr(row, param), row.mean_nrmse), xytext=(dx, dy),
                        textcoords="offset points", ha="right" if dx < 0 else "left", fontsize=7)
        ax.axvspan(train[param].min(), train[param].max(), color=COLORS[0], alpha=0.06)
        ax.margins(x=0.12, y=0.12)
        ax.set(xlabel=param, ylabel="Mean three-stream case NRMSE")
        ax.grid(alpha=0.2)
    finish_figure(fig, "proxy_error_vs_params.png", "Labels are case IDs; shaded x-range is training coverage. Only 15 validation cases: patterns are exploratory, not causal conclusions.")


def plot_uncertainty(coverage):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x=np.arange(3)
    for i, scope in enumerate(["all_scored", "positive_production"]):
        rows=coverage[(coverage.scope==scope)&(coverage.center=="raw")].set_index("stream")
        ax.bar(x+(i-.5)*.34, rows.loc[C.CUM_STREAMS, "coverage_2std"]*100, width=.34,
               label=scope.replace("_", " "))
    ax.axhline(95.45, color="gray", linestyle="--", label="Nominal Gaussian +/-2 std")
    ax.set_xticks(x, LABELS)
    ax.set(ylim=(0, 105), ylabel="Simulator points inside raw GP +/-2 std (%)", title="Validation uncertainty coverage — uncalibrated")
    ax.legend(fontsize=8, loc="lower right")
    finish_figure(fig, "proxy_val_uncertainty.png", "Approximate pointwise bands omit discarded-PC and parameter uncertainty. No band multiplier was fitted on validation.")


def main():
    train, val = validation_rows(read_master())
    history = read_history()
    hist = history_arrays(history)
    X, y = train[C.PARAM_COLS].to_numpy(float), train[y_columns()].to_numpy(float)
    XV, yv = val[C.PARAM_COLS].to_numpy(float), val[y_columns()].to_numpy(float)
    cases = val.case.to_numpy()
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Fixed-model validation: fit 1-70; evaluate 71-85; stop before test 86-100.", flush=True)
    # Only these predeclared comparison models are fitted; there is no search loop.
    configs = [(CV_CANDIDATE, GP_SETTINGS), ("dumb", dict(model="dumb")),
               ("ridge_d2_a0.0001", dict(model="ridge", degree=2, alpha=0.0001))]
    cv = pd.read_csv(C.OUTPUTS_DIR / "proxy_cv_extended_comparison.csv").set_index("candidate")
    reports, gp = [], None
    with threadpool_limits(limits=1):
        for name, settings in configs:
            print(f"Fitting {name} on 70 cases...", flush=True)
            model = ProxyPipeline(**settings)
            start = perf_counter()
            model.fit(X, y)
            fit_time = perf_counter()-start
            if settings["model"] == "gp":
                raw, std = model.predict_raw(XV, return_std=True)
                pred = model.correct_predictions(raw)
                gp = model
                gp_pred, gp_raw, gp_std = pred.copy(), raw.copy(), std.copy()
            else:
                raw = model.predict_raw(XV)
                pred = model.correct_predictions(raw)
            report, raw_report = proxy_report(yv, pred, history), proxy_report(yv, raw, history)
            for s in C.CUM_STREAMS:
                cv_score = float(cv.loc[name, f"{s}_nrmse_mean"])
                reports.append({"model": name, "stream": s, **report.loc[s].to_dict(),
                                "raw_nrmse": float(raw_report.loc[s, "nrmse_mean"]),
                                "cv_nrmse": cv_score, "validation_over_cv": float(report.loc[s, "nrmse_mean"] / cv_score),
                                "fit_seconds": fit_time})
    report = pd.DataFrame(reports)
    report.to_csv(C.OUTPUTS_DIR / "proxy_val_report.csv", index=False)
    chosen = report[report.model == CV_CANDIDATE].set_index("stream")
    true, predicted = split_y(yv), split_y(gp_pred)
    case_table = val[["case"]+C.PARAM_COLS].reset_index(drop=True).copy()
    for s in C.CUM_STREAMS:
        case_table[f"{s}_nrmse"] = nrmse(predicted[s], true[s], hist[s][SCORE_MASK].max())
    case_table["mean_nrmse"] = case_table[[f"{s}_nrmse" for s in C.CUM_STREAMS]].mean(axis=1)
    case_table["oil_water_nrmse"] = (case_table.oil_cum_nrmse+case_table.water_cum_nrmse)/2
    x01 = (XV-gp.xmin_)/gp.xrange_
    train01 = (X-gp.xmin_)/gp.xrange_
    case_table["outside_training_box"] = ((x01<0)|(x01>1)).any(axis=1)
    case_table["nearest_training_distance_scaled"] = np.sqrt(((x01[:,None,:]-train01[None,:,:])**2).sum(axis=2)).min(axis=1)
    order = np.argsort(case_table.mean_nrmse.to_numpy(), kind="stable")
    selected = [("Best", int(order[0])), ("Median", int(order[len(order)//2])), ("Worst", int(order[-1]))]
    case_table.to_csv(C.OUTPUTS_DIR / "proxy_val_case_errors.csv", index=False)
    rank_table, ranking = ranking_report(cases, score_vs_history(true, history).score.to_numpy(),
                                         score_vs_history(predicted, history).score.to_numpy())
    rank_table.to_csv(C.OUTPUTS_DIR / "proxy_val_ranking.csv", index=False)
    checks=[]
    def check(name, metric, value, threshold=None, direction="lt"):
        passed = None if threshold is None else bool(value < threshold if direction=="lt" else value <= threshold if direction=="le" else value >= threshold)
        checks.append({"check": name, "metric": metric, "value": value, "threshold": threshold,
                       "comparison": "report_only" if threshold is None else direction, "pass": passed})
    stream_rows=[]
    for s in C.CUM_STREAMS:
        d, _, _ = stream_diagnostics(true[s], predicted[s], hist[s][SCORE_MASK].max())
        stream_rows.append({"stream": s, **d})
        check("validation_accuracy", s, chosen.loc[s, "nrmse_mean"], .002)
        check("validation_vs_cv", s, chosen.loc[s, "validation_over_cv"], 2, "le")
        check("late_time_pooled_nrmse", s, d["late_nrmse_pooled"], .002)
        check("endpoint_mean_pct", s, d["end_pct_err_mean"], .5)
        check("endpoint_worst_pct", s, d["end_pct_err_max"])
        check("rate_mean_case_nrmse", s, d["rate_nrmse_mean_case"])
    for key, val_ in ranking.items():
        threshold = .95 if key=="spearman_rho" else .003 if key=="max_abs_score_error" else None
        check("ranking", key, val_, threshold, "ge" if key=="spearman_rho" else "lt")
    check("physics", "negative_values", int((gp_pred<0).sum()), 0, "le")
    check("physics", "decreasing_intervals", int((np.diff(gp_pred.reshape(-1,3,N_DATES),axis=2)<0).sum()), 0, "le")
    check("physics", "corrected_values", gp.last_n_fixed)
    check("physics", "max_correction_over_fit_scale", gp.last_max_correction_scaled)
    check("water_oil_error_ratio", "diagnostic_only", float(chosen.loc["water_cum", "nrmse_mean"]/chosen.loc["oil_cum", "nrmse_mean"]))
    coverage = pd.DataFrame(coverage_rows(yv, gp_raw, gp_pred, gp_std,
                                        {s: np.full(len(yv), st["scale"]) for s, st in gp.streams_.items()}))
    for r in coverage.itertuples():
        check("uncertainty_coverage", f"{r.stream}_{r.scope}_{r.center}", r.coverage_2std)
    coverage.to_csv(C.OUTPUTS_DIR / "proxy_val_uncertainty.csv", index=False)
    pd.DataFrame(stream_rows).to_csv(C.OUTPUTS_DIR / "proxy_val_late_rates.csv", index=False)
    parameter_rows=[]
    for i, param in enumerate(C.PARAM_COLS):
        parameter_rows.append({"parameter": param, "train_min": float(gp.xmin_[i]), "train_max": float(gp.xmax_[i]),
                               "validation_min": float(XV[:,i].min()), "validation_max": float(XV[:,i].max()),
                               "outside_training_range": int(((x01[:,i]<0)|(x01[:,i]>1)).sum()),
                               "error_spearman_exploratory": float(spearmanr(XV[:,i], case_table.mean_nrmse).statistic)})
    pd.DataFrame(parameter_rows).to_csv(C.OUTPUTS_DIR / "proxy_val_parameter_checks.csv", index=False)
    states = {s: {k:v for k,v in st.items() if k in {"scale", "n_components", "auto_components", "pca_error", "pca_tolerance_met", "fit_warnings", "gp_kernels", "gp_optimizer_audit"}} for s,st in gp.streams_.items()}
    selected_failed = sum(not o["selected_success"] for st in states.values() for o in st["gp_optimizer_audit"])
    check("optimizer", "selected_failed_component_fits", selected_failed, 0, "le")
    check_table=pd.DataFrame(checks)
    check_table.to_csv(C.OUTPUTS_DIR / "proxy_deep_checks.csv", index=False)
    audit={"created_utc": datetime.now(timezone.utc).isoformat(), "seed": C.SEED,
           "settings_fixed_before_validation": GP_SETTINGS, "train_cases": train.case.tolist(), "validation_cases": cases.tolist(),
           "test_evaluated": False, "automatic_tuning": False, "final_model_exported": False,
           "history_score_weights": C.SCORE_WEIGHTS, "streams": states,
           "train_X_y_sha256": hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(),
           "validation_X_y_sha256": hashlib.sha256(XV.tobytes()+yv.tobytes()).hexdigest(),
           "curve_examples": {label:int(cases[idx]) for label,idx in selected}, "ranking": ranking,
           "late_metric": "pooled RMSE across all 15 cases and final 8 dates, divided by history max; case-mean also reported",
           "rate_metric": "mean case NRMSE on scored dates; scale = max true validation rate; first date undefined; actual interval days",
           "uncertainty": "raw independent retained-PC approximation, uncalibrated; excludes discarded PCA and parameter uncertainty; corrected-center coverage reported separately",
           "versions": {p:version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]}}
    (C.OUTPUTS_DIR / "proxy_val_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    plot_curves_and_rates(cases, yv, gp_pred, history, selected)
    plot_ranking(rank_table, ranking["spearman_rho"])
    plot_late(cases, true, predicted, hist)
    plot_parameter_errors(case_table, train)
    plot_uncertainty(coverage)
    print("\nGP VALIDATION\n", chosen.to_string())
    print("\nRANKING", json.dumps(ranking))
    print("\nLATE/RATE\n",pd.DataFrame(stream_rows).to_string(index=False))
    print("\nWORST THREE\n",case_table.nlargest(3,"mean_nrmse").to_string(index=False))
    print("\nFAILED NUMERIC CHECKS\n",check_table.loc[check_table['pass'].eq(False)].to_string(index=False))
    print("STOP: final test cases 86-100 were not evaluated; no model settings changed.")


if __name__ == "__main__":
    main()
