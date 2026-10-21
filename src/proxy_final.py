"""Frozen T2 export and one-time test, deliberately separate from model selection.

python -m src.proxy_final --prepare  # fit 1-85, serialize, verify API, benchmark
python -m src.proxy_final --final    # predict 86-100 once, then report
python -m src.proxy_final --report-only  # regenerate reports from saved evaluation
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from threadpoolctl import threadpool_limits

from src import config as C
from src.load_data import read_master, read_history, y_columns
from src.metrics import SCORE_MASK, history_arrays, proxy_report, score_vs_history, split_y, nrmse
from src.proxy import ProxyPipeline, load_proxy, predict, param_bounds
from src.proxy_diagnostics import coverage_rows
from src.proxy_validation import validation_rows, ranking_report, stream_diagnostics, cumulative_rates


# Frozen before test access. Do not alter using test results.
FROZEN_SETTINGS = dict(model="gp", nu=2.5, pca_extra=2, pca_tolerance=0.0002,
                       max_components=10, gp_restarts=5, gp_length_upper=1e3,
                       gp_noise_lower=1e-10, gp_alpha=1e-10, gp_trace_optimizer=True)
MODEL = C.MODELS_DIR / "proxy.joblib"
META = C.MODELS_DIR / "proxy_meta.json"
STATE = C.OUTPUTS_DIR / "proxy_final_state.json"
CACHE = C.DATA_PROCESSED / "proxy_final_evaluation.npz"  # full curves remain local
REPORT = C.OUTPUTS_DIR / "proxy_test_report.csv"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(*arrays):
    return hashlib.sha256(b"".join(np.asarray(a, dtype=float).tobytes() for a in arrays)).hexdigest()


def source_hashes():
    return {name: digest_file(C.ROOT / "src" / name) for name in
            ["proxy.py", "proxy_final.py", "proxy_validation.py", "proxy_diagnostics.py", "config.py", "metrics.py", "load_data.py"]}


def final_fit_rows(master):
    train, validation = validation_rows(master)
    return pd.concat([train, validation]).sort_values("case")


def claim_test(path, payload):
    """Exclusive, durable marker before test prediction; a crash fails closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def prepare():
    if any(p.exists() for p in (MODEL, META, STATE, CACHE, REPORT)):
        raise RuntimeError("Final artifacts already exist; refusing to refit or overwrite them.")
    master = read_master()
    fit = final_fit_rows(master)
    X, y = fit[C.PARAM_COLS].to_numpy(float), fit[y_columns()].to_numpy(float)
    # T1 explicitly defines the search envelope using all 100 INPUTS, not targets.
    all_X = master[C.PARAM_COLS].to_numpy(float)
    if not np.isfinite(all_X).all():
        raise ValueError("Non-finite parameter envelope")
    bounds = np.column_stack([all_X.min(axis=0), all_X.max(axis=0)])
    model = ProxyPipeline(**FROZEN_SETTINGS)
    print("Frozen GP: fitting cases 1-85; test targets are not evaluated.", flush=True)
    start = perf_counter()
    with threadpool_limits(limits=1):
        model.fit(X, y)
    fit_seconds = perf_counter() - start
    model.search_bounds_ = bounds
    model.input_columns_ = list(C.PARAM_COLS)
    model.output_columns_ = y_columns()
    model.fit_case_ids_ = fit.case.tolist()
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL, compress=3)
    load_proxy(MODEL)
    # Serialization, column reordering, batching and sigma checks use fit inputs only.
    with threadpool_limits(limits=1):
        expected, expected_std = model.predict(X[:5], return_std=True)
    actual, actual_std = predict(fit.loc[:, C.PARAM_COLS].iloc[:5, ::-1], return_std=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(actual_std, expected_std, rtol=1e-12, atol=1e-8)
    rng = np.random.default_rng(C.SEED)
    random_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(50000, 4))
    print("Timing 50,000 full-curve predictions through the T3 API...", flush=True)
    start = perf_counter()
    batch = predict(random_X)
    seconds = perf_counter() - start
    if batch.shape != (50000, 123) or not np.isfinite(batch).all():
        raise RuntimeError("Invalid benchmark predictions")
    if (batch < 0).any() or (np.diff(batch.reshape(-1, 3, 41), axis=2) < 0).any():
        raise RuntimeError("Physical corrections failed")
    probe = np.array([0, 2047, 2048, 49999])
    with threadpool_limits(limits=1):
        np.testing.assert_allclose(batch[probe], model.predict(random_X[probe]), rtol=1e-10, atol=1e-6)
    states = {s: {k: v for k, v in st.items() if k in
              {"scale", "n_components", "auto_components", "pca_error", "pca_tolerance_met",
               "fit_warnings", "gp_kernels", "gp_optimizer_audit"}} for s, st in model.streams_.items()}
    failed = sum(not r["selected_success"] for st in states.values() for r in st["gp_optimizer_audit"])
    metadata = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "settings": FROZEN_SETTINGS,
        "seed": C.SEED, "fit_case_ids": model.fit_case_ids_, "test_case_ids": list(C.SPLIT["test"]),
        "test_evaluated": False, "gas_from_oil": False, "input_columns": C.PARAM_COLS,
        "output_columns": y_columns(), "dates": C.HISTORY_DATES.strftime("%Y-%m-%d").tolist(),
        "units": {"oil_cum": "STB", "gas_cum": "MSCF", "water_cum": "STB"},
        "search_bounds": param_bounds().to_dict(orient="index"),
        "bounds_source": "min/max of all 100 parameter inputs, per T1 contract; no test targets used",
        "fit_bounds": {p: {"min": float(model.xmin_[i]), "max": float(model.xmax_[i])} for i, p in enumerate(C.PARAM_COLS)},
        "fit_X_y_sha256": array_digest(X, y), "all_parameter_inputs_sha256": array_digest(all_X),
        "model_sha256": digest_file(MODEL), "source_sha256": source_hashes(),
        "history_sha256": array_digest(read_history()[C.CUM_STREAMS].to_numpy(float)),
        "versions": {p: version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "joblib"]},
        "fit_seconds": fit_seconds, "streams": states, "selected_optimizer_failures": failed,
        "benchmark": {"n_predictions": 50000, "seconds": seconds, "target_seconds": 60,
                      "pass": seconds < 60, "includes_loading": False, "return_std": False,
                      "batch_size": 2048, "blas_threads": 1, "seed": C.SEED,
                      "physics_violations": 0, "serialization_and_batch_checks": "passed"},
        "history_score_weights": C.SCORE_WEIGHTS,
        "score_tie_band": {"value": 0.0003, "status": "empirical caution heuristic, not guaranteed error bound"},
        "uncertainty": "uncalibrated retained-PC pointwise GP std; omits discarded-PC, model-selection and parameter uncertainty",
        "forecast_limit": "1998-01-01 to 2008-01-01 only; no learned 2008-2028 extrapolation",
    }
    write_json(META, metadata)
    print(f"Prepared model: {fit_seconds:.2f}s fit; 50,000 predictions {seconds:.2f}s; selected optimizer failures {failed}.")
    print("Test remains unevaluated. Next separate command: python -m src.proxy_final --final")


def evaluate_once():
    if any(p.exists() for p in (STATE, CACHE, REPORT)):
        raise RuntimeError("Test already started/evaluated. Use --report-only with the local cache; never repeat prediction.")
    meta = json.loads(META.read_text(encoding="utf-8"))
    if meta["settings"] != FROZEN_SETTINGS or meta["model_sha256"] != digest_file(MODEL) or meta["source_sha256"] != source_hashes():
        raise RuntimeError("Frozen config, source or artifact changed since preparation")
    master, history = read_master(), read_history()
    fit = final_fit_rows(master)
    if meta["fit_X_y_sha256"] != array_digest(fit[C.PARAM_COLS], fit[y_columns()]):
        raise RuntimeError("Fit data changed since preparation")
    if meta["all_parameter_inputs_sha256"] != array_digest(master[C.PARAM_COLS]):
        raise RuntimeError("Parameter inputs changed since preparation")
    if meta["history_sha256"] != array_digest(history[C.CUM_STREAMS]):
        raise RuntimeError("Reference history changed since preparation")
    model = load_proxy(MODEL)
    marker = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
              "test_case_ids": list(C.SPLIT["test"]), "model_sha256": meta["model_sha256"],
              "automatic_tuning": False, "prediction_batches": 1}
    claim_test(STATE, marker)
    test = master.loc[master.split == "test"].sort_values("case")
    XT, yt = test[C.PARAM_COLS].to_numpy(float), test[y_columns()].to_numpy(float)
    if not np.isfinite(yt).all():
        raise ValueError("Invalid test targets: evaluation stopped; do not silently retry")
    with threadpool_limits(limits=1):
        raw, std = model.predict_raw(XT, return_std=True)  # the sole test prediction
        predicted = model.correct_predictions(raw)
    corrections = {"count": model.last_n_fixed, "max_over_fit_scale": model.last_max_correction_scaled}
    np.savez_compressed(CACHE, case=test.case.to_numpy(), X=XT, true=yt, raw=raw, pred=predicted, std=std,
                        history=history_arrays(history)["oil_cum"],
                        history_all=np.column_stack([history_arrays(history)[s] for s in C.CUM_STREAMS]),
                        scales=np.array([model.streams_[s]["scale"] for s in C.CUM_STREAMS]))
    marker.update(status="predicted", cache_sha256=digest_file(CACHE), corrections=corrections)
    write_json(STATE, marker)
    render_report()


def render_report():
    """Read only frozen predictions; never load/refit a model or predict again."""
    marker = json.loads(STATE.read_text(encoding="utf-8"))
    if marker.get("cache_sha256") != digest_file(CACHE):
        raise RuntimeError("Evaluation cache is missing or changed")
    meta = json.loads(META.read_text(encoding="utf-8"))
    with np.load(CACHE, allow_pickle=False) as cached:
        d = {k: cached[k].copy() for k in cached.files}
    history = pd.DataFrame(d["history_all"], columns=C.CUM_STREAMS)
    history.insert(0, "date", C.HISTORY_DATES)
    true, pred, hist = split_y(d["true"]), split_y(d["pred"]), history_arrays(history)
    report = proxy_report(d["true"], d["pred"], history)
    val = pd.read_csv(C.OUTPUTS_DIR / "proxy_val_report.csv")
    val = val[val.model == "gp_nu2.5_extra2"].set_index("stream")
    cv = pd.read_csv(C.OUTPUTS_DIR / "proxy_cv_extended_comparison.csv").set_index("candidate").loc["gp_nu2.5_extra2"]
    rank_table, rank = ranking_report(d["case"], score_vs_history(true, history).score.to_numpy(),
                                     score_vs_history(pred, history).score.to_numpy())
    rank_table.to_csv(C.OUTPUTS_DIR / "proxy_test_ranking.csv", index=False)
    rows, stream_rows = [], []
    def check(group, metric, value, threshold=None, op="lt"):
        passed = None if threshold is None else bool(value < threshold if op == "lt" else value >= threshold if op == "ge" else value <= threshold)
        rows.append(dict(group=group, metric=metric, value=value, threshold=threshold,
                         comparison="report_only" if threshold is None else op, passed=passed))
    cases = pd.DataFrame(d["X"], columns=C.PARAM_COLS)
    cases.insert(0, "case", d["case"])
    for s in C.CUM_STREAMS:
        scale = hist[s][SCORE_MASK].max()
        diagnostics, rt, rp = stream_diagnostics(true[s], pred[s], scale)
        diagnostics["rate_scale_true_test_max"] = diagnostics.pop("rate_scale_true_validation_max")
        # Signed slope errors quantify the handover caution without assuming bias.
        late_bias = (rp[:, -8:] - rt[:, -8:]).mean(axis=1)
        end_rate_pct = (rp[:, -1] - rt[:, -1]) / rt[:, -1] * 100
        diagnostics.update(late_rate_bias_mean=float(late_bias.mean()),
                           late_rate_bias_max_abs_case=float(np.abs(late_bias).max()),
                           final_rate_signed_pct_mean=float(end_rate_pct.mean()),
                           final_rate_abs_pct_max=float(np.abs(end_rate_pct).max()))
        ratio = float(report.loc[s, "nrmse_mean"] / val.loc[s, "nrmse_mean"])
        stream_rows.append({"stream": s, **report.loc[s].to_dict(),
                            "cv_nrmse": float(cv[f"{s}_nrmse_mean"]),
                            "validation_nrmse": float(val.loc[s, "nrmse_mean"]),
                            "test_over_validation": ratio, **diagnostics})
        cases[f"{s}_nrmse"] = nrmse(pred[s], true[s], scale)
        cases[f"{s}_final_rate_signed_pct"] = end_rate_pct
        check(s, "test_nrmse", float(report.loc[s, "nrmse_mean"]), .002)
        check(s, "test_over_validation", ratio, 2, "le")
        check(s, "late_nrmse_pooled", diagnostics["late_nrmse_pooled"], .002)
        check(s, "endpoint_mean_pct", diagnostics["end_pct_err_mean"], .5)
        check(s, "endpoint_max_pct", diagnostics["end_pct_err_max"])
        check(s, "rate_nrmse_mean_case", diagnostics["rate_nrmse_mean_case"])
        check(s, "final_rate_abs_pct_max", diagnostics["final_rate_abs_pct_max"])
    for key, value in rank.items():
        check("ranking", key, value, .95 if key == "spearman_rho" else .003 if key == "max_abs_score_error" else None,
              "ge" if key == "spearman_rho" else "lt")
    check("physics", "negative_values", int((d["pred"] < 0).sum()), 0, "le")
    check("physics", "decreasing_intervals", int((np.diff(d["pred"].reshape(-1, 3, 41), axis=2) < 0).sum()), 0, "le")
    check("optimizer", "selected_failed_components", meta["selected_optimizer_failures"], 0, "le")
    check("speed", "seconds_for_50000", meta["benchmark"]["seconds"], 60)
    cases["mean_nrmse"] = cases[[f"{s}_nrmse" for s in C.CUM_STREAMS]].mean(axis=1)
    fit_bounds = meta["fit_bounds"]
    cases["outside_fit_box"] = np.column_stack([(d["X"][:, i] < fit_bounds[p]["min"]) |
                                                 (d["X"][:, i] > fit_bounds[p]["max"]) for i, p in enumerate(C.PARAM_COLS)]).any(axis=1)
    cases.to_csv(C.OUTPUTS_DIR / "proxy_test_case_errors.csv", index=False)
    coverage = pd.DataFrame(coverage_rows(d["true"], d["raw"], d["pred"], d["std"],
                                         {s: np.full(len(cases), d["scales"][i]) for i, s in enumerate(C.CUM_STREAMS)}))
    coverage.to_csv(C.OUTPUTS_DIR / "proxy_test_uncertainty.csv", index=False)
    for r in coverage.itertuples():
        check("uncertainty", f"{r.stream}_{r.scope}_{r.center}", r.coverage_2std)
    checks = pd.DataFrame(rows)
    checks.to_csv(REPORT, index=False)
    pd.DataFrame(stream_rows).to_csv(C.OUTPUTS_DIR / "proxy_test_streams.csv", index=False)
    meta.update(test_evaluated=True, test_ranking=rank, test_streams=stream_rows,
                test_numeric_gates_passed=bool(not checks.passed.eq(False).any()),
                test_uncertainty=coverage.to_dict(orient="records"))
    write_json(META, meta)
    marker.update(status="completed", report_sha256=digest_file(REPORT),
                  test_numeric_gates_passed=meta["test_numeric_gates_passed"])
    write_json(STATE, marker)
    make_charts(d, cases, rank_table, rank)
    print("\nFROZEN TEST RESULTS\n" + pd.DataFrame(stream_rows).to_string(index=False))
    print("\nRANKING " + json.dumps(rank))
    print("\nFAILED GATES\n" + checks.loc[checks.passed.eq(False)].to_string(index=False))
    print("No tuning performed. Future reports must use --report-only, never rerun the test.")


def make_charts(d, cases, ranks, rank):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = np.argsort(cases.mean_nrmse.to_numpy(), kind="stable")
    examples = [("Best", order[0]), ("Median", order[len(order)//2]), ("Worst", order[-1])]
    for rate in [False, True]:
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        for row, (label, idx) in enumerate(examples):
            for j, s in enumerate(C.CUM_STREAMS):
                ax = axes[row, j]
                t, p = split_y(d["true"])[s][idx:idx+1], split_y(d["pred"])[s][idx:idx+1]
                if rate:
                    t, p = cumulative_rates(t), cumulative_rates(p)
                ax.plot(C.HISTORY_DATES, t[0], color="#30343b", label="Simulator", linewidth=2)
                ax.plot(C.HISTORY_DATES, p[0], "--", color=["#2471a3", "#d9822b", "#28946c"][j], label="Proxy")
                ax.set_title(f"{label}: Case {d['case'][idx]} - {s.split('_')[0].title()}")
                ax.set_ylabel(("MSCF" if j == 1 else "STB") + (" / day" if rate else " cumulative"))
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.grid(alpha=.2)
                if row == 0 and j == 0:
                    ax.legend()
        fig.suptitle("One-time test: cases 86-100 | frozen GP fitted on 1-85", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(C.FIGURES_DIR / ("proxy_test_rates.png" if rate else "proxy_test_curves.png"), dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(ranks.score_true, ranks.score_proxy, color="#2471a3")
    lim = [0, max(ranks.score_true.max(), ranks.score_proxy.max()) * 1.04]
    axes[0].plot(lim, lim, "--", color="gray")
    axes[0].set(xlabel="Simulator history-match score", ylabel="Proxy history-match score",
                title=f"Test ranking: Spearman = {rank['spearman_rho']:.4f}")
    ordered = ranks.sort_values("score_true")
    axes[1].bar(np.arange(len(ranks)), ordered.signed_score_error, color="#2471a3")
    axes[1].set_xticks(np.arange(len(ranks)), ordered.case, rotation=50)
    axes[1].axhline(.0003, linestyle="--", color="gray", label="Empirical caution band")
    axes[1].axhline(-.0003, linestyle="--", color="gray")
    axes[1].set(xlabel="Case (best simulator match first)", ylabel="Proxy score minus simulator score", title="Score error; close ranks remain uncertain")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "proxy_test_ranking.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--final", action="store_true")
    action.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.final:
        evaluate_once()
    else:
        render_report()


if __name__ == "__main__":
    main()
