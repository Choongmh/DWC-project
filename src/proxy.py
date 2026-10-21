"""T2 development: fold-local preprocessing and model comparisons.

Run: python -m src.proxy
Extended comparison: python -m src.proxy --suite extended
CV commands use cases 1-70 only. Final export: python -m src.proxy_final --final.
Reference history supplies the shared metric's fixed scales, never training y.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

from src import config as C
from src.load_data import read_history, read_master, y_columns
from src.metrics import N_DATES, SCORE_MASK, nrmse, proxy_report, split_y


_CACHED_PROXY = None


def load_proxy(path=C.MODELS_DIR / "proxy.joblib"):
    """Load a trusted joblib artifact and cache it for the module-level API.

    Joblib uses pickle: only load artifacts from a trusted source. The exported
    model carries its schema and bounds, so raw competition files are not needed.
    """
    import joblib
    global _CACHED_PROXY
    model = joblib.load(Path(path))
    if not isinstance(model, ProxyPipeline) or not hasattr(model, "streams_"):
        raise ValueError("Expected a fitted ProxyPipeline artifact")
    if getattr(model, "input_columns_", None) != C.PARAM_COLS or getattr(model, "output_columns_", None) != y_columns():
        raise ValueError("Saved model schema does not match this code")
    bounds = np.asarray(getattr(model, "search_bounds_", None), dtype=float)
    if bounds.shape != (4, 2) or not np.isfinite(bounds).all() or (bounds[:, 0] > bounds[:, 1]).any():
        raise ValueError("Saved model has invalid search bounds")
    _CACHED_PROXY = model
    return model


def param_bounds():
    """Return the declared 100-case input envelope, columns min/max, in X order.

    These input-only ranges follow T1's search contract. Scaling/PCA/regressors
    use fit cases 1-85 only. Being inside this box does not guarantee accuracy.
    """
    model = _CACHED_PROXY if _CACHED_PROXY is not None else load_proxy()
    return pd.DataFrame(model.search_bounds_.copy(), index=C.PARAM_COLS, columns=["min", "max"])


def predict(X, return_std=False):
    """Raw-unit (n,4) inputs -> corrected cumulative (n,123), optionally std.

    DataFrames are reordered by C.PARAM_COLS; arrays must already follow that
    order. Std is an uncalibrated pointwise GP approximation before corrections.
    Out-of-envelope inputs warn and are not clipped. Batches bound memory use.
    """
    X = _inputs(X)
    model = _CACHED_PROXY if _CACHED_PROXY is not None else load_proxy()
    bounds = model.search_bounds_
    outside = ((X < bounds[:, 0]) | (X > bounds[:, 1])).any(axis=1)
    if outside.any():
        warnings.warn(f"{outside.sum()} input rows outside the declared parameter bounds; "
                      "these extrapolated predictions are unsupported.", UserWarning, stacklevel=2)
    means, stds = [], []
    with threadpool_limits(limits=1):
        for start in range(0, len(X), 2048):
            result = model.predict(X[start:start+2048], return_std=return_std)
            if return_std:
                mean, std = result
                means.append(mean)
                stds.append(std)
            else:
                means.append(result)
    mean = np.concatenate(means)
    return (mean, np.concatenate(stds)) if return_std else mean


def _inputs(X):
    if isinstance(X, pd.DataFrame):
        if len(X.columns) != len(C.PARAM_COLS) or set(X.columns) != set(C.PARAM_COLS):
            raise ValueError(f"Expected exactly the input columns {C.PARAM_COLS}")
        X = X[C.PARAM_COLS]
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 4 or len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a finite, nonempty (n, 4) array")
    return X


class GPOptimizerTrace:
    """Record each L-BFGS-B attempt using the same settings as sklearn's default."""
    def __init__(self):
        self.runs = []

    def __call__(self, objective, initial_theta, bounds):
        result = minimize(objective, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
        self.runs.append({"success": bool(result.success), "status": int(result.status),
                          "message": str(result.message), "iterations": int(result.nit),
                          "objective": float(result.fun), "theta": result.x.tolist()})
        if not result.success:
            warnings.warn(f"GP optimizer attempt {len(self.runs)-1}: {result.message} "
                          f"(status={result.status}, iterations={result.nit})", ConvergenceWarning)
        return result.x, result.fun

    def report(self):
        selected = int(np.argmin([r["objective"] for r in self.runs]))
        return {"selected_run": selected, "selected_success": self.runs[selected]["success"],
                "runs": self.runs}


class ProxyPipeline:
    """Fit scaling, per-stream PCA and regressors on only the supplied fit rows.

    All three streams are modelled independently. Predictions
    are in physical units and y_columns() order. The dumb model returns the mean
    training curve. Call predict_raw() to inspect results before physical fixes.
    """
    def __init__(self, model="ridge", degree=1, alpha=0.01,
                 pca_tolerance=0.0002, max_components=10, pca_extra=0,
                 nu=1.5, gp_restarts=5, max_depth=2, learning_rate=0.05,
                 n_estimators=500, gp_length_upper=1e3, gp_noise_lower=1e-10,
                 gp_alpha=1e-10, gp_trace_optimizer=False):
        if model not in {"dumb", "ridge", "gp", "xgb"}:
            raise ValueError("Supported models: dumb, ridge, gp, xgb")
        if degree not in {1, 2} or alpha <= 0 or pca_tolerance <= 0 or max_components < 1:
            raise ValueError("Invalid Ridge/PCA settings")
        self.model = model
        self.degree = degree
        self.alpha = alpha
        self.pca_tolerance = pca_tolerance
        self.max_components = max_components
        if pca_extra not in {0, 2} or nu not in {1.5, 2.5} or gp_restarts < 0:
            raise ValueError("Invalid PCA/GP settings")
        if max_depth < 1 or learning_rate <= 0 or n_estimators < 1:
            raise ValueError("Invalid XGBoost settings")
        self.pca_extra = pca_extra
        self.nu = nu
        self.gp_restarts = gp_restarts
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        if not np.isfinite([gp_length_upper, gp_noise_lower, gp_alpha]).all():
            raise ValueError("GP bounds and alpha must be finite")
        if gp_length_upper <= 1e-2 or not 0 < gp_noise_lower <= 1e-6 or gp_alpha <= 0:
            raise ValueError("Invalid GP bounds or alpha")
        self.gp_length_upper = gp_length_upper
        self.gp_noise_lower = gp_noise_lower
        self.gp_alpha = gp_alpha
        self.gp_trace_optimizer = gp_trace_optimizer

    def _fit_regressor(self, features, scores):
        if self.model == "ridge":
            return Ridge(alpha=self.alpha, solver="svd").fit(features, scores)
        regressors = []
        for j in range(scores.shape[1]):
            if self.model == "gp":
                kernel = ConstantKernel(1.0) * Matern(
                    length_scale=np.ones(4), length_scale_bounds=(1e-2, self.gp_length_upper), nu=self.nu
                ) + WhiteKernel(1e-6, (self.gp_noise_lower, 1e-2))
                regressor = GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True,
                    n_restarts_optimizer=self.gp_restarts, random_state=C.SEED,
                    alpha=self.gp_alpha,
                    optimizer=GPOptimizerTrace() if self.gp_trace_optimizer else "fmin_l_bfgs_b",
                )
            else:
                from xgboost import XGBRegressor
                regressor = XGBRegressor(
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    learning_rate=self.learning_rate, subsample=1.0,
                    objective="reg:squarederror", tree_method="hist", n_jobs=1,
                    random_state=C.SEED,
                )
            regressor.fit(features, scores[:, j])
            regressors.append(regressor)
        return regressors

    def fit(self, X, y):
        X = _inputs(X)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X), len(y_columns())) or not np.isfinite(y).all():
            raise ValueError("y must be finite with shape (n, 123), in y_columns() order")
        if len(X) < 2:
            raise ValueError("At least two training cases are needed")
        if (y < 0).any() or (np.diff(y.reshape(-1, 3, N_DATES), axis=2) < -1e-6).any():
            raise ValueError("Training cumulative curves must be nonnegative and nondecreasing")
        self.xmin_ = X.min(axis=0)
        self.xmax_ = X.max(axis=0)
        self.xrange_ = np.where(self.xmax_ > self.xmin_, self.xmax_ - self.xmin_, 1.0)
        self.mean_y_ = y.mean(axis=0)
        self.streams_ = {}
        features = (X - self.xmin_) / self.xrange_
        if self.model == "ridge":
            self.poly_ = PolynomialFeatures(degree=self.degree, include_bias=False)
            features = self.poly_.fit_transform((X - self.xmin_) / self.xrange_)
        for i, stream in enumerate(C.CUM_STREAMS):
            block = y[:, i*N_DATES:(i+1)*N_DATES]
            scale = float(block.max()) or 1.0
            # Learn only the initial all-zero period from this fold's training rows.
            zero_prefix = np.logical_and.accumulate(np.all(block == 0, axis=0))
            state = {"scale": scale, "zero_prefix": zero_prefix,
                     "n_components": 0, "pca_error": 0.0, "pca_tolerance_met": True}
            if self.model != "dumb":
                scaled = block / scale
                if np.all(scaled == scaled[0]):
                    state["constant"] = scaled[0].copy()
                else:
                    cap = min(self.max_components, len(X) - 1, N_DATES)
                    pca = PCA(n_components=cap, svd_solver="full")
                    scores = pca.fit_transform(scaled)
                    for k in range(1, cap+1):
                        reconstructed = scores[:, :k] @ pca.components_[:k] + pca.mean_
                        error = float(np.sqrt(np.mean((reconstructed[:, SCORE_MASK] - scaled[:, SCORE_MASK])**2)))
                        if error < self.pca_tolerance:
                            break
                    auto_k = k
                    k = min(auto_k + self.pca_extra, cap)
                    reconstructed = scores[:, :k] @ pca.components_[:k] + pca.mean_
                    error = float(np.sqrt(np.mean((reconstructed[:, SCORE_MASK] - scaled[:, SCORE_MASK])**2)))
                    # Warnings are recorded per stream instead of lost in console noise.
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        regressor = self._fit_regressor(features, scores[:, :k])
                    warning_counts = dict(Counter(str(w.message) for w in caught))
                    state.update(pca=pca, regressor=regressor, n_components=k,
                                 auto_components=auto_k, fit_warnings=warning_counts,
                                 pca_error=error, pca_tolerance_met=error < self.pca_tolerance)
                    if self.model == "gp":
                        state["gp_kernels"] = [str(r.kernel_) for r in regressor]
                        if self.gp_trace_optimizer:
                            state["gp_optimizer_audit"] = [r.optimizer.report() for r in regressor]
            self.streams_[stream] = state
        return self

    def predict_raw(self, X, return_std=False):
        """Return raw mean and optional approximate GP std in physical units.

        Std propagates independent retained-PC GP variances (including WhiteKernel
        noise), excluding discarded-PC error, kernel estimation and physical fixes.
        It is pointwise, not a simultaneous uncertainty band for a whole curve.
        A stream constant in the fit rows has zero modelled variance by convention.
        """
        if return_std and self.model != "gp":
            raise NotImplementedError("Uncertainty is currently supported only for GP")
        if not hasattr(self, "streams_"):
            raise ValueError("Fit the pipeline before predicting")
        X = _inputs(X)
        if self.model == "dumb":
            return np.tile(self.mean_y_, (len(X), 1))
        features = (X - self.xmin_) / self.xrange_
        if self.model == "ridge":
            features = self.poly_.transform(features)
        blocks = []
        std_blocks = []
        for stream in C.CUM_STREAMS:
            s = self.streams_[stream]
            if "constant" in s:
                scaled = np.tile(s["constant"], (len(X), 1))
                scaled_std = np.zeros_like(scaled)
            else:
                if self.model == "ridge":
                    scores = np.asarray(s["regressor"].predict(features)).reshape(len(X), s["n_components"])
                elif return_std:
                    results = [r.predict(features, return_std=True) for r in s["regressor"]]
                    scores = np.column_stack([result[0] for result in results])
                    score_std = np.column_stack([result[1] for result in results])
                    # sklearn already undoes its internal normalize_y scaling.
                    variance = score_std**2 @ s["pca"].components_[:s["n_components"]]**2
                    scaled_std = np.sqrt(np.maximum(variance, 0))
                else:
                    scores = np.column_stack([r.predict(features) for r in s["regressor"]])
                scaled = scores @ s["pca"].components_[:s["n_components"]] + s["pca"].mean_
            blocks.append(scaled * s["scale"])
            if return_std:
                std_blocks.append(scaled_std * s["scale"])
        mean = np.concatenate(blocks, axis=1)
        return (mean, np.concatenate(std_blocks, axis=1)) if return_std else mean

    def correct_predictions(self, raw):
        """Apply the declared corrections and keep their size visible in reports."""
        raw = np.asarray(raw, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 123 or not np.isfinite(raw).all():
            raise ValueError("Raw predictions must be a finite (n, 123) array")
        fixed = raw.copy().reshape(-1, 3, N_DATES)
        for i, stream in enumerate(C.CUM_STREAMS):
            fixed[:, i, self.streams_[stream]["zero_prefix"]] = 0
            fixed[:, i, :] = np.maximum.accumulate(np.maximum(fixed[:, i, :], 0), axis=1)
        fixed = fixed.reshape(-1, 123)
        self.last_n_fixed = int(np.count_nonzero(fixed != raw))
        self.last_max_correction_scaled = max(
            float(np.max(np.abs(fixed[:, i*N_DATES:(i+1)*N_DATES] - raw[:, i*N_DATES:(i+1)*N_DATES]))) / self.streams_[s]["scale"]
            for i, s in enumerate(C.CUM_STREAMS)
        )
        return fixed

    def predict(self, X, return_std=False):
        """Corrected mean; optional std belongs to the uncorrected GP approximation.

        Use predict_raw(..., return_std=True) to assess raw GP interval coverage.
        The same std around the corrected mean is only a separate diagnostic.
        """
        if return_std:
            raw, std = self.predict_raw(X, return_std=True)
            return self.correct_predictions(raw), std
        return self.correct_predictions(self.predict_raw(X))


def training_rows(master):
    """Check IDs/split labels, then return ONLY train rows for CV."""
    if master.shape != (100, 129) or not master.case.is_unique:
        raise ValueError("Expected 100 unique cases and 129 columns")
    for name, ids in C.SPLIT.items():
        if set(master.loc[master.split == name, "case"]) != set(ids):
            raise ValueError(f"Unexpected case IDs for {name}")
    train = master.loc[master.split == "train"].sort_values("case").copy()
    if not np.isfinite(train[C.PARAM_COLS + y_columns()].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in training data")
    return train


def cv_splits(case_ids):
    if len(case_ids) != 70 or set(case_ids) != set(C.SPLIT["train"]):
        raise ValueError("CV requires exactly training cases 1-70")
    return list(KFold(n_splits=5, shuffle=True, random_state=C.SEED).split(case_ids))


def candidates(suite="baseline"):
    if suite not in {"baseline", "extended"}:
        raise ValueError("Unknown comparison suite")
    specs = [{"candidate": "dumb", "model": "dumb", "degree": 1, "alpha": 0.01}] + [
        {"candidate": f"ridge_d{degree}_a{alpha:g}", "model": "ridge", "degree": degree, "alpha": alpha}
        for degree in (1, 2) for alpha in (0.0001, 0.01, 1.0)
    ]
    if suite == "extended":
        specs += [{"candidate": f"gp_nu{nu:g}_extra{extra}", "model": "gp",
                   "nu": nu, "pca_extra": extra, "gp_restarts": 5}
                  for nu in (1.5, 2.5) for extra in (0, 2)]
        specs += [{"candidate": f"xgb_depth{depth}_lr{lr:g}", "model": "xgb",
                   "max_depth": depth, "learning_rate": lr, "n_estimators": 500}
                  for depth in (2, 3) for lr in (0.05, 0.1)]
    return specs


def select_candidate(summary):
    """Choose a family winner, preferring simplicity within 10% of the minimum."""
    if not np.isfinite(summary.selection_metric).all() or (summary.selection_metric < 0).any():
        raise ValueError("Candidate scores must be finite and nonnegative")
    families = summary.sort_values("selection_metric", kind="stable").drop_duplicates("model")
    eligible = families.loc[families.selection_metric <= 1.1 * families.selection_metric.min()].copy()
    eligible["complexity"] = eligible.model.map({"dumb": 0, "ridge": 1, "xgb": 2, "gp": 3})
    return eligible.sort_values(["complexity", "selection_metric"], kind="stable").iloc[0]


def run_cv(train, history, suite="baseline"):
    X = train[C.PARAM_COLS].to_numpy(dtype=float)
    y = train[y_columns()].to_numpy(dtype=float)
    ids = train.case.to_numpy()
    folds = cv_splits(ids)
    fold_rows, audit_rows, case_rows, summary_rows = [], [], [], []
    for spec in candidates(suite):
        candidate = spec["candidate"]
        oof = np.full(y.shape, np.nan)
        candidate_folds = []
        for fold, (fit_idx, held_idx) in enumerate(folds, 1):
            model = ProxyPipeline(**{k: v for k, v in spec.items() if k != "candidate"})
            print(f"Fitting {candidate}, fold {fold}/5...", flush=True)
            start = perf_counter()
            model.fit(X[fit_idx], y[fit_idx])
            fit_seconds = perf_counter() - start
            start = perf_counter()
            raw = model.predict_raw(X[held_idx])
            pred = model.correct_predictions(raw)
            predict_seconds = perf_counter() - start
            oof[held_idx] = pred
            report = proxy_report(y[held_idx], pred, history)
            raw_report = proxy_report(y[held_idx], raw, history)
            row = {**spec, "fold": fold, "n_fit": len(fit_idx), "n_heldout": len(held_idx),
                   "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                   "n_values_corrected": model.last_n_fixed,
                   "max_correction_scaled": model.last_max_correction_scaled,
                   "fit_warning_count": sum(sum(s.get("fit_warnings", {}).values()) for s in model.streams_.values())}
            for s in C.CUM_STREAMS:
                row[f"{s}_nrmse"] = report.loc[s, "nrmse_mean"]
                row[f"{s}_raw_nrmse"] = raw_report.loc[s, "nrmse_mean"]
                row[f"{s}_end_pct_err"] = report.loc[s, "end_pct_err"]
            # T2 selection follows the spec: oil/water equally weighted.
            # Gas is still independently fitted and reported; T1 weights stay intact.
            row["selection_metric"] = (row["oil_cum_nrmse"] + row["water_cum_nrmse"]) / 2
            candidate_folds.append(row)
            fold_rows.append(row)
            audit_rows.append({"candidate": candidate, "fold": fold,
                               "fit_case_ids": ids[fit_idx].tolist(), "heldout_case_ids": ids[held_idx].tolist(),
                               "xmin": model.xmin_.tolist(), "xmax": model.xmax_.tolist(),
                               "streams": {s: {k: v for k, v in st.items() if k in
                                   {"scale", "auto_components", "n_components", "pca_error", "pca_tolerance_met", "fit_warnings", "gp_kernels"}}
                                   for s, st in model.streams_.items()}})
            t, p = split_y(y[held_idx]), split_y(pred)
            per_case = {s: nrmse(p[s], t[s], history[s].to_numpy()[SCORE_MASK].max()) for s in C.CUM_STREAMS}
            for j, case in enumerate(ids[held_idx]):
                case_rows.append({"candidate": candidate, "case": int(case), "fold": fold,
                                  **{f"{s}_nrmse": float(per_case[s][j]) for s in C.CUM_STREAMS}})
        if not np.isfinite(oof).all():
            raise RuntimeError("Missing or invalid out-of-fold predictions")
        f = pd.DataFrame(candidate_folds)
        combined = proxy_report(y, oof, history)
        summary = {**spec, "selection_metric": float(f.selection_metric.mean()),
                   "selection_std": float(f.selection_metric.std(ddof=1)),
                   "fit_seconds": float(f.fit_seconds.sum()),
                   "predict_seconds": float(f.predict_seconds.sum()),
                   "n_values_corrected": int(f.n_values_corrected.sum()),
                   "fit_warning_count": int(f.fit_warning_count.sum()),
                   "max_correction_scaled": float(f.max_correction_scaled.max())}
        for s in C.CUM_STREAMS:
            summary.update({f"{s}_nrmse_mean": float(f[f"{s}_nrmse"].mean()),
                            f"{s}_nrmse_std": float(f[f"{s}_nrmse"].std(ddof=1)),
                            f"{s}_raw_nrmse_mean": float(f[f"{s}_raw_nrmse"].mean()),
                            f"{s}_end_pct_err": float(combined.loc[s, "end_pct_err"]),
                            f"{s}_pooled_r2": float(combined.loc[s, "r2"])})
        summary_rows.append(summary)
        print(f"{candidate:22s} CV oil/water mean NRMSE = {summary['selection_metric']:.6f}", flush=True)
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), pd.DataFrame(case_rows), audit_rows


def plot_comparison(summary, filename="proxy_cv_compare.png", best_families=False):
    if best_families:
        summary = summary.sort_values("selection_metric", kind="stable").drop_duplicates("model")
        summary = summary.set_index("model").loc[["dumb", "ridge", "xgb", "gp"]].reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(len(summary))
    for i, (s, label, color) in enumerate(zip(C.CUM_STREAMS, ["Oil", "Gas", "Water"], ["#2471a3", "#d9822b", "#28946c"])):
        ax.bar(positions+(i-1)*0.25, summary[f"{s}_nrmse_mean"], width=0.25,
               yerr=summary[f"{s}_nrmse_std"], capsize=2, label=label, color=color)
    ax.set_xticks(positions, summary.candidate, rotation=20, ha="right")
    ax.set_ylabel("Mean case NRMSE (history maximum as fixed scale)")
    ax.set_title("T2 proxy: five-fold CV on training cases 1–70")
    if best_families:
        ax.set_yscale("log")
        ax.axhline(0.002, color="#555555", linestyle="--", linewidth=1, label="Working target: 0.002")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "Lower is better. Error bars: fold SD. Each fold fits 56 cases and evaluates 14. Validation/test not evaluated.", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(C.FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["baseline", "extended"], default="baseline")
    args = parser.parse_args()
    extended = args.suite == "extended"
    if extended:
        # Check the optional dependency before spending time on the GP fits.
        import xgboost  # noqa: F401
    train = training_rows(read_master())
    history = read_history()
    print(f"T2 {args.suite} CV: 70 train cases; validation 71-85 and test 86-100 are not evaluated.", flush=True)
    print("All scaling and PCA are fitted separately inside each fold. All three streams are modelled.", flush=True)
    # Tiny matrices are faster without BLAS spawning many worker threads.
    with threadpool_limits(limits=1):
        summary, folds, per_case, audit = run_cv(train, history, args.suite)
    C.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "proxy_cv_extended" if extended else "proxy_cv"
    summary.to_csv(C.OUTPUTS_DIR / f"{prefix}_comparison.csv", index=False)
    folds.to_csv(C.OUTPUTS_DIR / f"{prefix}_folds.csv", index=False)
    per_case.to_csv(C.OUTPUTS_DIR / f"{prefix}_case_errors.csv", index=False)
    best = select_candidate(summary)
    metadata = {"stage": f"train-only {args.suite} CV", "seed": C.SEED,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "selection_metric": "mean of oil and water NRMSE; gas reported separately",
                "score_normalization": "fixed reference-history maxima from T1 proxy_report",
                "validation_evaluated": False, "test_evaluated": False,
                "gas_from_oil": False,
                "selected_candidate": best.candidate,
                "selection_rule": "best per family, then simplest within 10% of lowest score; dumb < ridge < xgb < gp",
                "pca_extra_rule": "min(fold-local auto count + extra, 10, n_fit-1, 41)",
                "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn"] + (["xgboost"] if extended else [])},
                "folds": audit}
    (C.OUTPUTS_DIR / f"{prefix}_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_comparison(summary, f"{prefix}_compare.png", best_families=extended)
    dumb_score = summary.loc[summary.model == "dumb", "selection_metric"].iloc[0]
    print(f"\nSelected CV candidate: {best.candidate} ({best.selection_metric:.6f})")
    if best.selection_metric >= dumb_score:
        print("No selected-model improvement over the mean curve. Investigate before adding complexity.")
    print("This is a development comparison, not a final model or unbiased final-test result.")
    print(f"Results saved in {C.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
