# Team contributions and development record

This supporting document explains the team's T1–T3 work, technical decisions and hand-offs. Contributions are described by task rather than assigning unverified individual authorship. For final results and current reproduction instructions, read `README.md` first.

| Team task | Contribution | Hand-off to the next stage |
|---|---|---|
| T1 — data and scoring | Prepared the master table and field history, aligned production rates, defined shared metrics, ranked existing simulator cases and analysed sensitivity. | Clean data tables and a shared scoring interface for T2 and T3. |
| T2 — historical proxy | Compared models using training-only five-fold cross-validation, checked validation performance, froze the configuration, completed the one-time test and exported the trained proxy. | `models/proxy.joblib`, model metadata, prediction interface, evaluation reports and limitations for T3. |
| T3 — history matching and forecasting | Searched parameter combinations with the frozen proxy, selected ten diverse near-best matches, extrapolated oil/gas/water production, examined sensitivities and filled the deliverable workbook. | Selected parameters, forecast tables, charts and the final Excel deliverable. |

The stages connect through the same parameter schema and history-score functions: T1 supplies consistent data and scoring, T2 supplies a tested emulator, and T3 uses both to construct the final forecast. The team retained the main scenario and original nine-observation cap for submission.

The detailed entries below preserve the development sequence. Statements such as “not yet evaluated,” intermediate test counts, provisional model choices and pending review refer to the stage when they were recorded, not the final submission status. The final README takes precedence for running the submitted solution. This record documents the technical work and hand-offs; it is not a complete individual contribution log or a record of T4 presentation work.

---

# DWC2026: ML-Assisted History Matching & Production Forecasting

## Submission quick start — INTERN'S LEVEL

The final submission uses **`main`**, with the original cap of **13,247.5803 STB/day** computed from **nine quarter-ending observations (2006-01-01 through 2008-01-01 inclusive)**. The trained proxy is `models/proxy.joblib`; do not retrain it or rerun its one-time final test to generate forecasts. The filled Excel and published forecast values are preserved from the reviewed T3 version. Alternative calibrated scenarios are sensitivity analyses only.

Use Python 3.12. Install `requirements.txt` in a virtual environment; when loading the serialized model, use the library versions recorded in `models/proxy_meta.json` (the verified direct-package versions are pinned in `requirements.txt`). Place the confidential organizer workbooks `01...xlsx`, `02...xlsx`, and `09 Template Deliverable.xlsx` under `data/raw/`. Raw input data and virtual environments are deliberately excluded from the archive.

Run from this folder:

```text
python -m src.load_data
python -m src.optimise
python -m src.t3_curves
python -m src.forecast_all
python -m src.fill_template
python -m src.p10_p50_p90
python -m src.water_cap_sensitivity
python -m pytest -q
```

The main forecast is `outputs/t3_forecast_all.csv` filtered to `scenario == "main"`; the template writer uses that scenario. The filled workbook is `outputs/09 Template Deliverable_filled.xlsx`. Scripts `forecast` and `cold_test` are optional diagnostics, not required to produce the final all-stream forecast. The detailed development record below documents earlier stages and their limitations.

Private repo. **The dataset is confidential and must never be committed.**

## Setup
## 1. Put the data in place

Copy the organiser Excel files into **`data/raw/`** (create the folder if it does not exist):

```
data/raw/
├── 01_Introduction.xlsx            # parameters (Uncertainty Values) + glossary
├── 02_Field_Production.xlsx        # simulated curves Case_1–100 + real history (RAMP_History)
├── 03_…xlsx  to  06_…xlsx          # Train cases 1–70   (3D grid, optional)
├── 07_…xlsx                        # Validation cases 71–85 (3D grid, optional)
├── 08_…xlsx                        # Test cases 86–100  (3D grid, optional)
└── 09_…xlsx                        # deliverable template
```

Only the **2-digit prefix** matters (`01`, `02`, …). The rest of the file name may differ.
**Required for the core pipeline: `01` and `02`.** Files `03`–`08` are only needed for the optional grid conversion.


## 2. Run the pipeline

**Required on every laptop (the data files are not in Git):**

| Command | Time | Produces |
|---|---|---|
| `python -m src.load_data` | ~30 s | `data/processed/`: `params`, `curves_long`, `history`, `master_table` (.parquet) + `master_table.csv` |

It prints a summary. You should see `params (100, 6)`, `master_table (100, 129)` and **no warnings**.

**Already in the repo (`outputs/`). Re-run only if you change `config.py` / `metrics.py`, or to reproduce results:**

| Command | Time | Produces |
|---|---|---|
| `python -m pytest -q` | ~2 s | checks the shared scoring → expect `6 passed` (run after ANY change to `metrics.py`) |
| `python -m src.baseline` | ~10 s | `outputs/baseline_ranking.csv`, `outputs/figures/baseline_top10*.png` |
| `python -m src.sensitivity` | ~10 s | `outputs/sensitivity_*.csv`, `outputs/figures/sensitivity_*.png` |

**Optional, not in Git (only for 3D grid visuals / features):**

| Command | Time | Produces |
|---|---|---|
| `python -m src.load_data --grid` | ~40 min | `data/processed/grid/case_###.parquet` |

The grid conversion is resumable: if it stops, run it again and it continues. Keep the laptop plugged in with
sleep set to *Never*. It is faster to share the finished `grid/` folder privately than for everyone to run it.


## 3. Using the data in your code

```python
from src import config as C
from src.load_data import read_master, read_history, read_curves, read_params, y_columns
from src.metrics import score_vs_history, proxy_report, split_y, curves_to_arrays

master = read_master()                        # 100 rows: case, split, 4 params, 123 y columns
X = master[C.PARAM_COLS]                      # fault_trans, poro_mult, perm_mult, aquifer_pv
y = master[y_columns()]                       # oil_cum_t01..t41, gas_cum_t01..t41, water_cum_t01..t41
train = master[master.split == "train"]       # cases 1–70 (validation 71–85, test 86–100)

# score any predicted curves against the REAL history (lower = better)
scores = score_vs_history(split_y(y_pred), read_history())
```

In notebooks (inside `notebooks/`), put this in the first cell so `src` can be imported:
```python
import sys, pathlib; sys.path.insert(0, str(pathlib.Path.cwd().parent))
```

## 4. Project structure

```
src/config.py        all settings (paths, split, dates, scoring). Change values HERE only
src/load_data.py     Excel → parquet + master table; read_* helpers
src/metrics.py       shared scoring (vs history; proxy accuracy)
src/baseline.py      ranks the 100 existing cases → RMSE_baseline (score to beat)
src/sensitivity.py   which parameter drives which output (Spearman, tornado, scatter)
tests/               pytest checks for metrics.py
notebooks/           exploration only, prefix t1_/t2_/t3_/t4_. Clear outputs before committing
outputs/             result tables + figures (committed)
models/              saved proxy models
data/                raw + processed data (NEVER committed)
```

## 7. Team conventions

- One branch per track: `t1-data`, `t2-proxy`, `t3-forecast`, `t4-story`.
- Merge to `main` only through a **Pull Request reviewed by another track**.

**T2 current status (13 October 2026):** final GP is exported and the one-time
test is complete. See [Final model and T3 handover](#t2-final-model-and-t3-handover).
The stage sections below retain the development record; their split restrictions
describe each individual command, not the current completion status.

## T2 stage 1: normalization and five-fold model comparison

Run from the repository root, using the project's Python environment:

```powershell
.\.venv\Scripts\python.exe -m src.proxy
.\.venv\Scripts\python.exe -m pytest -q
```

This development command compares a mean-curve predictor (`dumb`) with six Ridge
configurations. It uses only cases 1-70. Validation cases 71-85 and test cases
86-100 are not evaluated, and no final model is saved. Re-running the command
replaces the stage-1 reports; it does not open the final test.

### What happens in each fold

The same seeded five folds are used for every candidate: 56 cases fit the model
and 14 cases evaluate it. Every training case receives one out-of-fold prediction.

1. Fit input min/max on the 56 cases and apply that scaling to held-out inputs.
   Held-out values can fall outside [0, 1]; they are not clipped to that interval.
2. Divide each stream by one maximum learned from those 56 cases.
3. Fit PCA separately for oil, gas and water on the scaled training curves.
   Choose the smallest component count with fit reconstruction RMSE below 0.0002
   on scored dates, measured relative to that training-stream maximum, capped at
   10 components. Save the reconstruction error and whether the target was met.
4. Fit degree-1 or degree-2 polynomial Ridge on the PCA scores. Alpha is selected
   from 0.0001, 0.01 and 1.0. The mean-curve baseline ignores the inputs and uses
   only the fold's mean training outputs, without PCA.
5. Reconstruct 123 predictions in physical units, in `y_columns()` order. Enforce
   the initial all-zero period observed in that fold's training curves, then
   nonnegative and nondecreasing cumulative values. Record raw and corrected
   errors, the count of changed values, and maximum correction relative to each
   training stream's scale.
6. Evaluate predictions against the corresponding simulator cases with T1's
   `proxy_report()`. Reference history supplies fixed scoring denominators only;
   it is never a training target. All three streams are fitted independently.

### How to read the results

- `outputs/proxy_cv_comparison.csv`: one row per candidate. `degree` is the
  polynomial degree; `alpha` controls Ridge regularization. Smaller error is better.
- `outputs/proxy_cv_folds.csv`: fold-by-fold errors and correction diagnostics.
- `outputs/proxy_cv_case_errors.csv`: out-of-fold errors for each training case.
- `outputs/proxy_cv_audit.json`: fold membership, fitted scales, PCA counts,
  reconstruction errors, seed, package versions and evaluation scope.
- `outputs/figures/proxy_cv_compare.png`: errors for every candidate and stream.

For each stream, `*_nrmse_mean` is the mean of the five folds' mean case NRMSE;
`*_nrmse_std` is the sample standard deviation across those folds (not a confidence
interval). `*_end_pct_err` is the mean absolute percentage error of the January
2008 cumulative value over all out-of-fold predictions. Pooled R-squared is a
secondary diagnostic: the shared time trend can make it high even for a weak proxy.

The provisional T2 selection metric is the equal average of oil and water NRMSE,
as proposed in the T2 spec because gas closely tracks oil. Gas is still fitted
and reported separately. T1's three-stream history-match weights are unchanged.
This proxy metric measures agreement with simulator outputs, not agreement with
field history; do not compare it to Case 6's history-match score as an improvement.

Initial stage-1 run (2026-10-12):

| Candidate | Oil NRMSE | Gas NRMSE | Water NRMSE | Oil/water selection metric |
|---|---:|---:|---:|---:|
| Mean curve | 0.033346 | 0.033375 | 0.083425 | 0.058386 |
| Best linear Ridge (alpha 0.01) | 0.005046 | 0.005020 | 0.009010 | 0.007028 |
| Best quadratic Ridge (alpha 0.0001) | 0.000899 | 0.000874 | 0.001973 | 0.001436 |

The quadratic candidate's out-of-fold endpoint errors were 0.128% for oil,
0.124% for gas and 0.314% for water. These are development results used to compare
configurations, not an independent final-test estimate. The next stage is further
candidate comparison followed by validation diagnostics, before freezing choices
and evaluating the test set. The proxy covers 1998-2008 only.

Eight additional synthetic-data tests check training-only scaling, the mean-curve
baseline, Ridge reconstruction and column order, input validation, physical fixes,
fold isolation, exclusion of validation/test target values, and constant streams.
Together with T1's six tests, stage 1 had 14 passing tests.

## T2 stage 2: compare model families on the same training folds

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m src.proxy --suite extended
.\.venv\Scripts\python.exe -m pytest -q
```

Install the updated `requirements.txt` in the project environment first if needed.
The thread settings avoid numerical-library memory allocation problems on this
Windows machine. They do not change the training data or hyperparameters.

The extended comparison keeps the seven original mean/Ridge candidates and adds:

- Four Gaussian Process (GP) candidates: Matern smoothness `nu` of 1.5 or 2.5,
  with either the fold-local automatic PCA count or that count plus two. The
  latter is capped at 10 (and at the available training rank). Each retained
  component has its own GP, with four learned input length scales, target
  normalization, and five optimizer restarts using seed 42.
- Four XGBoost candidates: tree depth 2 or 3, learning rate 0.05 or 0.1,
  and 500 trees per PCA component. All training rows are used in each fit,
  with seed 42, squared-error loss, and one worker thread.

There are 15 configurations and 75 fold fits. All use the same 56/14 case splits,
fold-local scaling/PCA, physical corrections, and shared T1 metrics. Oil, gas and
water remain independently fitted; gas is not reconstructed from an oil ratio.
The selection metric remains the equal average of oil and water NRMSE.

Select the lowest-error configuration within each family. Among those family
winners, prefer a simpler family if its error is within 10% of the lowest error:
mean, then Ridge, then XGBoost, then GP. The choice is provisional until validation.
More complex models are not required to win.

Results use the separate `outputs/proxy_cv_extended_*` prefix, preserving the
stage-1 reports. The comparison CSV contains all candidates; the figure shows
the best configuration of each family on a log error scale. Its 0.002 line is
a working engineering target, not an official competition cutoff. The audit JSON
records the selected candidate, PCA choices, GP kernels and fit warnings. GP
bound/convergence warnings are recorded, not treated as automatic pass or failure.

No validation or test predictions, final refit, or model export are performed by
either comparison command. No GP uncertainty intervals are exposed at this stage.
The expanded synthetic-data suite has 20 tests, including GP/XGBoost learning,
constant streams, the capped PCA extension, and the model-selection rule.

Stage-2 run (2026-10-12), best configuration from each family:

| Candidate | Oil NRMSE | Gas NRMSE | Water NRMSE | Oil/water selection metric |
|---|---:|---:|---:|---:|
| Mean curve | 0.033346 | 0.033375 | 0.083425 | 0.058386 |
| Quadratic Ridge, alpha 0.0001 | 0.000899 | 0.000874 | 0.001973 | 0.001436 |
| XGBoost, depth 3, learning rate 0.1 | 0.003482 | 0.003517 | 0.008430 | 0.005956 |
| GP, nu 2.5, auto PCA + 2 | 0.000279 | 0.000261 | 0.000602 | 0.000441 |

The provisional winner is `gp_nu2.5_extra2`: six components each for oil and gas,
seven for water in each fold. No simpler family is within 10% of its score. Its
oil/water score is 69.31% lower than the best Ridge score, on these development
folds only. This is not improvement over T1's history-match baseline. Endpoint
errors are 0.0462% oil, 0.0431% gas, and 0.1108% water. Every stream in every GP
fold is below the working 0.002 target. Water error is still about 2.16 times oil
error, so the spec's separate water-to-oil diagnostic is not met; that relative
diagnostic does not override the low absolute errors or establish forecast quality.

All 75 fold audits contain only cases 1-70, all original mean/Ridge scores were
reproduced, and all PCA reconstruction targets were met. Fit time across the
comparison was about 349 seconds on this machine, excluding startup/reporting.
The winning configuration took about 33 seconds across its five fold fits.

GP caveat: the winning candidate recorded 125 parameter-bound warnings and two
L-BFGS convergence warnings across its component/restart fits. Most bound warnings
concern the fault-transmissibility length scale reaching 1000 or the noise level
reaching its lower bound. These are numerical/model diagnostics, not proof that
a physical parameter is irrelevant. Warnings alone do not identify whether an
affected optimizer restart was the selected solution. Keep the full warning
messages and learned kernels in the audit for review before finalization. The
largest physical correction was under 0.00001 of its stream's fitted scale; raw
and corrected scored errors agree at the displayed precision.

Next: investigate the optimizer diagnostics as needed, then fit the fixed
candidate on cases 1-70 and evaluate validation cases 71-85, including curve,
rate, endpoint and ranking checks. Validation and final-test reports are not yet
produced. No final model has been exported or pushed to GitHub.

## T2 stage 3: GP diagnostics and approximate uncertainty

Run `python -m src.proxy_diagnostics` in the project environment, using the
single-thread environment settings above. This runs only the existing winning
GP configuration (`nu=2.5`, fold-local auto PCA + 2) on the same five folds in
cases 1-70. It compares original bounds (length-scale upper bound 1000, noise
lower bound 1e-10) with wider bounds (100000 and 1e-12). The separate numerical
stability term `gp_alpha` stays at 1e-10. No configuration is automatically
promoted and the original model-comparison files are preserved.

The diagnostic optimizer uses the same SciPy L-BFGS-B call and defaults as
scikit-learn. It records all six attempts per component (the initial attempt
plus five restarts), their convergence status, and which attempt supplies the
fitted kernel. This distinguishes a failed discarded attempt from a failed
selected attempt. A synthetic regression test checks that tracing leaves the
default fit's predictions unchanged.

GP pipelines now support `predict_raw(X, return_std=True)` and
`predict(X, return_std=True)`, each returning two arrays of shape `(n, 123)`:
mean and standard deviation in physical units. Other model families raise
`NotImplementedError` for uncertainty requests. The ordinary `predict(X)` API
still returns just the corrected mean.

The approximation propagates each retained component's GP variance through
squared PCA loadings, assuming independent component models. It includes the
fitted WhiteKernel noise, but excludes discarded-PCA reconstruction error,
kernel/parameter estimation uncertainty and the effect of nonlinear physical
corrections. The standard deviation returned beside a corrected mean remains
the raw approximation. A constant training stream has zero modelled variance
by convention, which is not evidence of certainty everywhere. These are
pointwise estimates, not a calibrated simultaneous band for a whole curve.

Coverage is measured on held-out predictions both before and after physical
corrections, with scored dates and positive-production dates reported separately.
No uncertainty multiplier is fitted to force coverage to a desired percentage.

Gas checks retain independent gas modelling. Ratios are in MSCF/STB. Division
excludes oil at or below 1e-8 of the relevant fitting-data oil maximum. Each
fold's reference ratio is fitted on its 56 fitting cases only; predictions are
also compared with the corresponding held-out simulator ratio. The full
70-case ratio row is descriptive and never used to fit fold models.

New output files:

- `outputs/proxy_gp_bounds_comparison.csv`: original/wider scores, timing,
  optimizer failures, and prediction changes.
- `outputs/proxy_gp_diagnostic_folds.csv`: errors, corrections and optimizer
  status counts by fold.
- `outputs/proxy_gp_diagnostic_case_errors.csv`: case-level errors by stream.
- `outputs/proxy_gp_uncertainty.csv`: coverage and average standard deviations.
- `outputs/proxy_gas_check.csv`: simulator and predicted gas/oil consistency.
- `outputs/proxy_gp_diagnostic_audit.json`: fitting/held-out case IDs, scales,
  PCA settings, kernels, warnings and all optimizer attempts.
- `outputs/figures/proxy_gp_diagnostics.png`: accuracy and coverage comparison.

The suite now has 26 passing tests. This stage does not evaluate validation or
test cases, export a final model, or implement the final T3 loading interface.

Stage-3 results (2026-10-13):

| Bounds | Oil NRMSE | Gas NRMSE | Water NRMSE | Oil/water mean | Failed optimizer attempts | Failed selected fits |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0.000279 | 0.000261 | 0.000602 | 0.000441 | 2 / 570 | 0 / 95 |
| Wider | 0.000272 | 0.000255 | 0.000580 | 0.000426 | 4 / 570 | 2 / 95 |

The original results reproduce the saved extended-CV winner. Both original
convergence warnings came from discarded attempts; every selected component fit
converged. Widening bounds changes the selection score by only -3.34%, and many
bound warnings remain. Its two selected nonconverged fits are fold 3 gas PC1 and
fold 5 water PC1. Keep the original bounds as the provisional candidate: the
small CV gain does not resolve the wider model's numerical diagnostics. Defaults
remain unchanged. This sensitivity check supports stability of the mean predictions
to these bounds; it does not prove geological irrelevance or validate uncertainty.

For original bounds, raw +/-2 std coverage on positive-production points is
85.79% oil, 85.44% gas and 90.40% water (versus nominal Gaussian 95.45%). Wider
bounds give 85.25%, 84.83% and 89.86%, respectively. Corrected-center coverage
on these positive points is identical. These are under-covering, approximate
pointwise intervals: do not describe them as calibrated 95% intervals. The report
also includes all scored points, where zeros and numerical reconstruction at
startup change coverage. Validation coverage remains unmeasured. No calibration
has been fitted or claimed.

The training simulator's median gas/oil ratio is approximately 0.3633 MSCF/STB;
its maximum relative deviation is 0.0778%, below the spec's 0.1% diagnostic.
All held-out proxy folds also pass that diagnostic using their own fitting-fold
reference ratios. For original bounds the largest predicted-ratio deviation from
its fitting-fold reference is 0.0840%, and the largest ratio error relative to
the corresponding held-out simulator value is 0.0491%. Gas remains independently
modelled. This is evidence of consistency over these cases/dates, not a guarantee
for arbitrary new parameters or future years.

Next proposed stage: validate the original-bounds candidate on cases 71-85,
including ranking, late-time, rate, parameter-region and uncertainty checks.
Cases 71-100 have not been evaluated by these T2 diagnostics.

## T2 stage 4: fixed-model validation (stop before final test)

Run `python -m src.proxy_validation` in the project environment with the
single-thread environment settings above. It fits the original-bounds GP
(`nu=2.5`, fold-local auto PCA + 2, five optimizer restarts) on cases 1-70 and
predicts cases 71-85. The mean predictor and previously chosen quadratic Ridge
are fitted on the same training rows for comparison. There is no model search,
automatic tuning, uncertainty calibration, final refit or export in this command.
Cases 86-100 are never evaluated. The 30-test synthetic suite checks the rate,
late-time and ranking calculations and verifies test values are excluded.

The validation script records all eight requested checks:

1. Per-stream mean case NRMSE below 0.002 using unchanged T1 scoring.
2. Per-stream validation/CV error ratio no greater than 2.
3. History-score rank correlation at least 0.95 and max absolute score difference
   below 0.003. Also report the best-case IDs, top-three/top-ten overlap and
   maximum rank displacement. Both score lists use unchanged T1 history weights.
4. Pooled RMSE across all validation cases and last eight dates, normalized by
   history maximum, below 0.002; mean January 2008 endpoint percentage error
   below 0.5%. Mean-case late NRMSE and worst endpoint error are also reported.
5. Rates derived from cumulative differences divided by actual interval days,
   assigned to the interval-ending date. The first date is undefined. Rate NRMSE
   is scaled by the maximum true validation rate on scored dates and is report-only.
6. Best, median and worst cases selected by mean three-stream case NRMSE;
   plot cumulative curves, normalized errors and derived rates separately.
7. Error versus each parameter, training-range checks and scaled distance to
   the nearest training case. Correlations from 15 cases are exploratory.
8. Approximate raw/corrected +/-2 std coverage on all scored and positive-production
   points separately. No calibration or nominal-95% claim is made.

Extra checks report physical corrections, count negative/decreasing outputs,
and inspect selected GP optimizer convergence. The water/oil error ratio is
reported without using it as a pass/fail gate.

Tables: `outputs/proxy_val_report.csv`, `proxy_deep_checks.csv`,
`proxy_val_case_errors.csv`, `proxy_val_ranking.csv`, `proxy_val_late_rates.csv`,
`proxy_val_parameter_checks.csv`, and `proxy_val_uncertainty.csv`.
`outputs/proxy_val_audit.json` records fixed settings, fitting/validation IDs,
input-data hashes, learned PCA/kernels, optimizer attempts, metric definitions,
versions and the explicit `test_evaluated: false` flag.

Figures in `outputs/figures/`: `proxy_val_curves.png`, `proxy_val_curve_errors.png`,
`proxy_val_rates.png`, `proxy_ranking.png`, `proxy_val_late_time.png`,
`proxy_error_vs_params.png`, and `proxy_val_uncertainty.png`.

Validation results (2026-10-13), fixed original-bounds GP trained on 1-70:

| Metric | Oil | Gas | Water | Working requirement |
|---|---:|---:|---:|---|
| Validation mean case NRMSE | 0.000177 | 0.000164 | 0.000366 | < 0.002 |
| Validation / CV NRMSE | 0.635 | 0.631 | 0.608 | <= 2 |
| Last-eight-date pooled NRMSE | 0.000247 | 0.000225 | 0.000670 | < 0.002 |
| Mean January 2008 error (%) | 0.0237 | 0.0213 | 0.0635 | < 0.5% |
| Worst January 2008 error (%) | 0.0803 | 0.0654 | 0.1689 | report |
| Rate mean case NRMSE | 0.001761 | 0.001608 | 0.001831 | report |
| Raw +/-2 std coverage, positive production (%) | 82.16 | 79.10 | 86.87 | report, uncalibrated |

All requested numerical accuracy/ranking gates pass. Spearman is 1.0000; every
one of the 15 validation cases has the same rank under simulator and proxy
history-match scores. Mean absolute score difference is 0.000103726; maximum
is 0.000236174, below 0.003. Both select case 76 as the best history match
within this validation subset. Top-three and top-ten sets match exactly. This
does not establish ranking fidelity at every new parameter combination or
improvement over T1's Case 6 baseline.

Curve examples are case 71 (best proxy accuracy), 76 (median) and 73 (worst).
The simulator/proxy curves and derived rates visually overlap closely; separate
error panels expose the small differences. Best proxy accuracy is distinct from
best match to RAMP history. The three highest mean case errors are 73, 81 and 74;
their aquifer PV values are about 93.3, 84.5 and 56.3. Error versus aquifer PV has
exploratory Spearman -0.446, suggesting some relatively higher errors at lower
aquifer PV, but 15 cases cannot establish a reliable weak region or causality.
Case 84 has the highest water case NRMSE (0.000737), still below 0.002.

Two validation cases lie just outside the training parameter box: case 75 has
permeability above the train maximum, and case 82 has permeability above and
fault transmissibility below the training range. Predictions were not clipped.
Their mean three-stream NRMSE values are about 0.000275 and 0.000274. These two
checks do not license arbitrary extrapolation.

There are zero negative/decreasing corrected outputs and zero selected GP fits
with failed optimizer status. 209 values were corrected; the largest change is
8.36e-6 of a fitting-stream maximum. The water/oil NRMSE ratio is 2.067 and remains
report-only. Mean/Ridge comparisons are in the report: Ridge water NRMSE is
0.002168, while GP passes all three streams. No settings were changed in response
to validation, and no uncertainty calibration was fitted.

Uncertainty remains a limitation. Raw coverage over all scored points is
80.70% oil, 77.54% gas and 81.75% water; positive-production coverage is shown in
the table above. The corrected-center positive-production water coverage is
87.07% (oil/gas unchanged). These bands under-cover and must not be advertised
as calibrated 95% intervals or used as validated risk bounds for T3.

At this validation checkpoint, the numerical gates supported retaining the
model. The validation command stops before cases 86-100. The subsequently
authorized final export and test are recorded below.

## T2 final model and T3 handover

The frozen original-bounds GP was refitted on **cases 1-85** and evaluated on
**86-100 once**, with no subsequent model changes. Settings: Matern nu=2.5,
length-scale upper bound 1e3, noise floor 1e-10, alpha=1e-10, five optimizer
restarts, seed 42. Each stream uses fit-only scaling/PCA, automatic reconstruction
count plus two (cap 10, reconstruction tolerance 0.0002). Final retained counts
are oil 6, gas 6, water 7. Gas is independently modelled, not derived from oil.
T1's scoring and production-data processing remain unchanged.

| Proxy accuracy metric | Oil | Gas | Water |
|---|---:|---:|---:|
| Five-fold CV NRMSE, cases 1-70 | 0.000278954 | 0.000260727 | 0.000602408 |
| Validation NRMSE, fit 1-70 / evaluate 71-85 | 0.000177059 | 0.000164421 | 0.000366018 |
| Final test NRMSE, fit 1-85 / evaluate 86-100 | **0.000219347** | **0.000203361** | **0.000459747** |
| Test / validation NRMSE | 1.239 | 1.237 | 1.256 |
| Test last-eight-date pooled NRMSE | 0.000472531 | 0.000452306 | 0.001213078 |
| Test mean absolute 2008 cumulative error | 0.0418% | 0.0374% | 0.0681% |
| Test worst absolute 2008 cumulative error | 0.2304% | 0.2164% | 0.3275% |
| Test mean case rate NRMSE | 0.00190138 | 0.00174142 | 0.00199465 |
| Test worst absolute final-quarter rate error | 2.8571% | 2.2043% | 1.5002% |
| Raw GP +/-2 std coverage, positive-production test points | 85.77% | 83.42% | 89.15% |

NRMSE uses T1's fixed history maximum for each stream and the 38 scored dates.
Rate NRMSE instead uses the maximum true rate in the evaluated split, as in the
validation diagnostics. Rates are differences of cumulative production divided
by actual interval days, assigned to the interval's ending date. The first
date's rate is undefined. CV scores informed model selection; validation and
test are separate checks, and fitting on 85 rather than 70 cases changes the
training sample size. T1 previously ranked all 100 simulator cases against
history; this is a one-time **T2 proxy test**, not a claim that nobody on the
project previously saw these cases.

Validation and test ranking Spearman are both **1.000**, with all 15 positions
preserved in each split. On test, top-three overlap is 3/3 and top-ten overlap
10/10; case 97 is the best history match for both simulator and proxy. Mean
absolute history-score error is **0.000192100**; maximum is **0.001023477** in
case 91. That passes the prespecified 0.003 gate but is larger than 0.0003.
All numerical gates pass, including NRMSE <0.002, test/validation <=2,
late pooled NRMSE <0.002 and mean cumulative endpoint error <0.5%.

The exported model passed serialization, column-order and batched-prediction
checks. All selected component optimizers converged. There are zero negative
or decreasing corrected test predictions; 203 entries were corrected, with a
maximum correction of 5.37e-6 of a fitted stream maximum. **50,000 complete
123-value predictions took 3.24 seconds** on this laptop using one BLAS thread
and batches of 2,048, excluding loading and without std (target <60 seconds).
This is a measured local benchmark, not a guarantee for other computers.
The code suite passed **36 tests**, including synthetic tests of export,
input validation, uncertainty batching and the one-time guard.

### Use the saved model

From the repository root, install the project requirements and import:

```python
import numpy as np
from src import config as C
from src.proxy import load_proxy, predict, param_bounds
from src.metrics import split_y, score_vs_history
from src.load_data import read_history

model = load_proxy()  # optional: predict() also loads and caches the default model
bounds = param_bounds()  # index in C.PARAM_COLS order; columns min/max
X = bounds.mean(axis=1).to_numpy()[None, :]  # one illustrative midpoint
y = predict(X)  # shape (1, 123), physical units
curves = split_y(y)  # oil_cum, gas_cum, water_cum: each (1, 41)
scores = score_vs_history(curves, read_history())  # T1 objective for T3
mean, std = predict(X, return_std=True)  # uncalibrated, optional diagnostic
```

Array input order is **fault_trans, poro_mult, perm_mult, aquifer_pv** in raw
units. Named DataFrames containing exactly these four columns are reordered
automatically. Output order is all 41 oil cumulative dates, then gas, then
water, each from 1998-01-01 to 2008-01-01 (`y_columns()` provides exact names).
Oil/water units are STB; gas is MSCF. Prediction returns a corrected mean;
std remains the uncorrected pointwise GP approximation. Inputs outside the
declared bounds emit a warning and are not clipped.

Prediction and `param_bounds()` require the saved artifact and code, **not raw
Excel or parquet files**. Scoring against field history still requires T1's
local history table. The search bounds follow T1's contract: min/max over all
100 **input parameter sets**. Test production targets were excluded from fitting;
normalization/PCA/GP use 1-85 only. Metadata separately records the narrower fit
box. Case 87 falls just below the fit permeability minimum, though within the
declared search box. Stay inside the declared bounds; accuracy is not guaranteed
everywhere inside this box or between sparsely sampled parameter combinations.

Use a trusted joblib artifact only. For reproducibility, match the recorded
runtime: Python 3.12, numpy 2.5.4, scipy 1.18.1, pandas 3.0.6,
scikit-learn 1.9.1 and joblib 1.6.0. General `requirements.txt` lists minimum
versions; it is not an exact model-loading environment lock.

### Files and final-test safeguard

| File | Contents |
|---|---|
| `src/proxy.py` | Existing CV pipeline plus cached `load_proxy()`, `predict()`, `param_bounds()` |
| `src/proxy_final.py` | Frozen preparation, one-time test, cached reporting and test charts |
| `tests/test_proxy_final.py` | Synthetic interface, serialization, split and repeat-guard tests |
| `models/proxy.joblib` | Trained 1-85 pipeline, input bounds and schemas |
| `models/proxy_meta.json` | Frozen settings, versions, hashes, input/output definitions, fit audit, benchmark and test results |
| `outputs/proxy_final_state.json` | Durable record that test prediction completed once |
| `outputs/proxy_test_report.csv` | Numerical gates, including ranking, plus report-only diagnostics |
| `outputs/proxy_test_streams.csv` | CV/validation/test comparison and cumulative/rate diagnostics |
| `outputs/proxy_test_ranking.csv` | Per-case simulator/proxy history scores and ranks |
| `outputs/proxy_test_case_errors.csv` | Per-case accuracy, parameters, final-rate error and fit-box flag |
| `outputs/proxy_test_uncertainty.csv` | Raw/corrected, all-scored/positive-production interval coverage |
| `outputs/figures/proxy_test_*.png` | Best/median/worst cumulative and rate curves, plus ranking |

Executed sequentially after freezing the implementation:

```powershell
python -m src.proxy_final --prepare
python -m src.proxy_final --final
```

**These commands have already completed; do not rerun them or delete the guard.**
Preparation refuses to overwrite final artifacts. Evaluation verifies hashes
and writes an exclusive marker before predicting on test cases. A crash fails
closed rather than silently repeating evaluation. Full evaluation arrays stay
in ignored `data/processed/proxy_final_evaluation.npz`. On the originating
laptop, `python -m src.proxy_final --report-only` can rebuild reports and figures
from that cache without loading a model or making test predictions. Teammates
without the cache should use committed reports. Raw data, the evaluation cache
and `.venv/` are excluded from Git; the trained artifact is included in this
private repository for the authorized T3 handover.

### Guidance for T3 and presentation claims

1. **Do not over-interpret close scores.** Treat differences within 0.0003 as
   practical ties at minimum; this came from validation and is an empirical
   caution, **not a validated score resolution or global error bound**. Test
   errors reached 0.001023 in case 91, whose true history score is 0.147174.
   The best test match, case 97 (true score 0.003839641), had error 0.000017981;
   one near-optimal example does not establish accuracy throughout that region.
   Perfect ranking across widely separated scores does not resolve near-ties.
2. Start with the candidate pool within 0.0003 of the best proxy score and
   select distinct candidates spread across **scaled** perm, aquifer and fault
   parameters. Report the scores and use a broader pool as a sensitivity check;
   if fewer than ten distinct candidates qualify, disclose any relaxation.
   Near-ties and larger possible surrogate errors should be acknowledged.
   A diverse ten-case scenario set does **not** make its P10/P50/P90 calibrated
   probabilities. Those need explicit uncertainty assumptions and weighting.
3. **Do not claim a proven optimization gain from a tiny proxy-score change.**
   A gain under 0.0003 is inconclusive; a larger gain is not automatically real
   either, given the test maximum and possible error in both compared values.
   Use simulator verification for new candidates if available. Otherwise label
   improvements as proxy-estimated and show sensitivity. Keep T1's direct
   simulator baseline as the honest reference; do not change its weights to
   make improvement look larger.
4. Validation suggested relatively weaker accuracy at low aquifer volume near
   porosity 1.3-1.4, but only 15 validation cases support this exploratory
   observation. Test case 91 shows another weak combination (porosity about
   0.881 and permeability 0.727); case 87 has very low permeability and low
   aquifer volume. The first region is not an exhaustive weakness map.
5. Uncertainty remains **uncalibrated**: positive-production coverage was about
   82%/79%/87% on validation and 86%/83%/89% on test for oil/gas/water, rather
   than nominal 95%. No interval multiplier was fitted on validation or test.
   These are correlated pointwise measurements, not independent confidence
   trials or simultaneous whole-curve coverage. T3's main path need not use std.
6. Small cumulative errors can affect the final slope used for decline curves.
   Final-quarter test rate errors reached 2.86%/2.20%/1.50%; signed mean errors
   were +0.319%/+0.254%/-0.150% for oil/gas/water. Check sensitivity to the chosen
   starting rate and recent fitting window. Do not infer a universal correction
   from these test results or retune T2 using them.
7. The proxy covers **1998-2008 only**. Producing curves for new parameters
   reproduces the simulator's historical response; it neither uniquely
   identifies the underground reservoir nor validates 2008-2028 forecasts.
   T3 must define and justify the separate future-production method.

T2 passes its agreed numerical gates and is ready for the user's review before
requesting T3's pull-request review. The limitations above remain part of the
handover; no post-test model or scoring changes were made.

---

## T3: history matching selection and 2008–2028 production forecast

All forecast values are **proxy-estimated** (GP surrogate, not simulator-verified).

**Final team decision (16 October 2026): submit the existing `main` scenario with the original nine-observation liquid cap (13,247.5803 STB/d).** Keep the existing forecast values and filled workbook unchanged. `rate_calibrated` and `uncapped` remain sensitivity scenarios, not submission values. A retrospective comparison favoured a new fitted-curve calibration on average but not at every cutoff; it was not adopted. Retaining main is a team decision, not a claim that it had the lowest backtest error.

The sensitivity category is now labelled `last_9q_max` instead of `last_8q_max`; only the label changed. Its date window, cap and numerical forecasts are identical.

**Presentation notes:** describe the selected ten as near-best history matches, with diversity used only inside the approximate 0.0003 tie band. The original water cold test is a conditional component diagnostic using known future proxy oil, and the cases were selected against full history. It is not independent validation of the complete 20-year forecast. P10/P90 describe the selected scenarios; neither their spread nor the additional assumption range is a calibrated uncertainty interval. The best-score difference from Case 6 does not establish a clear improvement.

### Run order

Run `python -m src.load_data` first (T1 prerequisite, produces `data/processed/`).

| # | Command | Reads | Writes |
|---|---------|-------|--------|
| 1 | `python -m src.optimise` | frozen proxy, history, Case 6 master-table row | `outputs/t3_candidates_top500.csv`, `outputs/t3_top10_params.csv`, `data/processed/t3_candidates_all.parquet` |
| 2 | `python -m src.t3_curves` | `outputs/t3_top10_params.csv` | `data/processed/t3_top10_curves.parquet`, `outputs/t3_top10_curves.csv`, `outputs/figures/t3_top10_curves.png`, `outputs/figures/t3_top10_closeup.png` |
| 3 | `python -m src.forecast` | `data/processed/t3_top10_curves.parquet`, history | `outputs/t3_forecast_oil.csv`, `outputs/figures/t3_forecast_oil.png` |
| 4 | `python -m src.cold_test` | `data/processed/t3_top10_curves.parquet`, history | `outputs/figures/t3_cold_test_oil.png` (printed diagnostics only) |
| 5 | `python -m src.forecast_all` | `data/processed/t3_top10_curves.parquet`, history | `data/processed/t3_forecast_all.parquet`, `outputs/t3_forecast_all.csv`, `outputs/figures/t3_forecast_all.png` |
| 6 | `python -m src.fill_template` | `outputs/t3_forecast_all.csv`, `data/raw/09 Template Deliverable.xlsx` | `outputs/09 Template Deliverable_filled.xlsx` ¹, `outputs/t3_template_mapping.csv` |
| 7 | `python -m src.p10_p50_p90` | `outputs/t3_forecast_all.csv`, history | `outputs/t3_p10_p50_p90.csv`, `outputs/figures/t3_fan_chart.png`, `outputs/figures/t3_uncertainty_compare.png` |
| 8 | `python -m src.water_cap_sensitivity` | `outputs/t3_forecast_all.csv`, history | `outputs/t3_water_cap_sensitivity.csv`, `outputs/figures/t3_water_cap_sensitivity.png` |

¹ `.xlsx` is excluded by `.gitignore`; deliver the filled template to the team outside Git.

Scripts 3 and 4 are diagnostic; script 5 supersedes script 3 for the final forecast.

### Key results

Score convention: lower is better (equal-weight NRMSE on oil, gas, water). **P90 = LOW (conservative), P10 = HIGH (optimistic).**

- **Best proxy-estimated score: 0.003454** (our id 1). Simulator Case 6 scores 0.003559; the difference (0.000105) is inside the 0.0003 practical tie band — not a clear improvement.
- **Mean proxy score of our 10 picks: 0.003655** (proxy-estimated). Baseline simulator top-10 mean: 0.004528 (simulator). The proxy-selected pool scores lower (better) on average, but proxy and simulator scores are not directly comparable.
- **P10–P90 parameter spread at 2028:** ≈ 0.6 % of the median (oil 0.62 %, gas 0.66 %, water 0.56 %). **Scenario/assumption spread:** ≈ 4.7 % for oil and gas, 15.7 % for water. Scenario spread dominates parameter spread by 7–28×. The P10/P90 values are **not calibrated probabilities**.

### Assumptions and limitations

1. **Proxy-estimated throughout.** All outputs come from the frozen GP proxy (1998–2008 only). No simulator verification of the 2008–2028 extrapolation was performed.
2. **Candidate selection.** 50 000 uniform samples (seed 42) were scored via the proxy. The 10 cases were chosen by greedy farthest-point diversity from the pool scoring within 0.0003 of the best (practical tie band), spread across scaled perm\_mult, aquifer\_pv, and fault\_trans.
3. **Option B start.** The Arps and WOR fits use the proxy rates in the 2003-01-01 to 2008-01-01 window; the proxy cumulative at 2008-01-01 is not used for fitting.
4. **Oil and gas: harmonic Arps (b = 1).** The b parameter hit the upper bound (1.0) for all 10 cases; AICc and the cold test both favour harmonic over exponential (b = 0). Exponential was rejected because the cold test showed it over-declines in the 2–4 year horizon.
5. **Water: log WOR vs cumulative oil.** Chosen by a 3-year cold test over two alternatives (log WOR vs time, logistic fw vs time). Total liquid rate is capped at 13 248 STB/d — the maximum observed over 9 quarter-ending observations, 2006-01-01 through 2008-01-01 inclusive — reflecting the modelling assumption that history shows no rising total liquid; this is not a physical hard limit. The cap first binds around 2012 (median first-binding year across cases), so from that point the water forecast is mostly determined by (cap − oil rate) rather than by the WOR trend. Even the best water method over-predicted the 3-year cumulative gain by about 15 % versus the proxy in the cold test; uncapped results are stored as a separate scenario for comparison only and are not in the template. Cap sensitivity (proxy-estimated): the median 2028 water cumulative changes from 86.3 M STB at the tightest cap tested (13 029 STB/d, last quarter before 2008, binding from 2011) to 87.6 M STB at the current cap (13 248 STB/d, binding from 2012) to 100.9 M STB at the historical maximum cap (17 486 STB/d, binding from 2025) and 101.3 M STB uncapped — a 16 % range driven almost entirely by the choice between a tight vs. loose cap.
6. **rate\_calibrated scenario.** Each stream's rates are scaled by the ratio (observed rate at 2008-01-01) / (proxy rate at 2008-01-01): oil and gas ≈ 0.87, water ≈ 1.11. See also assumption 10.
7. **Cold test is optimistic.** The 2–4 year cold-test horizon is much shorter than the real 20-year extrapolation; longer-term forecast errors may be larger.
8. **Uncertainty size.** At 2028-01-01 the 10-case P10–P90 parameter spread is ≈ 0.6 % of the median (oil 0.62 %, gas 0.66 %, water 0.56 %). The scenario/assumption range is ≈ 4.7 % for oil and gas, and 15.7 % for water (driven by the liquid-cap assumption). Scenario spread dominates parameter spread by roughly 7–28×; the P10/P90 values are **not calibrated probabilities**.
9. **Fit-window sensitivity (oil).** 2028-01-01 oil cumulative from the 2003-01-01 fit window is 59.2–59.8 M STB (our main scenario, harmonic). Using a 2005-01-01 window gives 60.5–61.4 M STB; a 2006-01-01 window with free b gives 62.7–64.5 M STB. The window choice alone changes 2028 oil by about 7 %; this is the dominant extrapolation sensitivity for oil, outside the scenario range shown in the fan chart.
10. **Proxy bias at 2008-01-01.** Across the 10 selected sets, the proxy oil rate at 2008-01-01 is 12–15 % above the observed value (proxy 5896–6039 STB/d vs observed 5247 STB/d), while the proxy water rate is 10–11 % below observed (proxy 6953–7025 STB/d vs observed 7782 STB/d). This systematic offset is the reason the rate\_calibrated scenario exists.
