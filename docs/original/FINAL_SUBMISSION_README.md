# INTERN'S LEVEL — DataWorks Challenge 2026

## Final submission

This project builds a machine-learning proxy for reservoir simulation, uses it to identify parameter combinations that reproduce observed field history, and extrapolates the selected production profiles to January 2028.

The submitted forecast is the **main scenario**. The team retained the existing forecast values and the supplied final `09 Template Deliverable_filled.xlsx`. The workbook is the final version confirmed by the team on 16 October 2026. It contains ten cases, each with 80 quarterly cumulative gas, oil and water values from 1 April 2008 to 1 January 2028.

The total-liquid rate cap is **13,247.5803 STB/day**, calculated from **nine quarter-ending observations from 1 January 2006 through 1 January 2008 inclusive**. Earlier references to eight quarters were wording errors; the original calculation and forecast values have been retained.

## 1. Files in this package

| Path | Purpose |
|---|---|
| `src/` | T1 data preparation/scoring, T2 proxy, and T3 optimization/forecasting code |
| `models/proxy.joblib` | Frozen trained proxy, ready to load |
| `models/proxy_meta.json` | Training configuration, input/output schema, bounds and model metadata |
| `tests/` | Existing automated checks for metrics, proxy behaviour and evaluation safeguards |
| `outputs/` | Saved baseline, development/validation/test reports, selected parameters and forecasts |
| `outputs/figures/` | Analysis and forecast charts |
| `outputs/09 Template Deliverable_filled.xlsx` | Final main-scenario Excel deliverable |
| `requirements.txt` | Exact direct-package versions used in the verified Python 3.12.4 environment |
| `DEVELOPMENT_NOTES.md` | T1–T3 contributions, hand-offs, development decisions and methodological caveats |

The archive excludes the confidential original dataset, `.venv` and Git metadata. The original challenge workbooks must be provided separately to rerun the workflow. No raw-data download is performed by the code.

## 2. Inputs and problem definition

Place the organizer's files in `data/raw/`:

```text
data/raw/
  01 Introduction.xlsx
  02 Field Production.xlsx
  09 Template Deliverable.xlsx
```

Files `01` and `02` supply the four parameters and simulated/observed production used by the core pipeline. The loader finds these files by their two-digit prefixes. The current template writer expects the exact filename `09 Template Deliverable.xlsx`.

Files `03`–`08` contain reservoir-grid data. They are not required by the chosen four-parameter proxy pipeline; they are optional for the project's grid-conversion functionality.

The proxy inputs, in model order, are:

1. Fault transmissibility (`fault_trans`).
2. Porosity multiplier (`poro_mult`).
3. Permeability multiplier (`perm_mult`).
4. Aquifer pore volume (`aquifer_pv`).

Each input combination produces three cumulative curves on 41 quarterly dates from January 1998 through January 2008. Model output order is oil, gas, then water: 123 values in total. Oil and water cumulative volumes are in STB; gas cumulative volume is in MSCF.

Two different comparisons must be distinguished:

- **Proxy accuracy:** predicted curves versus the corresponding simulator curves for held-out parameter cases.
- **History-match quality:** simulator/proxy curves versus the observed `RAMP_History` field data.

Observed field history is not a target used to train the T2 emulator. It is used for history matching and for the observed cumulative starting totals of the T3 forecast.

## 3. T1 — prepare data, define scoring, establish the baseline

`src/load_data.py` reads parameters and cumulative production, aligns the simulator outputs with the 41 observation dates, and writes:

- `data/processed/params.parquet`.
- `data/processed/curves_long.parquet`.
- `data/processed/history.parquet`.
- `data/processed/master_table.parquet` and `master_table.csv`.

The master table has 100 rows and 129 columns: case/split identifiers, four parameters and 123 cumulative targets. Production rates are derived consistently as cumulative increments divided by actual days in the quarter, assigned to the quarter-ending date. Initial zero-production values are retained as valid data.

`src/metrics.py` defines the shared working history score. For each stream, it calculates RMSE over dates from October 1998 onward and divides by the maximum observed cumulative value for that stream. The combined score averages oil, gas and water NRMSE equally. This is the team's declared metric, not a claim about a disclosed hidden organizer scoring formula.

`src/baseline.py` compares the 100 existing simulator curves directly with field history. No machine-learning training is needed for this baseline. The best existing case is Case 6, with score approximately **0.00355856**. `src/sensitivity.py` provides the parameter/output sensitivity analysis.

## 4. T2 — train and verify the historical proxy

The case split is fixed:

| Stage | Cases | Role |
|---|---|---|
| Development training | 1–70 | Five-fold internal cross-validation and model comparison |
| Validation | 71–85 | Held-out development checks |
| Final fit | 1–85 | Fit the frozen chosen configuration |
| Final test | 86–100 | One-time assessment after configuration freeze |

Scaling and PCA are fitted within each cross-validation training fold. The selected model uses independent Gaussian-process models for oil, gas and water, a Matérn kernel with smoothness 2.5, and PCA compression of each curve. Gas is modelled separately, not generated by multiplying oil by a fixed ratio. The final retained component counts are six for oil, six for gas and seven for water.

Saved one-time test results:

| Stream | Mean proxy NRMSE versus simulator |
|---|---:|
| Oil | 0.00021935 |
| Gas | 0.00020336 |
| Water | 0.00045975 |

These establish proxy accuracy for the evaluated historical simulator curves. They do not establish 2008–2028 forecast accuracy. See `outputs/proxy_test_streams.csv` and `outputs/proxy_test_report.csv` for the full diagnostics.

The saved model is already trained. **Do not run `src.proxy_final --prepare` or `--final` to reproduce the submission forecast.** The original test was already completed; its saved reports are included.

Public interface:

```python
from src.proxy import load_proxy, predict, param_bounds

load_proxy()
bounds = param_bounds()
# X: DataFrame with the four named input columns, or an array in model order.
curves = predict(X)  # shape: (number_of_cases, 123)
```

Stay within the supplied parameter bounds. Optional standard deviations are approximate and are not calibrated 95% coverage guarantees.

## 5. T3 — select near-best history matches

`src/optimise.py` draws 50,000 parameter combinations uniformly inside the model bounds using seed 42. It predicts their historical curves with the frozen proxy and scores them using the shared history metric.

The best generated proxy score is approximately **0.00345414**. Its difference from the simulator Case 6 baseline is approximately **0.00010442**, inside the practical **0.0003** score-resolution band. We do not claim this establishes a clear improvement over Case 6.

The algorithm identifies the candidates within 0.0003 of the best score and selects ten using parameter diversity within that group. These are **ten near-best history matches**, not the ten strict smallest numerical scores. Diversity is secondary to the near-best history-score restriction. The tolerance is a heuristic, not a statistically guaranteed equality interval.

Outputs:

- `data/processed/t3_candidates_all.parquet`: all 50,000 candidates.
- `outputs/t3_candidates_top500.csv`: best 500 candidates.
- `outputs/t3_top10_params.csv`: selected ten parameter combinations and scores.

`src/t3_curves.py` predicts the ten historical curves, derives their rates and writes `data/processed/t3_top10_curves.parquet` and `outputs/t3_top10_curves.csv`.

## 6. T3 — forecast January 2008 to January 2028

The proxy only covers 1998–2008. `src/forecast_all.py` performs a separate extrapolation:

- Fit harmonic decline curves to oil and gas rates over January 2003–January 2008.
- Fit log water/oil ratio against cumulative oil for water forecasting.
- Feed forecast oil rates and cumulative oil into the water model.
- Apply the selected liquid-rate cap by limiting water to the remaining capacity after oil. This treatment and the cap are explicit modelling assumptions, not a measured facility capacity supplied by the FAQ.
- Start each stream at the observed January 2008 cumulative total and add predicted quarterly rates multiplied by actual quarter lengths.

The saved table contains `main`, `rate_calibrated` and `uncapped` scenarios. **Only `main` is submitted.** The other scenarios are sensitivity comparisons. The table includes a January 2008 anchor row; the Excel contains only the following 80 forecast quarters.

The team deliberately retained main after considering retrospective calibration comparisons. This is not a claim that main won every backtest. Alternative experimental forecasts are not substituted into the final workbook.

Current main-scenario medians at January 2028:

| Stream | Cumulative total | Unit |
|---|---:|---|
| Oil | 59,747,756.35 | STB |
| Gas | 21,706,343.67 | MSCF |
| Water | 87,596,481.91 | STB |

These totals include production accumulated before January 2008, not just the subsequent 20 years.

`src/fill_template.py` writes gas, oil and water in the template's column order. Scenario id 1 maps to Case 1, and so on; these are selected forecast scenarios, not original simulator case numbers. The writer extends the short Case 1 in the supplied blank template to January 2028.

`src/p10_p50_p90.py` calculates descriptive quantiles across the ten selected cases. `src/water_cap_sensitivity.py` compares alternative liquid caps. The main cap's category is now named `last_9q_max`; only this label was corrected from its earlier `last_8q_max` name.

## 7. How to run

The checked environment uses **Python 3.12.4**. `requirements.txt` records the exact direct-package versions used for verification and model loading. It is not a fully pinned transitive environment lockfile.

Example on Windows, from this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Put the organizer workbooks under `data/raw/`, then run:

```powershell
.\.venv\Scripts\python.exe -m src.load_data
.\.venv\Scripts\python.exe -m src.optimise
.\.venv\Scripts\python.exe -m src.t3_curves
.\.venv\Scripts\python.exe -m src.forecast_all
.\.venv\Scripts\python.exe -m src.fill_template
.\.venv\Scripts\python.exe -m src.p10_p50_p90
.\.venv\Scripts\python.exe -m src.water_cap_sensitivity
.\.venv\Scripts\python.exe -m pytest -q
```

On other systems, use the equivalent virtual-environment Python executable. For small-model performance, setting `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to `1` can avoid unnecessary threading overhead.

Optional diagnostics:

```text
python -m src.baseline
python -m src.sensitivity
python -m src.forecast
python -m src.cold_test
```

Use the virtual-environment interpreter for these commands too. `src.forecast` is an oil-only diagnostic; `src.forecast_all` produces the final all-stream forecast.

If only reviewing the submission outputs, the final Excel, CSV tables, reports and charts are already included. Rerunning is not necessary to view them. Reproduction can show small numerical fitting differences across library/platform versions.

## 8. Verification and limitations

Final checks:

- All **36 existing automated tests** passed.
- The packaged source was extracted, imported and used to load the frozen model and reproduce a selected historical prediction.
- All **2,400 production cells** in the filled workbook match the saved main forecast within floating-point precision.
- All ten sheets have the correct 80 ordered quarterly dates through January 2028, correct gas/oil/water units, finite nonnegative values and nondecreasing cumulative production.
- Model and saved numerical forecast files were preserved. The sensitivity data changed only its cap-category text, and its figure legend now says nine quarters.
- ZIP integrity and all archived file bytes were checked.

Interpretation limits:

- A low history score is not proof that the inferred reservoir parameters are unique or that future production is accurate.
- The original water cold test uses known future proxy oil to isolate water-model behaviour. It is a conditional diagnostic, not independent validation of the complete forecast pipeline. The cases were selected using full field history.
- Short-horizon diagnostics do not establish 20-year accuracy; longer-term errors may be larger.
- The ten cases were selected deliberately. Their P10/P50/P90 values are scenario quantiles, not calibrated probabilities. The additional assumption range also does not cover every source of uncertainty.
- Cap selection and decline-fit window materially affect forecasts. Maintain these caveats when interpreting the submitted main scenario.

The filled workbook's SHA-256 is `1cbb76ee14ad220b296c5a8c7536d82bf133ae1adf33dc8f5ea3ae2534b4977e`. It is unchanged from the final file supplied and confirmed by the team.
