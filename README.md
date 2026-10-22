# T1 — Data preparation, scoring and baseline

T1 turns the supplied reservoir workbooks into consistent modeling tables, defines the shared scoring method, ranks existing simulator cases against observed history, and explores parameter sensitivity.

The best existing simulator match is **Case 6**, with a history score of approximately **0.00355856**. Lower is better. This is the baseline for later history matching, not a measure of proxy accuracy.

## Folder guide

| Path | Purpose |
|---|---|
| `data/raw/` | Original workbooks 01–08; the core workflow uses 01 and 02 |
| `data/processed/` | Prepared parameters, simulated curves, observed history and master table |
| `src/` | Data preparation, scoring, baseline and sensitivity code |
| `outputs/` | Baseline rankings, sensitivity results and figures |
| `tests/test_metrics.py` | Six automated scoring checks using small examples with known answers |
| `requirements.txt` | Package versions for reproduction |
| `.gitattributes` | Sends large Excel files through Git LFS |
| `.gitignore` | Excludes environments, caches, temporary files and credential files |

## Setup and run

Use Python 3.12. From this branch's root:

```powershell
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.load_data
.\.venv\Scripts\python.exe -m src.baseline
.\.venv\Scripts\python.exe -m src.sensitivity
```

Data preparation should produce a master table with **100 rows and 129 columns**. Existing prepared tables are included for inspection. Running the commands overwrites generated outputs, so use a separate copy for experiments.

Optional scoring checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected result: **6 passed**. Optional reservoir-grid conversion:

```powershell
.\.venv\Scripts\python.exe -m src.load_data --grid
```

Grid workbooks 03–08 are much larger and are not needed by the selected four-parameter proxy workflow. Grid conversion takes considerably longer than preparing the core tables.

## Implementation

| Module | Responsibility |
|---|---|
| `src/config.py` | Central paths, parameter names, case split, dates and scoring weights |
| `src/load_data.py` | Read Excel, align curves, derive rates and save prepared tables; provide `read_*` helpers |
| `src/metrics.py` | Shared scoring and proxy-evaluation functions |
| `src/baseline.py` | Rank existing simulator cases against field history without training an ML model |
| `src/sensitivity.py` | Analyze parameter/output relationships and generate charts |

## Prepared data contract

| File in `data/processed/` | Contents | Use |
|---|---|---|
| `params.parquet` | 100 cases, four parameters, case identifier and split | Model inputs |
| `curves_long.parquet` | 100 cases × 41 dates, with cumulative production and derived rates | Plotting and later rate fitting |
| `history.parquet` | Observed field history on the same 41 dates, plus original reported rates | History-match reference |
| `master_table.parquet` | One row per case: two identifiers, four inputs, 123 cumulative targets | Proxy training/evaluation table |
| `master_table.csv` | CSV version of the master table | Inspection and interchange |

The 123 targets are three cumulative curves × 41 quarterly dates. Columns run from `oil_cum_t01` through `oil_cum_t41`, then the gas and water equivalents. Oil and water totals use STB; gas totals use MSCF.

### Input parameters

| Parameter | Approximate sampled minimum | Approximate sampled maximum | Meaning |
|---|---:|---:|---|
| `fault_trans` | 0.050 | 0.150 | Ease of fluid movement across the modeled fault |
| `poro_mult` | 0.80 | 1.49 | Multiplier for pore-space storage |
| `perm_mult` | 0.52 | 9.78 | Multiplier for ease of flow in the specified lower layers |
| `aquifer_pv` | 54 | 200 | Aquifer size parameter |

These are rounded descriptive bounds. Use the precise values from the supplied data/model metadata in numerical work.

### Dates and production rates

Historical curves share 41 quarterly dates from **1 January 1998 to 1 January 2008**. The original simulator sheets have irregular dates and blanks in individual case columns. The loader processes each column independently and selects the common observation dates.

The original T1 hand-off reports exact values at those dates for the supplied data. The code also has an interpolation fallback with a warning; investigate any such warning instead of assuming all values were observed.

Reported simulator and field rates use different quarter conventions. To compare them consistently, the project derives both from cumulative totals:

```text
rate(t) = [cumulative(t) - cumulative(previous date)] / actual elapsed days
```

Each derived rate describes the quarter ending at its labeled date. The first rate is set to zero as a boundary convention. Use `oil_rate`, `gas_rate` and `water_rate` for comparisons; original field rates are preserved as `*_reported` columns.

For optional grid conversion, `-999.25` denotes inactive cells and is converted to missing data. It is not a physical negative rock property.

## Split and modeling rules

- Cases **1–70**: development training.
- Cases **71–85**: validation.
- Cases **86–100**: final proxy test; do not use them to tune the model.
- `RAMP_History` is observed field history. It is a reference for matching, not a T2 simulator-training target.
- Inputs for a new case must be computable from the four available reservoir parameters.
- Keep parameter searches inside the specified bounds and acknowledge proxy approximation error.

T1's ranking of all 100 supplied simulations is a descriptive baseline and includes held-out cases. It must not be used to retune T2 against final-test performance. The supplied production history ends in January 2008; forecasting beyond that requires the separate T3 method.

## Shared scoring method

For each cumulative stream:

1. Score the 38 dates beginning **1 October 1998**, excluding the first three shared zero-production dates.
2. Calculate root mean squared error (RMSE) against the reference curve.
3. Divide by the maximum observed cumulative production for that stream to obtain NRMSE.
4. Average oil, gas and water NRMSE with equal weights of one third.

```text
score = (NRMSE_oil + NRMSE_gas + NRMSE_water) / 3
```

Zero indicates a perfect match. For example, an RMSE of 10 with reference maximum 100 gives NRMSE 0.10. This metric is not a classification accuracy percentage. It is the team's declared working score, not a disclosed hidden organizer formula.

| Function | Comparison | Used by |
|---|---|---|
| `score_vs_history()` | Candidate curves versus observed field history | T1 baseline and T3 search |
| `proxy_report()` | Proxy predictions versus corresponding simulator curves | T2 evaluation |

Rate-based scoring provides a secondary diagnostic. Agreement with cumulative rankings supports consistency, but does not prove forecast accuracy. Gas and oil are strongly related in this dataset; equal weighting is an explicit modeling choice. Alternative weights were discussed in the hand-off, but the retained main score uses one third per stream.

## Findings and interpretation

- Case 6 is the best existing simulator match, with score approximately 0.00355856. Inspect `outputs/baseline_ranking.csv` for the full ranking and per-stream errors.
- Water has the largest normalized cumulative mismatch among the strongest baseline matches.
- The original analysis found porosity multipliers around 1.35–1.38 among the top ten matches, while permeability, aquifer and fault settings varied more widely. This indicates non-unique parameter combinations within the sampled model family.
- Late historical mismatches persist: the top cases tend to overpredict oil/gas cumulative production and underpredict water. Inspect residuals and rates rather than relying only on the overall score.
- Rate-based and cumulative rankings share eight of their top ten cases in the original analysis. This is a useful consistency check, not independent validation.

Parameter correlations describe this dataset and time window. They do not establish that other parameters never matter. Several good historical matches do not automatically produce a calibrated forecast probability distribution. Later T3 results also showed that forecasting assumptions can matter more than the spread across selected parameter settings.

Useful figures:

- `outputs/figures/baseline_top10.png`: top simulator curves and observed history.
- `outputs/figures/baseline_top10_closeup.png`: detailed historical mismatches.
- `outputs/figures/sensitivity_tornado.png`: ranked associations between parameters and outputs.
- `outputs/figures/sensitivity_scatter.png`: individual parameter/output relationships.

## Use T1 tables in code

```python
from src import config as C
from src.load_data import read_master, read_history, y_columns

master = read_master()
history = read_history()
train = master.loc[master['split'].eq('train')]
X_train = train[C.PARAM_COLS]
y_train = train[y_columns()]
assert X_train.shape == (70, 4)
assert y_train.shape == (70, 123)
```

T2 uses the master table and proxy metrics. T3 uses observed history, shared scoring and derived rates. T4 uses T1's findings, ranking and figures to explain the data and baseline.
