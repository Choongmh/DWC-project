# Reservoir History Matching and Production Forecasting

This DataWorks 2026 project prepares reservoir simulation data, builds a machine-learning proxy, searches for settings that match field history, and forecasts oil, gas and water production through January 2028. Power BI and presentation slides explain the findings.

**Main contains the complete T1–T4 workflow.** Each task branch contains its own focused files and explanation.

## The four stages

| Stage | What happens | Main result |
| --- | --- | --- |
| [T1 — Data preparation](https://github.com/Choongmh/DWC-project/tree/t1-data) | Clean workbooks, align dates, define scoring and compare simulator cases against field history | Prepared tables, baseline ranking and sensitivity analysis |
| [T2 — Proxy modelling](https://github.com/Choongmh/DWC-project/tree/t2-proxy) | Learn the relationship between four reservoir parameters and simulated production | Saved Gaussian Process model and evaluation reports |
| [T3 — History matching and forecasting](https://github.com/Choongmh/DWC-project/tree/t3-forecast) | Search 50,000 parameter combinations, select ten diverse near-best matches and forecast production | Forecast tables, charts and completed Excel deliverable |
| [T4 — Dashboard and presentation](https://github.com/Choongmh/DWC-project/tree/t4-storytelling) | Explain results through interactive visuals and slides | Power BI report and PowerPoint presentation |

T2 maps four inputs—fault transmissibility, porosity multiplier, permeability multiplier and aquifer pore volume—to **123 historical values**: oil, gas and water at 41 quarterly dates from 1998 to 2008. T3 uses separate decline and water models to forecast beyond that period.

## Repository contents

| Folder | Contents |
| --- | --- |
| `data/raw/` | Original Excel workbooks and blank deliverable template; large files use Git LFS |
| `data/processed/` | Prepared T1 tables and cached final T2 evaluation |
| `src/` | T1–T3 implementation and shared helpers |
| `models/` | Frozen T2 model and metadata |
| `outputs/` | Metrics, selected cases, forecasts, charts and final Excel workbook |
| `tests/` | Automated checks of scoring and proxy behaviour |
| `dashboard/` | Power BI report |
| `presentations/` | Project PowerPoint slides |
| `learning/` | Beginner Word workbook covering T1–T4 |

Start with [the learning workbook](learning/DWC2026_Learning_Workbook.docx), [result charts](outputs/figures/) or [the final Excel deliverable](outputs/09%20Template%20Deliverable_filled.xlsx).

## Run the integrated workflow

Use Python 3.12 and Git LFS. From the repository folder in PowerShell:

```powershell
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Prepared tables, the trained model and saved results are already included. To regenerate data preparation and forecasts, use a separate working copy because these commands overwrite generated outputs:

```powershell
.\.venv\Scripts\python.exe -m src.load_data
.\.venv\Scripts\python.exe -m src.baseline
.\.venv\Scripts\python.exe -m src.sensitivity
.\.venv\Scripts\python.exe -m src.optimise
.\.venv\Scripts\python.exe -m src.t3_curves
.\.venv\Scripts\python.exe -m src.forecast_all
.\.venv\Scripts\python.exe -m src.fill_template
.\.venv\Scripts\python.exe -m src.p10_p50_p90
.\.venv\Scripts\python.exe -m src.water_cap_sensitivity
```

These commands reuse the frozen T2 model. For model-development experiments, follow the T2 guide in a separate copy. Do not rerun `proxy_final --prepare` or `--final` over the published artifacts or delete its evaluation guard.

The core workflow uses workbooks 01, 02 and 09. Workbooks 03–08 are optional grid inputs retained with T1's complete source data. Run the commands in order: `t3_curves` creates the intermediate curves needed for forecasting.

## Dashboard and presentation

Open `dashboard/exon_recovered.pbix` in Power BI Desktop and update its data-source paths. Use `outputs/t3_top10_params.csv`, `outputs/t3_top10_curves.csv` and `outputs/t3_forecast_all.csv`, with observed history in `data/processed/history.parquet`. Select the `main` forecast scenario for the primary report. Follow the [T4 guide](https://github.com/Choongmh/DWC-project/tree/t4-storytelling#how-to-build-the-dashboard) for table preparation, relationships and visuals.

Original slides are in `presentations/`. Native opening and refreshing of the recovered PBIX remains unverified.

## Results and interpretation

- T2 uses cases 1–70 for development, 71–85 for validation and 86–100 for the held-out test. The final model is fitted on cases 1–85. Mean test NRMSE is approximately **0.000219 oil, 0.000203 gas and 0.000460 water**. These measure agreement with simulator curves, not future real-world accuracy.
- The final workbook contains **2,400 values**: ten cases × 80 future quarters × three cumulative production streams, from April 2008 to January 2028.
- The best generated history score is about **0.003454**, versus **0.003559** for simulator Case 6. This small difference is inconclusive. The slides' improvement comparison mixes proxy and simulator scores and should be treated as descriptive.
- Forecasts depend on extrapolation and assumptions. The liquid cap uses **nine** quarter-ending observations from January 2006 through January 2008, not eight, and is not a measured facility specification. The cold test does not independently validate the complete long-term forecast.
- **P50 means the median of ten selected cases**, not necessarily the most likely outcome. Scenario quantiles are not calibrated probabilities. Existing dashboard cards use maxima; use explicit date/scenario filters and a median calculation when presenting P50. Small cumulative errors can conceal larger rate errors.
- Production estimates alone do not establish profitability: this project does not calculate revenue, costs or NPV.
