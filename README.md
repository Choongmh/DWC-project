# T3 — History matching and production forecasting

T3 searches for reservoir settings that match observed production, then forecasts oil, gas and water production through January 2028. It uses the trained T2 proxy model to explore possible settings without rerunning the reservoir simulator.

## How it works

1. **Search 50,000 parameter combinations.** Use the T2 model to predict historical production for each combination of fault transmissibility, porosity, permeability and aquifer pore volume.
2. **Compare against field history.** Score the predicted cumulative oil, gas and water curves against observed production. Lower normalised error means a closer match. This is called history matching (历史拟合).
3. **Select ten plausible cases.** Choose diverse settings from a pool of near-best matches, rather than simply taking the ten lowest scores. Several different reservoir settings can explain similar production history.
4. **Forecast production.** Fit declining oil and gas rates using the 2003–2008 period. Estimate water using the relationship between water-to-oil ratio and cumulative oil. Apply an assumed liquid-production cap and build future cumulative totals from observed January 2008 totals.
5. **Compare scenarios and export.** Save the main forecast, sensitivity comparisons (敏感性分析), scenario summaries and the completed Excel deliverable.

## What comes out?

| Output | Purpose |
| --- | --- |
| `outputs/t3_top10_params.csv` | Ten selected reservoir settings and their matching scores |
| `outputs/t3_top10_curves.csv` | Their predicted historical production curves |
| `outputs/t3_forecast_all.csv` | Oil, gas and water forecasts for the main and sensitivity scenarios |
| `outputs/t3_p10_p50_p90.csv` | Summaries of the spread across selected cases |
| `outputs/09 Template Deliverable_filled.xlsx` | Final workbook: ten cases, 80 future quarters and three production streams |
| `outputs/figures/` | History-match, forecast and diagnostic charts |

The final workbook covers April 2008 to January 2028: **2,400 cumulative production values**. Oil and water use STB; gas uses MSCF. These outputs also support the [T4 dashboard and presentation](https://github.com/Choongmh/DWC-project/tree/t4-storytelling).

## Files needed by T3

| Location | Purpose |
| --- | --- |
| `models/` | Frozen T2 proxy model and metadata |
| `data/processed/history.parquet` | Observed field production for scoring and forecast starting totals |
| `data/processed/master_table.parquet` | T1 table used for the Case 6 baseline comparison |
| `data/processed/master_table.csv` | Readable copy of that table |
| `data/raw/09 Template Deliverable.xlsx` | Blank template required to generate the final workbook |
| `outputs/baseline_ranking.csv` | Simulator baseline scores used by the comparison script |
| `src/` | T3 search, forecast and export scripts, plus shared dependencies |
| `tests/` | Checks for scoring and proxy behaviour |

The shared proxy and data-reader modules are required even though their names refer to earlier tasks. T3 does not need the original training workbooks or the unused T1 parameter and long-format curve tables. The full project remains on [main](https://github.com/Choongmh/DWC-project/tree/main).

## Run the workflow

Use Python 3.12 and run from the repository folder. The saved results are already included. For experiments, use a separate copy because these commands overwrite generated outputs.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.optimise
.\.venv\Scripts\python.exe -m src.t3_curves
.\.venv\Scripts\python.exe -m src.forecast_all
.\.venv\Scripts\python.exe -m src.fill_template
.\.venv\Scripts\python.exe -m src.p10_p50_p90
.\.venv\Scripts\python.exe -m src.water_cap_sensitivity
.\.venv\Scripts\python.exe -m pytest -q
```

Run the commands in order: `t3_curves` creates the prepared curves needed by the forecast scripts. The blank Excel template uses Git LFS; after cloning, run `git lfs pull` if it has not downloaded. Do not run `src.load_data` here; rebuilding the original tables belongs to T1.

## How to interpret the results

- The selected cases are plausible scenarios, not proof of the true reservoir settings. The small best-score difference from baseline Case 6 is inconclusive.
- The main forecast assumes declining oil/gas rates and a liquid cap derived from nine observations between January 2006 and January 2008. This cap is an assumption, not a measured facility limit.
- P50 is the median of the ten selected cases. P90 denotes the lower estimate and P10 the higher estimate using the petroleum convention; these are not calibrated probabilities.
- The long-term forecast is an extrapolation (外推). Historical diagnostic checks do not independently validate the full forecast through 2028.
