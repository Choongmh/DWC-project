# Reservoir History Matching and Production Forecasting
This repository presents an end-to-end reservoir engineering workflow that combines machine learning and dynamic reservoir simulation to predict long-term field performance. Built around a Gaussian Process (GP) proxy model, this project accelerates the history matching process, enabling rapid parameter search, precise alignment with historical production data, and robust production forecasting through January 2028.

Alongside the core modeling pipeline developed for the ExxonMobil Malaysia DataWorks Challenge (DWC 2026), the repository includes interactive Power BI dashboards, an executive presentation, raw datasets, and a beginner-friendly learning workbook for those exploring ML-assisted reservoir workflows.

## Problem & Objectives
Traditional history matching relies on computationally expensive full-physics reservoir simulators. Running iterative simulations to condition geological parameters against observed field production creates a severe bottleneck when quantifying uncertainty and generating forecasts.

This project aims to:

Train a fast, accurate Gaussian Process proxy to approximate simulator outputs at a fraction of the cost.

Efficiently search and calibrate key subsurface parameter sets against observed historical production.

Build highly detailed multi-phase production forecasts (Oil, Gas, Water) through January 2028.

Deliver clear visual storytelling and comprehensive documentation to bridge the gap between data science and reservoir engineering.

## Start here

- `learning/DWC2026_Learning_Workbook.docx`: start-from-zero lessons, worked examples, exercises and answers.
- `docs/REPRODUCE.md`: safe run order and environment setup.
- `docs/SOURCE_MAP.md`: where the material came from and what is complete.
- `docs/PRESENTATION_CORRECTIONS.md`: corrections to use when presenting older slides.
- `docs/ATTRIBUTION.md`: source project and contribution context.
- `dashboard/`: original recovered Power BI file and inspectable report definitions.
- `presentations/`: original team slides.

## Four tracks

| Branch | Focus | Main modules |
|---|---|---|
| `t1-data` | Understand data, align dates, calculate rates, score and rank baseline | `load_data`, `metrics`, `baseline`, `sensitivity` |
| `t2-proxy` | Train, compare and evaluate a simulator proxy | `proxy`, `proxy_validation`, `proxy_final`, `proxy_diagnostics` |
| `t3-forecast` | Search parameters, select ten cases and forecast production | `optimise`, `t3_curves`, `forecast_all`, `fill_template` |
| `t4-storytelling` | Power BI, interpretation and presentation | `dashboard/`, `presentations/`, workbook T4 lessons |

`main` is the integrated project. The four branches begin from this same integrated snapshot so each can run independently; their names identify the focus of subsequent work, not four invented historical implementations.

## Data & ML Architecture
Git LFS Requirement
The complete local dataset is stored under data/raw/. Because the raw workbooks are large, this repository requires Git Large File Storage (LFS).
- Install Git LFS locally.
- Clone the repository.
- Run git lfs pull to retrieve the full datasets.

Proxy Model Specifications
The GP proxy model maps 4 reservoir parameters to 123 historical cumulative outputs (covering three streams across 41 quarterly dates from January 1998 to January 2008).

- Training Split: Cases 1–70 (Development Training)
- Validation Split: Cases 71–85
- Test Split: Cases 86–100
- The frozen final model was fitted using Cases 1–85.

Final Forecast Results
The saved final forecast comprises 10 selected cases mapped across 80 future quarters. Including historical production prior to 2008, the median cumulative totals for the main scenario by January 2028 are:

- Oil: ~59.75 million STB
- Gas: ~21.71 million MSCF
- Water: ~87.60 million STB

## Interpretation

The best generated history score is about 0.003454 versus 0.003559 for simulator Case 6. This small difference does not establish a clear improvement. The proxy describes historical simulator behaviour; forecasting to 2028 requires separate decline and water models. Selected-case quantiles are descriptive scenario summaries, not calibrated probabilities. Production alone does not establish profitability.
