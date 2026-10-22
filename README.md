# Reservoir History Matching and Production Forecasting

Personal portfolio and learning edition for **Choongmh**, based on the Intern's Level DataWorks Challenge 2026 team project.

This project combines machine learning with supplied reservoir simulation data to study long-term field performance. A Gaussian Process (GP) proxy supports rapid parameter search and comparison with historical production, followed by separate oil, gas and water forecasting models through January 2028.

Alongside the modeling pipeline developed for the ExxonMobil Malaysia DataWorks Challenge 2026, the repository includes a Power BI dashboard, presentation materials, raw datasets and a beginner learning workbook.

## Problem and objectives

Full-physics reservoir simulations can be expensive to repeat when searching for settings that match observed production. This project uses a proxy to explore many parameter combinations efficiently.

- Approximate historical simulator outputs using a Gaussian Process proxy.
- Search reservoir parameter settings for close matches to observed field history.
- Produce oil, gas and water scenarios through January 2028.
- Explain results, assumptions and limitations through Power BI and presentation materials.

## Choose a task

| Branch | What it contains |
|---|---|
| [main](https://github.com/Choongmh/DWC-project/tree/main) | Complete integrated project and learning workbook |
| [t1-data](https://github.com/Choongmh/DWC-project/tree/t1-data) | Raw inputs, cleaning, scoring, baseline and sensitivity |
| [t2-proxy](https://github.com/Choongmh/DWC-project/tree/t2-proxy) | Proxy modeling, evaluation, trained artifact and necessary data helpers |
| [t3-forecast](https://github.com/Choongmh/DWC-project/tree/t3-forecast) | History matching, forecasting, selected results and frozen proxy dependency |
| [t4-storytelling](https://github.com/Choongmh/DWC-project/tree/t4-storytelling) | Power BI, slides, video, presentation references and supporting results |

Each task branch has a focused README and only the files relevant to that track. Main retains the full workflow. Do not merge main wholesale into a task branch: doing so would bring unrelated files back. Use targeted changes or cherry-pick relevant commits instead.

## Main folder guide

| Folder | Purpose |
|---|---|
| `src/` | T1–T3 Python implementation |
| `data/` | Original workbooks and prepared modeling tables |
| `models/` | Frozen proxy and metadata |
| `outputs/` | Saved metrics, plots and final filled forecast workbook |
| `dashboard/` | Power BI report |
| `presentations/` and `video/` | Original presentation and available T2 video |
| `learning/` | Beginner Word workbook covering T1–T4 |
| `docs/` | Reproduction guide, sources, attribution and presentation corrections |
| `tests/` | Existing automated checks |

Start with [the learning workbook](learning/DWC2026_Learning_Workbook.docx), [run instructions](docs/REPRODUCE.md), or [presentation corrections](docs/PRESENTATION_CORRECTIONS.md).

## Results and interpretation

The proxy maps four reservoir parameters to 123 historical outputs on 41 quarterly dates. Development cases are 1–70, validation cases 71–85, and final test cases 86–100. The selected configuration was fitted on cases 1–85 before final testing.

The main forecast contains ten selected scenarios and 80 future quarters. January 2028 median cumulative totals are approximately 59.75 million STB oil, 21.71 million MSCF gas and 87.60 million STB water, including production before 2008.

The best generated history score is about 0.003454 versus 0.003559 for simulator Case 6. That small difference does not establish a clear improvement. Forecasts use separate decline and water models. Scenario quantiles are descriptive, not calibrated probabilities; production alone does not establish profitability.

## Data and source

Raw Excel workbooks use Git LFS. After cloning, run `git lfs pull`. Optional grid workbooks 03–08 are retained on main and T1; the core pipeline uses workbooks 01, 02 and 09.

Original source: https://github.com/khoonlyn913/dwc2026-internlvl. See [attribution](docs/ATTRIBUTION.md) and [source map](docs/SOURCE_MAP.md). The project was assembled from local final-submission materials; comparison against the original repository's latest remote contents remains unverified.
