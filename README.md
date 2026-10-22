# T4 — Dashboard and presentation

T4 communicates the reservoir analysis and production forecasts through an interactive Power BI dashboard and presentation slides.

## What is included?

| File | Purpose |
| --- | --- |
| `dashboard/exon_recovered.pbix` | Power BI report |
| `presentations/interns level ppt.pptx` | Original project presentation |

This branch contains the presentation deliverables. Data and modelling code are available in [main](https://github.com/Choongmh/DWC-project/tree/main). No video is included here.

## How to build the dashboard

1. **Get the prepared results** from [T3's outputs folder](https://github.com/Choongmh/DWC-project/tree/t3-forecast/outputs): `t3_forecast_all.csv`, `t3_top10_curves.csv` and `t3_top10_params.csv`. To add observed field history, obtain `data/processed/history.parquet` from main.
2. **Import the files into Power BI Desktop.** Use Power Query to set `date` as a date, `id` as a whole number and production values as decimals. Keep the `main` forecast scenario for the primary report; the other scenarios are sensitivity comparisons.
3. **Prepare a production table.** Append the selected historical curves and forecast rows using their shared columns. The files overlap on January 2008: keep that boundary once per case. Add a History/Forecast label and name the combined query `Master Production`. Keep observed field history as a separate reference table.
4. **Connect the parameters.** Create a one-to-many relationship from `t3_top10_params[id]` to `Master Production[id]`. Each selected case should have one parameter row and many dated production rows.
5. **Build the visuals.** Add date and case slicers, line charts for oil/gas/water over time, a parameter table and cards for end-date cumulative production. Use one case per line or an explicit scenario summary; do not sum cumulative production across dates or across alternative cases. Label oil/water in STB and gas in MSCF.
6. **Check and save.** Compare the displayed numbers against the source CSVs, check units and filters, and save the PBIX. For a P50 card, calculate the median across the ten cases at the selected date.

To work from the supplied PBIX instead, open it in Power BI Desktop, inspect its queries and update the source paths to your downloaded data. Then refresh and check the visuals. Opening and refreshing this recovered PBIX in Power BI Desktop has not yet been verified.

## Presentation notes

The slides are the original submission. When presenting them, use these clarifications:

- P50 means the median of ten selected scenarios, not necessarily the most likely outcome. These scenarios are not calibrated probabilities. Existing dashboard cards use maxima, not P50.
- The water-cap assumption uses nine quarter-ending observations from January 2006 through January 2008, not eight. It is not a measured facility limit.
- The quoted score improvement is a descriptive comparison of proxy and simulator scores. The cold test does not independently validate the full long-term forecast; small cumulative errors can also hide larger rate errors.
- Profitability was a motivating question, but the project does not calculate costs, revenue or NPV.

## Credits

Based on the DataWorks 2026 team submission by Choong Ming Hang, Koay Khoon Lyn, Tan Pei Ying and Lam Zhuo Yi. Original source: [dwc2026-internlvl](https://github.com/khoonlyn913/dwc2026-internlvl). This repository is a personal portfolio and study edition.
