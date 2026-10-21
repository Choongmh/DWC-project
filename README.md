# Reservoir History Matching and Production Forecasting

An independent learning and portfolio edition for **Choongmh**, based on the completed Intern's Level DataWorks Challenge 2026 team project.

This project uses a Gaussian-process proxy of reservoir simulation, searches for parameter sets that match observed production history, and builds oil, gas and water scenarios through January 2028. It also includes the original Power BI report, presentation, data and a beginner learning workbook.

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

## Data and saved results

The complete local data is under `data/raw/`. Large raw workbooks use Git LFS. Install Git LFS before cloning, then run `git lfs pull`. Files 03–08 are optional reservoir-grid inputs for this implementation; the core pipeline uses 01, 02 and 09. Public distribution is being prepared on the project owner's confirmation that sharing is permitted after the challenge.

The model maps four reservoir parameters to 123 historical cumulative outputs: three streams on 41 quarterly dates from January 1998 to January 2008. Cases 1–70 are development training, 71–85 validation, and 86–100 final test. The frozen final model was fitted on cases 1–85.

The saved final forecast has ten cases and 80 future quarters. The January 2028 median cumulative totals for the main scenario are approximately 59.75 million STB oil, 21.71 million MSCF gas and 87.60 million STB water. These include historical production before 2008.

## Interpretation

The best generated history score is about 0.003454 versus 0.003559 for simulator Case 6. This small difference does not establish a clear improvement. The proxy describes historical simulator behaviour; forecasting to 2028 requires separate decline and water models. Selected-case quantiles are descriptive scenario summaries, not calibrated probabilities. Production alone does not establish profitability.

## Source and status

Original repository: https://github.com/khoonlyn913/dwc2026-internlvl

This local edition was assembled from the final submission archive, the T2 working folder and local deliverables. Live upstream completeness has not yet been verified because the original repository requires accessible GitHub authentication. Publishing to `Choongmh` remains a separate step; this README is not evidence that upload has succeeded.
