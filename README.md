# T3 — History matching and forecasting

Use the frozen historical proxy to select ten plausible settings and produce future oil, gas and water scenarios.

## Contents

Forecast/search modules, T3 outputs, the blank template, and the frozen T2 model. Shared config, data readers, metrics, proxy class and prepared T1 tables are necessary dependencies. T1 baseline-building and T2 training/evaluation workflows are not part of this branch.

## Run

Use Python 3.12 and install `requirements.txt` in a virtual environment. Run from this branch's root. Scripts overwrite their saved results, so use a separate copy for experiments.

```powershell
python -m src.optimise
python -m src.t3_curves
python -m src.forecast_all
python -m src.fill_template
python -m src.p10_p50_p90
python -m src.water_cap_sensitivity
python -m pytest -q
```

## Notes

Prepared historical tables are included; do not run `src.load_data` here because workbooks 01–02 belong to T1/main. The filled workbook is `outputs/09 Template Deliverable_filled.xlsx`. The selected main scenario is preserved. Scenario quantiles are not calibrated probabilities, and the best-score difference from Case 6 is inconclusive.

## Other tracks

[Complete project and learning workbook](https://github.com/Choongmh/DWC-project/tree/main). This branch contains one track and required dependencies. Do not merge main wholesale into it. Large source files use Git LFS; run `git lfs pull` after cloning.

[Original project and contribution credits](docs/ATTRIBUTION.md).
