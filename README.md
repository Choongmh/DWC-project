# T1 — Data preparation and baseline

Prepare the original data, align dates, derive consistent production rates, score existing cases and inspect parameter sensitivity.

## Contents

Raw workbooks 01–08, cleaned tables, six data/scoring modules, baseline/sensitivity outputs and the metric tests. Grid workbooks 03–08 are optional. The T1 hand-off is in `docs/original/`.

## Run

Use Python 3.12 and install `requirements.txt` in a virtual environment. Run from this branch's root. Scripts overwrite their saved results, so use a separate copy for experiments.

```powershell
python -m src.load_data
python -m src.baseline
python -m src.sensitivity
python -m pytest -q
```

## Notes

Outputs: `data/processed/`, `outputs/baseline_ranking.csv`, `outputs/sensitivity_*.csv` and related figures. Case 6 is the saved best existing simulator match.

## Other tracks

[Complete project and learning workbook](https://github.com/Choongmh/DWC-project/tree/main). This branch contains one track and required dependencies. Do not merge main wholesale into it. Large source files use Git LFS; run `git lfs pull` after cloning.

[Original project and contribution credits](docs/ATTRIBUTION.md).
