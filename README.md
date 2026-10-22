# T2 — Simulator proxy

Train and evaluate the historical simulator proxy and inspect the preserved final model.

## Contents

Proxy implementation, trained model, validation/test reports and tests. T1 config, loader, metrics and cleaned tables are required dependencies; workbooks 01–02 allow rebuilding those tables.

## Run

Use Python 3.12 and install `requirements.txt` in a virtual environment. Run from this branch's root. Scripts overwrite their saved results, so use a separate copy for experiments.

```powershell
python -m src.load_data
python -m pytest -q
```

## Notes

The final model already exists in `models/proxy.joblib`. Read `outputs/proxy_test_streams.csv` for saved test accuracy. For development experiments use `python -m src.proxy --help` in a separate copy. Do not rerun `proxy_final --prepare` or `--final`, or delete the final-evaluation guard, to reproduce saved results. The cached evaluation arrays are retained for report reconstruction.

## Other tracks

[Complete project and learning workbook](https://github.com/Choongmh/DWC-project/tree/main). This branch contains one track and required dependencies. Do not merge main wholesale into it. Large source files use Git LFS; run `git lfs pull` after cloning.

[Original project and contribution credits](docs/ATTRIBUTION.md).
