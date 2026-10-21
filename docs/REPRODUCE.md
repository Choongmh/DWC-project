# Reproduce and learn

Run from the project root. Use a separate working copy for experiments because scripts overwrite results. Keep the archived submission outputs for comparison.

The supplied requirements were recorded for Python 3.12.4. They are direct dependency pins, not a complete transitive lockfile.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.load_data
.\.venv\Scripts\python.exe -m src.baseline
.\.venv\Scripts\python.exe -m src.sensitivity
.\.venv\Scripts\python.exe -m src.optimise
.\.venv\Scripts\python.exe -m src.t3_curves
.\.venv\Scripts\python.exe -m src.forecast_all
.\.venv\Scripts\python.exe -m src.fill_template
.\.venv\Scripts\python.exe -m src.p10_p50_p90
.\.venv\Scripts\python.exe -m src.water_cap_sensitivity
.\.venv\Scripts\python.exe -m pytest -q
```

The frozen T2 model and its saved test reports are included. Do not rerun `proxy_final --prepare` or `--final` or delete its evaluation guard to reproduce the forecast. Learning experiments should use development cases in a separate copy. Reading saved test reports is different from using test data to tune a new model.

For Power BI, open `dashboard/exon_recovered.pbix`, review data-source paths, point them to this project's outputs, and validate the filters before trusting cards. Native report open/refresh is not yet verified.

The complete local project includes the large grid workbooks. They must stay in Git LFS, not normal Git blobs.
