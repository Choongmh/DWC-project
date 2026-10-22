# T2 — Predicting reservoir production with machine learning

T2 builds a **proxy model**: a machine-learning approximation of a reservoir simulator. Given four reservoir parameters, it estimates cumulative oil, gas and water production over 1998–2008. T3 uses this model to explore different reservoir settings.

## Inputs and outputs

| Input | Meaning |
| --- | --- |
| Fault transmissibility | How easily fluid flows across a fault |
| Porosity multiplier | Adjusts the amount of pore space in the rock |
| Permeability multiplier | Adjusts how easily fluid moves through the rock |
| Aquifer pore volume | Describes the size of the connected water-bearing system |

The output is **123 values**: total oil, gas and water produced up to each of 41 quarterly dates. Forecasting beyond 2008 is a separate T3 step.

## How it works

1. **Start with T1's cleaned data.** The master table contains 100 simulation cases, their four inputs and their production curves.
2. **Compare models on cases 1–70.** Cross-validation compares a mean-curve baseline, Ridge regression, boosted trees and Gaussian Process models. Scaling and curve compression are fitted within each training fold to prevent data leakage.
3. **Validate on cases 71–85.** Check curve accuracy, production rates and how well the model ranks cases against field history.
4. **Fit the final model on cases 1–85.** The selected method uses PCA to compress the curves and Gaussian Process regression with a Matérn 2.5 kernel to predict them. Oil, gas and water are modelled separately.
5. **Evaluate once on cases 86–100.** These held-out cases provide the saved final test results. Export the model for T3.

Training targets are **simulated production curves**. Observed field history is used for scoring and comparison, not as the training target.

## Saved results

| Production stream | Mean test NRMSE |
| --- | ---: |
| Oil | 0.000219 |
| Gas | 0.000203 |
| Water | 0.000460 |

NRMSE is a normalised prediction error; lower is better. Here, errors are scaled by the maximum observed field-history value for each stream. The small errors show close agreement with held-out simulator curves; they do not establish accuracy for future real-world production.

See [detailed test results](outputs/proxy_test_streams.csv) and [test curve comparisons](outputs/figures/proxy_test_curves.png).

## Files

| Location | Purpose |
| --- | --- |
| `data/processed/master_table.parquet` | T1's training table used by the code |
| `data/processed/master_table.csv` | Readable copy of that table |
| `data/processed/history.parquet` | Field history needed for scoring and diagnostics |
| Other prepared tables | Supporting T1 hand-off data |
| `data/processed/proxy_final_evaluation.npz` | Cached final evaluation for rebuilding reports |
| `src/` | Model development, validation, diagnostics and shared helpers |
| `models/` | Saved final model and metadata |
| `outputs/` | Saved comparisons, test results and charts |
| `tests/` | Automated checks of scoring and model behaviour |

**Raw Excel files are not required for T2.** They remain on [T1](https://github.com/Choongmh/DWC-project/tree/t1-data) and [main](https://github.com/Choongmh/DWC-project/tree/main). The shared `src/load_data.py` stays because T2 imports its prepared-table readers. Do not run its raw-data conversion command on this branch.

## Try it locally

Use Python 3.12. From the repository folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

To predict the first case with the saved model, run this Python code:

```python
from src import config as C
from src.load_data import read_master
from src.proxy import predict

table = read_master()
inputs = table.loc[:, C.PARAM_COLS].iloc[[0]].to_numpy()
curves = predict(inputs)
print(curves.shape)  # (1, 123): oil, gas and water, each with 41 dates
```

Use a separate copy for training experiments because development commands overwrite saved reports. Start with `python -m src.proxy --help`. Keep the final evaluation guard intact; do not rerun `proxy_final --prepare` or `--final` over the published model.
