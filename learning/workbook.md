# Reservoir Data Science: A Workbook from Zero

## How to use this workbook

This is a course for a beginner learning the completed DataWorks Challenge project. Read one lesson, do the small calculation, type the exercise yourself, and explain the result aloud. You do not need an AI assistant to complete the exercises. You do need patience: understanding a model is more than being able to run its script.

Use a notebook with four columns: question, what I tried, what happened, and what I learned. When something fails, record the complete error and the smallest example that reproduces it. Look up one unfamiliar function at a time in its installed help. Change one thing at a time. Save your original data unchanged.

English is the main language. Difficult terms have Chinese explanations on first use. For example, infer means 推断: reaching a conclusion from evidence rather than directly observing it. A parameter is 参数: a number controlling a model. An assumption is 假设: something we accept for the calculation but have not necessarily measured.

The four tracks are T1: business understanding and data preparation; T2: simulator proxy; T3: history matching and forecasting; T4: dashboard and communication. These are the team's workflow tracks. They are not four different datasets and should not be confused with the four required submission deliverables.

Suggested learning pace: spend sessions 1–4 on foundations, 5–9 on T1, 10–15 on T2, 16–21 on T3, and 22–26 on T4 and reproduction. Take longer whenever you cannot explain a step in your own words. Completion means you can produce the outputs and defend the choices, not that you have read every page.

# 1. Understand the question before touching Python

## What are we trying to find?

Imagine a reservoir as a complicated underground sponge containing oil, gas and water. We cannot inspect every part directly. Engineers build a physics simulator: a computer program that represents the reservoir and calculates how fluids might move and be produced over time. The simulator is useful but can be expensive to run many times.

The challenge supplies simulations for 100 different reservoir settings. It also supplies real field production history. We want to find settings whose simulated history resembles the observed field, then consider future production. A fast machine-learning proxy, also called a surrogate (代理模型), lets us try many settings without rerunning the full physics simulator.

Write the task in three sentences: First, learn the simulator's response to four inputs. Second, use that learned response to search for settings that match field history. Third, make future production scenarios using an explicit forecasting method. Each sentence requires different evidence. Mixing them is a common source of misleading conclusions.

The business motivation is planning: how much oil and gas might be produced, and how much water must be handled? Profitability requires prices and costs as well. Production forecasts alone cannot answer whether an investment makes money.

## Your first hour on an unfamiliar challenge

1. Read the instructions and list required outputs, formats, dates and evaluation rules. Do not start by choosing a neural network.
2. Make a copy of the raw input files. Record names, sizes and what each workbook describes.
3. Write the prediction question: what is known when making a prediction, and what must be estimated?
4. Identify one row's meaning. Here a modeling row represents one simulated parameter case, not one calendar date.
5. Separate observed history, simulated training targets and future dates. They play different roles.
6. Define a simple reference method, called a baseline (基准). Here it is ranking existing simulator cases against history.
7. Create an assumptions log. Record any choice not explicitly specified by the organizer.

Exercise 1: Explain why the historical field curve must not become an input feature when predicting a new reservoir setting. Then explain why the field curve is still needed later.

Checkpoint: You should be able to draw four boxes on paper: raw files, clean tables, model/search, and forecast/dashboard. Under every arrow, write the exact information being passed.

# 2. Reservoir language without an engineering degree

## The four model inputs

Fault transmissibility, `fault_trans`, describes how easily fluids can cross a fault in the modeled reservoir. Think of a connection between two sponge compartments. A more open connection allows different movement than a restricted connection. It is not automatically a probability or a percentage.

Porosity multiplier, `poro_mult`, scales pore space. Porosity (孔隙度) concerns storage: how much of the rock volume consists of spaces that can contain fluid. A multiplier of 1.2 means scaling a reference property by 1.2, not that the rock is 120% empty.

Permeability multiplier, `perm_mult`, scales ease of flow. Permeability (渗透率) differs from porosity: two sponges can store similar amounts of liquid but release it at different speeds. In this dataset the multiplier applies to specified lower layers, as described in the hand-off.

Aquifer pore volume, `aquifer_pv`, describes the size parameter of the water body connected to the reservoir. Aquifer means 含水层. Avoid assigning an unverified physical unit to this input; consult the nomenclature workbook before interpreting its numeric values physically.

In the existing hand-off, approximate sampled bounds are fault 0.05–0.15, porosity multiplier 0.80–1.49, permeability multiplier 0.52–9.78, and aquifer parameter 54–200. The production code reads the precise bounds from its data/model metadata. Do not replace precise bounds with these rounded teaching values in an experiment.

## Outputs and units

Oil and water cumulative production use STB, stock-tank barrels. Gas cumulative production uses MSCF, thousand standard cubic feet. Their numeric values cannot be added as though they have the same unit. Rate adds a time denominator: STB/day or MSCF/day.

Cumulative (累计) means the amount produced up to a date. Rate (速率) means the amount produced per unit time. A declining daily oil rate can still produce an increasing cumulative oil curve. Think of saving money: even if your monthly deposit gets smaller, your total savings can continue growing.

Exercise 2: A field produces 100 STB/day for 10 days and then 60 STB/day for 10 days. What is total production? Is the cumulative curve decreasing during the second period? Explain.

# 3. Computer and Python foundations

## Files, folders and commands

A path is a file's address. A working directory is the folder from which a command runs. Many errors come from running the correct command in the wrong folder. Open a terminal in `dwc2026-portfolio`, the folder containing `src` and `requirements.txt`.

Python executes instructions in order. A variable names a value. A list stores ordered values. A dictionary maps keys to values. A function packages a calculation with inputs and a return value. An import makes another module's functionality available. A module is usually a `.py` file.

```python
rates = [100, 60]             # a list of daily rates
days = [10, 10]               # a list of durations
total = 0                    # a variable initially holding zero
for rate, duration in zip(rates, days):
    total = total + rate * duration
print(total)
```

Read this aloud: pair each rate with its duration, multiply them, and add the production to the running total. `zip` pairs the first items, then the second items. Indentation marks the instructions inside the loop. The output is 1600.

```python
def production(rate, days):
    return rate * days

assert production(100, 10) == 1000
```

`return` sends the answer back to the caller. `assert` checks a condition and raises an error when it is false. An assertion is useful for checking a known property, but it does not prove an entire model is correct.

## Environment setup

A virtual environment (虚拟环境) separates this project's packages from other projects. The preserved code was verified originally with Python 3.12.4 and the versions in `requirements.txt`. Use that environment for reproduction before experimenting with upgrades.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import pandas; print(pandas.__version__)"
```

The first command creates the environment. The second installs packages. The third confirms which pandas version that interpreter sees. Installation requires package access and may fail offline; that does not mean your data is wrong. If you already have a working project environment, use its Python executable explicitly.

`python -m src.load_data` means run the `load_data` module inside the `src` package. `-m` helps Python resolve package imports consistently. It is different from double-clicking a file in Explorer.

Exercise 3: Change the toy example to three intervals, including a leap-year quarter with 91 days. Calculate the result by hand before running it. Explain every line without reading the comments.

# 4. Tables, arrays and the meaning of one row

A pandas DataFrame is a table with named columns. A Series is one labeled sequence, often a column. A NumPy array is an efficiently stored numeric collection. Shape describes its dimensions: `(100, 4)` means 100 rows and four columns.

```python
import pandas as pd
from pathlib import Path

root = Path.cwd()
master = pd.read_parquet(root / 'data/processed/master_table.parquet')
print(master.shape)
print(master.columns[:8])
print(master.head(3))
print(master['split'].value_counts())
```

Expected shape is `(100, 129)`: two identifier columns, four input columns, and 123 output columns. The split counts are 70 train, 15 validation and 15 test. `head(3)` displays the first three rows. `master['split']` selects one column. `value_counts()` counts category occurrences.

The 123 outputs are 41 oil totals, 41 gas totals and 41 water totals. Their order matters when reshaping predictions. A numeric array has no column labels to save you if oil and gas are accidentally swapped.

Long form has one row for each case and date, useful for charts. Wide form has one row per case and separate output columns for dates, useful for this machine-learning model. The long historical simulation table should have 100 × 41 = 4,100 rows. A join (连接表格) combines related rows using a key such as `case`. A many-to-many join can accidentally multiply rows.

Exercise 4: Calculate 2 + 4 + 3 × 41. Explain each term. Then explain why splitting 4,100 time rows randomly is not the same as splitting 100 cases.

# 5. T1: inventory the raw workbooks

Files 01 and 02 contain the parameter settings and production curves used by the core pipeline. Files 03–08 contain reservoir-grid data. File 09 is the blank output template. Do not assume a workbook named Test Cases should be used for training because a short README describes it loosely. The implementation's fixed split reserves cases 86–100 for final evaluation.

```python
import pandas as pd
book = pd.ExcelFile('data/raw/01 Introduction.xlsx')
print(book.sheet_names)
params = pd.read_excel(book, sheet_name='Uncertainty Values')
print(params.head())
print(params.dtypes)
```

Expected introduction sheets include `Nomenclature` and `Uncertainty Values`. First inspect the sheet names, then the header and data types. Do not automatically load every 300 MB grid workbook into memory just to explore four scalar inputs. The grid conversion is optional in this approach.

Create a data dictionary (数据字典) with these columns: file, sheet, column, meaning, unit, valid range, missing-value rule, and role. A role can be identifier, input, training output, observed reference, or future output. This document prevents accidental leakage and unit confusion later.

The grid sentinel `-999.25` means an inactive cell, not a real negative rock property. Convert it to missing only in the relevant grid context. Replacing every negative number in every file with missing would be an unjustified rule.

Exercise 5: Write the dictionary entries for `case`, `poro_mult`, `oil_cum`, and `RAMP_History`. Which one should identify a row? Which one should be the historical reference rather than a T2 training target?

# 6. T1: dates, missing values and alignment

Alignment (对齐) means comparing quantities at the same dates and under the same definition. If one curve is July production and the other is October production, subtracting them measures a timing mismatch as well as a model mismatch.

The cumulative production sheets contain a code row, a names row, a units row, then data. The loader uses `header=1` because pandas counts rows from zero. It parses Date, removes the non-date units row, and handles each case column separately. Different cases can have values on different simulator dates.

The shared history grid has 41 quarter-start dates from 1998-01-01 through 2008-01-01. The project looks up cumulative values on those dates. For the supplied dataset, the hand-off reports exact observations at all target dates. The code nevertheless contains a fallback interpolation warning. If that warning appears with new data, investigate instead of silently claiming every point was observed.

Interpolation (插值) estimates a value between known observations. Extrapolation (外推) estimates beyond the observed range. Interpolation also requires assumptions. For example, values 100 on day 0 and 200 on day 10 give 150 on day 5 only if you assume a straight line between them.

```python
dates = pd.date_range('1998-01-01', '2008-01-01', freq='QS')
assert len(dates) == 41
assert dates.is_unique
assert dates.is_monotonic_increasing
```

Check duplicate dates, missing case IDs, numeric strings and units before fitting a model. A blank is not automatically zero. Zero may mean no production; blank may mean no observation. In early history, genuine zero-production totals are retained.

Exercise 6: Suppose oil totals are recorded for January and July but not April. List two possible ways to handle April. What evidence would you need before choosing either? Explain why deleting all rows with any blank could remove valid observations from other cases.

# 7. T1: calculate rates correctly

The team found that reported simulator and observed rates described different quarter conventions. It recomputed both from cumulative totals to make the comparison consistent. The convention used throughout the project is the average daily rate for the quarter ending at the labeled date.

Rate at date t = (cumulative at t − cumulative at previous date) / actual days between the dates.

Toy example: cumulative oil is 1,000 STB on January 1 and 1,910 STB on April 1 of a leap year. The increase is 910 STB over 91 days, so the average rate is 10 STB/day. Dividing by 90 would be a small systematic error. Units check: STB divided by days gives STB/day.

```python
toy = pd.DataFrame({
    'date': pd.to_datetime(['2008-01-01', '2008-04-01', '2008-07-01']),
    'oil_cum': [1000., 1910., 2820.]
})
toy['days'] = toy['date'].diff().dt.days
toy['oil_rate'] = toy['oil_cum'].diff() / toy['days']
print(toy)
```

The first rate is missing because there is no preceding observation. The production loader intentionally sets the first derived rate to zero as a boundary convention. That zero is not independent evidence of the preceding quarter's production. Never confuse a convenient boundary convention with a measurement.

Cumulative curves hide some local errors. A small difference between two large cumulative totals may produce a noticeable difference in rates. This is why the project inspects late rates even when cumulative NRMSE is small.

Exercise 7: Totals are 1,000, 1,900 and 2,820 across two intervals of 90 and 92 days. Calculate the two rates. Then add 90 to only the middle cumulative value and recalculate. What happens to neighboring rate errors?

# 8. T1: establish a score and baseline

A residual (残差) is prediction minus reference. If residuals are +10 and −10, their mean is zero even though both predictions are wrong. Squaring removes cancellation. RMSE means root mean squared error: square each residual, average, then take the square root.

Toy calculation: reference [10, 20], prediction [12, 18]. Residuals are [2, −2]; squares are [4, 4]; mean is 4; square root is 2. RMSE is 2 in the original unit. Divide by reference maximum 20 and NRMSE is 0.10. This is normalization (归一化), putting errors on comparable scales.

The team's history score takes the RMSE of each cumulative stream from October 1998 onward, divides by that observed stream's maximum, and averages the three normalized errors equally. Lower is better; zero is a perfect match. The first three dates are omitted from scoring because shared zeros would make accuracy look better without adding useful discrimination.

```python
import numpy as np
reference = np.array([10., 20.])
prediction = np.array([12., 18.])
rmse = np.sqrt(np.mean((prediction - reference) ** 2))
nrmse = rmse / reference.max()
assert np.isclose(nrmse, 0.1)
```

This is a teaching calculation. For project results, use `src.metrics` so all stages apply the same convention. If the reference maximum is zero, do not divide by it blindly; define and document a special rule. The project's actual scoring targets have positive maxima over the scoring period.

The baseline scores all existing simulator cases directly against observed history. It needs no fitted ML model. The saved best case is Case 6, score approximately 0.00355856. This answers, 'How well can an existing supplied case match the field?' It does not answer, 'How accurately does our proxy reproduce the simulator?'

The all-case baseline is descriptive and includes cases from held-out splits. Keep it separate from choosing T2 hyperparameters. Looking at held-out cases during development can weaken claims of complete blindness, even if final model fitting excludes them. A stricter new study would freeze training decisions before exposing test-case diagnostics.

Exercise 8: Oil NRMSE is 0.01, gas 0.02, water 0.03. Calculate the combined score. Explain why '98% accuracy' is a poor replacement for the actual metric name and definition.

# 9. T1: explore relationships without overstating them

Exploratory data analysis, EDA (探索性数据分析), means inspecting distributions and relationships before building a complicated model. Start with ranges, missing values, scatter plots and a few individual curves. Ask which patterns are strong and which may be artifacts of how data was generated.

Sensitivity (敏感性) asks how an output changes when an input changes. A correlation (相关性) summarizes an association in the sampled data. Spearman correlation compares ranks and can capture a monotonic relationship without assuming a straight line. Correlation alone does not establish a causal mechanism.

The original analysis found a strong relationship between porosity multiplier and historical output. Good history matches had porosity multipliers near 1.36, while other parameters varied substantially. This suggests the history constrains porosity more strongly within the sampled settings. It does not prove that other properties never matter or that the underground reservoir is uniquely known.

```python
train = master.loc[master['split'].eq('train')].copy()
summary = train[['poro_mult', 'oil_cum_t41']].corr(method='spearman')
print(summary)
train.plot.scatter(x='poro_mult', y='oil_cum_t41')
```

Use development data for new modeling decisions. Inspect unusual points rather than deleting them because they hurt a score. A true extreme reservoir case may be exactly where the model needs to work.

Exercise 9: If two different input settings produce nearly identical historical curves, can history identify which setting is the real one? Give one reason to keep several plausible settings.

# 10. T2: supervised learning and the correct prediction target

Supervised learning (监督学习) learns a relationship from examples that contain both inputs X and known outputs Y. Here X has four reservoir settings, and Y contains simulator-produced cumulative curves. The model approximates the simulator mapping.

Training (训练) adjusts a model using examples. Prediction uses the fitted model on inputs. Generalization (泛化) means doing well on examples not used for fitting. Memorizing the 70 training rows does not demonstrate generalization to new settings.

```python
from src import config as C
from src.load_data import y_columns

train = master.loc[master['split'].eq('train')]
X_train = train[C.PARAM_COLS]
Y_train = train[y_columns()]
assert X_train.shape == (70, 4)
assert Y_train.shape == (70, 123)
```

`C.PARAM_COLS` fixes the input order. `y_columns()` fixes the target order. Using the project helpers prevents a quiet mismatch between training and prediction. `RAMP_History` is not part of Y_train. It becomes the comparison target in the later search stage.

A simple mean-curve model predicts the same average training curve for every case. A nearest-neighbor model returns a curve from a nearby input setting. A linear model learns additive linear effects. These baselines reveal whether the more complicated model provides useful additional skill.

Exercise 10: Explain X and Y to a friend using a cake recipe analogy. Then identify where the analogy breaks: the reservoir simulator produces many time outputs, and the real field may differ from the simulator.

# 11. T2: split data and prevent leakage

Data leakage (数据泄漏) occurs when information unavailable at genuine prediction time influences training or selection. It makes evaluation look better than the model's real performance. A common example is fitting a scaler on all rows before cross-validation.

The fixed split uses cases 1–70 for development, 71–85 for validation, and 86–100 for the final test. During development, use five-fold cross-validation within the first 70 cases. In each round, fit on approximately 56 cases and evaluate on the remaining 14. Repeat until every training case has a held-out prediction.

Keep every date from a case together. If January from Case 1 is in training and April from Case 1 is in validation, the validation example shares the exact reservoir settings and strongly related outputs. It is not a clean test of a new parameter case.

Validation (验证) supports development checks and model selection. Final testing (最终测试) measures the frozen configuration on untouched cases. Once test results influence changes, those cases are no longer an untouched final test for the modified system.

The final project fitted the chosen configuration on 85 cases after selection and evaluated the 15 test cases once. It saved guard files to prevent casual repeated evaluation. Preserve those files. You can study the saved reports repeatedly, but do not tune a new model to their case-level mistakes and still call the same test independent.

Exercise 11: Which is leakage: fitting PCA on all 100 targets before a split, or fitting PCA only on each cross-validation training fold? Explain why unsupervised preprocessing can still leak information.

# 12. T2: build a simple model yourself first

Scaling (缩放) makes inputs comparable in numerical magnitude. Aquifer values around 100 would otherwise dominate a Euclidean distance compared with fault values around 0.1. Standard scaling subtracts the training mean and divides by the training standard deviation.

Ridge regression is a useful first model: it fits linear coefficients but penalizes very large coefficients. The penalty is regularization (正则化), a way to discourage an excessively flexible fit. It is not the final model selected in this project; this exercise is for learning the full fit/predict loop.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

baseline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(baseline, X_train, Y_train, cv=folds)
assert oof.shape == Y_train.shape
```

`make_pipeline` links preprocessing and regression so each fold fits its own scaler. `alpha` controls the penalty. `random_state=42` makes the shuffle reproducible (可重复). `oof` means out-of-fold predictions: each row was predicted by a model that did not train on that row.

Do not judge this model using predictions on the same rows it fitted. Compare out-of-fold errors using the same metric across candidate methods. Record configuration, split, metric and runtime in a simple experiment table. Never invent results for a model you have not run.

Exercise 12: Run the same pipeline with alpha 0.1 and 10 on development folds. Record results before choosing. Explain why comparing 100 arbitrary settings on the same validation set can overfit the validation process itself.

# 13. T2: why PCA compresses curves

Principal component analysis, PCA (主成分分析), replaces many correlated outputs with a smaller collection of patterns. Forty-one points on a smooth curve often move together. Rather than learning each point independently, we can learn how strongly a curve uses a few common shapes.

Think of describing a song by a few instruments and their loudness. The mean curve is the background. Each component is an additional shape, and each component score tells you how much of that shape to add. This is only an analogy: PCA components are numerical directions, not necessarily physical mechanisms.

The reconstruction is approximately: mean curve plus score 1 × component 1 plus score 2 × component 2, and so on. Retaining fewer components simplifies prediction but loses some curve detail. Retaining too many may add noise or make the model harder to learn.

```python
from sklearn.decomposition import PCA
oil_cols = [c for c in y_columns() if c.startswith('oil_cum_')]
oil = Y_train[oil_cols].to_numpy()
pca = PCA(n_components=3)
scores = pca.fit_transform(oil)
reconstructed = pca.inverse_transform(scores)
print(scores.shape, reconstructed.shape)
print(pca.explained_variance_ratio_.sum())
```

The example compresses 70 × 41 to 70 × 3, then reconstructs 70 × 41. It illustrates the operation, not the final model's complete preprocessing. The supplied final model retains six oil components, six gas components and seven water components. In model comparison, PCA must be fitted inside the training fold.

Exercise 13: Reconstruct training curves with 1, 3 and 6 components. Compare reconstruction error. Why is small reconstruction error alone insufficient to show that component scores can be predicted accurately for unseen inputs?

# 14. T2: Gaussian processes in plain language

A Gaussian process, GP (高斯过程), predicts by modeling similarity between inputs. Nearby parameter settings are expected to have related outputs, with the meaning of nearby controlled by a kernel (核函数). A length scale controls how quickly the relationship changes as an input changes.

Imagine placing flexible rulers through known examples. A very long length scale creates slow variation; a very short one allows rapid variation. If the model is too flexible, it may fit individual examples while behaving poorly between them. The training procedure estimates suitable settings, but the validation still has to test whether those choices work.

The selected model uses a Matérn kernel with smoothness 2.5 and separates oil, gas and water models. It predicts compressed curve representations and reconstructs complete historical curves. Gas is fitted separately; it is not obtained by multiplying oil by a fixed ratio, even though the data shows a strong relationship between them.

GP uncertainty expresses the model's uncertainty under its assumptions. It is not automatically the full uncertainty about the real reservoir or the future. PCA truncation, simulator mismatch, model specification and future operational changes can contribute additional error.

For a first implementation, use a library GP on one scalar target such as oil cumulative at January 2008. Learn its `fit`, `predict`, kernel and standard-deviation outputs. Only then study the multi-output PCA architecture in `src/proxy.py`. This progression makes debugging possible.

Exercise 14: Explain the difference between a parameter case outside the input bounds and a future date after 2008. Why are these two different forms of extrapolation? Why does the supplied historical GP not automatically solve either one?

# 15. T2: evaluate and freeze the model

Read `outputs/proxy_cv_comparison.csv` for development comparisons and `outputs/proxy_test_streams.csv` for saved final test results. Understand the comparator before quoting any metric: proxy predictions are compared with simulator curves here, not with observed field history.

The supplied final report records mean test NRMSE approximately 0.00021935 for oil, 0.00020336 for gas and 0.00045975 for water. These are historical simulator emulation results. They cannot be converted into a claim of equivalent 2028 forecast accuracy.

Inspect per-case errors as well as means. A good average can hide a weak region. Inspect residuals over time: a model can reproduce cumulative totals but get the final slope wrong. Inspect physical plausibility: totals should be finite and nonnegative; cumulative production should not decrease without a clearly explained correction convention.

```python
from src.proxy import load_proxy, predict, param_bounds
load_proxy()
bounds = param_bounds()
example = X_train.iloc[[0]]
predicted = predict(example)
assert predicted.shape == (1, 123)
assert np.isfinite(predicted).all()
```

This loads the preserved artifact and predicts a training input for an interface demonstration. It is not a new accuracy evaluation. Serialized Python model files can execute code on load, so use the project's trusted artifact rather than arbitrary downloaded joblib files.

Freeze (冻结配置) means record the chosen preprocessing, kernel, component rule, random seed and training population, then stop changing them before final testing. Save package versions and a model hash so future results can be traced to the exact artifact.

Exercise 15: A model has a very high R-squared but poor late water rates. What extra plots would you request? Explain why R-squared and correlation do not replace domain checks.

# 16. T3: turn a forward model into a search

The forward problem asks: given reservoir parameters, what curve does the simulator produce? The inverse problem (反问题) asks: given an observed curve, which parameters could have produced it? An inverse problem can have many plausible answers.

T3 uses the frozen proxy as a fast forward model inside a search. It draws 50,000 parameter sets uniformly within the permitted bounds, predicts historical curves, and scores those curves against the observed field. Uniform sampling means equal density within each input range, not equal probability that every reservoir is physically correct.

```python
from src.optimise import sample_candidates, score_candidates
from src.load_data import read_history

X_candidates = sample_candidates(bounds, n=100, seed=42)
history = read_history()
scores = score_candidates(X_candidates, history)
print(scores.min())
```

Use 100 candidates to understand the steps; the saved project uses 50,000. This mini-exercise does not reproduce its best score. A random seed repeats the sampled numbers, but it cannot guarantee the search finds the global optimum (全局最优解).

The best saved generated score is about 0.00345414. Simulator Case 6 scores about 0.00355856. The difference is approximately 0.00010442. That difference is smaller than the project's heuristic tie band of 0.0003, and the proxy itself has approximation error. The evidence does not establish a clear improvement over Case 6.

Exercise 16: Why can an optimizer exploit a proxy's errors? Imagine the proxy underestimates errors in one corner of the input space. What would the search tend to select? What independent verification would help?

# 17. T3: select ten useful scenarios

The project first keeps candidates scoring within 0.0003 of the best generated score. It then selects diverse settings inside that pool. These are ten near-best matches, not necessarily the ten numerically smallest scores.

Diversity (多样性) avoids selecting ten nearly identical points. The code scales permeability, aquifer and fault parameters to a common zero-to-one range, selects the best-scoring candidate first, then repeatedly selects the point farthest from its nearest already selected point. Porosity is excluded from this distance calculation in the supplied code.

Toy example: three acceptable points lie at 0.10, 0.11 and 0.90 on a scaled axis. If the first selection is 0.10, choosing 0.90 adds more diversity than choosing 0.11. This does not mean 0.90 is more probable. It means the selected set covers a wider range among acceptable candidates.

```python
picks = pd.read_csv('outputs/t3_top10_params.csv')
assert len(picks) == 10
print(picks[['id', 'poro_mult', 'score']])
assert picks['id'].is_unique
```

The new IDs 1–10 identify selected scenarios. They are not the original simulator Case 1–10. Keep this distinction in tables, joins and slide captions. The original simulator best case is Case 6; selected scenario ID 6 is a different thing.

Exercise 17: Explain why distance must be scaled before comparing a fault range around 0.1 with an aquifer range around 100. What would happen if you used their raw units?

# 18. T3: forecast oil and gas beyond 2008

The historical proxy stops at January 2008. The future forecast is a separate extrapolation, not 80 extra GP outputs. The project fits decline curves to selected historical rates from January 2003 through January 2008, then extends those curves forward.

Exponential decline has the form q(t) = qi × exp(−D × t). Harmonic decline has q(t) = qi / (1 + D × t). Here qi is a starting rate, D controls decline, and t is elapsed time in units consistent with D. The implementation uses years for time. Hyperbolic decline adds a shape parameter b; harmonic corresponds to b = 1.

Worked example: qi = 1,000 STB/day, D = 0.1 per year. At five years, harmonic rate is 1,000 / 1.5 = 666.67 STB/day. Exponential rate is approximately 606.53 STB/day. The difference grows as the forecast horizon grows, illustrating why model choice matters over 20 years.

Fitting (拟合) chooses parameters that make the curve resemble the historical rates. The implementation tries several initial guesses and keeps the successful fit with the smallest squared residual sum. Bounds stop the optimizer from choosing some unreasonable parameter values, but bounds are themselves assumptions.

```python
t = np.array([0., 1., 5., 20.])
harmonic = 1000. / (1. + 0.1 * t)
exponential = 1000. * np.exp(-0.1 * t)
print(pd.DataFrame({'year': t, 'harmonic': harmonic,
                    'exponential': exponential}))
```

This toy example uses invented numbers to teach the equation. Actual forecast parameters come from the project fits. A fitting window that includes startup and plateau behavior may produce a different decline estimate from a mature-production window. Test the choice explicitly rather than describing it as a law of nature.

Exercise 18: If the same formula uses t in days but D remains per year, what happens? Calculate the five-year harmonic rate with the wrong time unit to see the scale of the mistake.

# 19. T3: water, ratios and the liquid cap

Water-oil ratio, WOR, is water rate divided by oil rate. If water is 600 STB/day and oil is 400 STB/day, WOR is 1.5. Water cut is water divided by total liquid: 600 / 1,000 = 0.6. WOR and water cut describe related quantities but are not the same number.

The selected water method fits log(WOR) as a straight-line function of cumulative oil. Log means logarithm (对数). Transforming a positive ratio into log space lets the fitted ratio grow multiplicatively while remaining positive after exponentiation. The code uses small positive floors to avoid taking log(0).

For forecasting, calculate future oil rate and cumulative oil first, use cumulative oil to predict WOR, then multiply predicted WOR by oil rate to obtain water rate. This makes water errors depend partly on oil errors. Evaluating water while supplying the true future oil curve isolates only part of the full problem.

The main scenario limits water to the remaining liquid-rate capacity after oil, using a cap of 13,247.5803 STB/day. The cap comes from nine quarter-ending observations from January 2006 through January 2008 inclusive. It is a modeling assumption inferred from recent history, not a supplied engineering facility capacity.

Toy calculation: if the cap is 1,000, oil is 400 and predicted water is 900, remaining capacity is 600, so capped water is 600. If oil alone exceeds the cap, clipping water to zero does not enforce a total cap on oil. Inspect that edge case rather than assuming the water-only rule solves it.

Exercise 19: Explain how a water cap can dominate a late forecast even when the fitted WOR model looks sophisticated. Why should an uncapped comparison remain available?

# 20. T3: convert rates into the submission workbook

The forecast begins from observed cumulative totals at January 2008. For each future quarter, multiply its predicted daily rate by the actual number of days and add that amount to the previous cumulative total. This anchors (锚定) the forecast to the measured history at the transition date.

The code uses quarter-ending predicted rates as interval averages in this accumulation. This is a numerical modeling convention, not exact analytical integration of a continuously declining function. A smaller time step or exact integration could give different totals. State the convention when reproducing results.

```python
start = pd.Timestamp('2008-01-01')
dates = pd.date_range('2008-04-01', '2028-01-01', freq='QS')
previous = start
cumulative = 1000.0
for date in dates[:2]:
    quarter_rate = 10.0
    cumulative += quarter_rate * (date - previous).days
    print(date.date(), cumulative)
    previous = date
```

The template uses columns Date, Gas, Oil, Water. The model's historical output order is Oil, Gas, Water. Do not copy arrays positionally without checking the schema. The blank Case 1 template is shorter than the required horizon; the supplied writer extends it to 80 future quarters.

Ten cases × 80 dates × three streams = 2,400 production values. Validate all of them, not only the last row. Check exact dates, unique rows, finite values, nonnegative values, monotonic cumulative totals, units and scenario selection. The submitted scenario is `main`; `rate_calibrated` and `uncapped` are sensitivity comparisons.

Exercise 20: Why should 2028 cumulative production include the amount already produced by 2008? How would you calculate only the additional production during the forecast period?

# 21. T3: backtesting and uncertainty

Backtesting (回测) pretends you are at an earlier date, fits using information available then, and compares the forecast with later known observations. A cold test is useful only to the extent that its information boundary matches the real prediction problem.

The existing water cold test uses future proxy oil to isolate water-model error. The selected cases also came from matching the complete history. Therefore this is a conditional diagnostic, not a clean independent evaluation of the entire future forecasting process. A stronger new backtest would redo selection and fitting using only the earlier information, then forecast both oil and water without future inputs.

A three-year backtest cannot establish 20-year accuracy. Future operations, model mismatch and fitting assumptions can matter more over a longer horizon. Keep assumptions visible rather than hiding them behind a narrow shaded band.

The project labels P90 as the low, 10th-percentile value; P50 as the median; and P10 as the high, 90th-percentile value. This follows its exceedance-style naming convention. Ordinary statistical percentile labels often use the opposite numeric direction, so always state the convention.

```python
values = np.array([10., 20., 30., 40., 50.])
low, median, high = np.percentile(values, [10, 50, 90])
print({'P90_low': low, 'P50': median, 'P10_high': high})
```

The output is 14, 30 and 46 with linear interpolation. These are descriptive summaries of these five numbers. The project's ten selected cases were chosen deliberately, not sampled from a calibrated probability distribution, so their quantiles are not guaranteed probabilities about the real future.

Exercise 21: Explain the difference between parameter uncertainty and method uncertainty. Which changes when you keep ten parameters fixed but remove the water cap? Is a median necessarily the most likely outcome?

# 22. T4: prepare data for Power BI

Power BI has three distinct jobs: Power Query transforms data, the model relates tables, and visuals summarize the result. DAX calculates measures in the current filter context (筛选上下文). A chart can look attractive while aggregating the wrong rows.

The recovered original report refers to a `Master Production` table and a `t3_top10_params` table. Its visible page is named Final; Page 1 is hidden. The saved card definitions use maxima of cumulative columns. A maximum across selected cases is not the median across cases.

For a clean learning rebuild, import `outputs/t3_forecast_all.csv` with Get Data > Text/CSV, then choose Transform Data. Set `date` to Date, `id` to Whole Number, `scenario` to Text, and rate/cumulative columns to Decimal Number. Set decimal and date interpretation explicitly if the computer's locale reads values differently.

Filter to `main` for the main dashboard or retain the scenario column and add a single-select scenario slicer. Without this, oil totals from several alternative scenarios can be mixed. Import `outputs/t3_top10_params.csv` as a case dimension with one row per id. Create a one-to-many relationship from that id to the forecast id. Prefer a single filter direction unless you can explain why a more complicated relationship is necessary.

Keep field history in a separate table. Do not duplicate the same observed history ten times and then sum it across cases. That creates ten times the actual field volume. If combining historical and future lines, create an explicit series type and aggregation rule.

For the competition's Excel-source requirement, import the ten filled Case sheets, add a case identifier to each before appending them, and standardize the three production column names. Parse the text dates using the month/day/year convention. The CSV route is convenient for learning and additional scenario analysis; the filled Excel remains the submitted output.

Exercise 22: Describe the grain (数据粒度) of the forecast table: one row per what? Which combination of columns should uniquely identify a row? Why is id alone not a unique key there?

# 23. T4: build measures that answer the right question

Rename the imported forecast query to `Forecast` in a learning report so the following examples match. A measure computes a result when a visual is evaluated. A calculated column stores a row-level result. Use a measure for a summary that should respond to filters.

```text
Oil P50 at visible date =
MEDIANX(
    VALUES(Forecast[id]),
    CALCULATE(MAX(Forecast[oil_cum]))
)
```

This measure takes one cumulative value per case and then the median across cases. It assumes the visual context contains exactly one date and one scenario. Put date on the line chart axis and set a single scenario. At a context containing many dates, MAX would choose the largest date value per case, which is a different question.

```text
Oil P50 2028 main =
CALCULATE(
    [Oil P50 at visible date],
    REMOVEFILTERS(Forecast[date]),
    REMOVEFILTERS(Forecast[scenario]),
    Forecast[date] = DATE(2028, 1, 1),
    Forecast[scenario] = "main"
)
```

This second measure fixes the date and scenario for a headline card while retaining case filters. If you introduce a separate date dimension, its filters may also need clearing for a fixed-date card. The exact model relationship determines filter behavior, so test rather than copying formulas blindly.

Never sum cumulative production over dates: January already includes earlier production, and April includes January. Summing them double-counts production. Never sum alternative reservoir scenarios as if they were separate producing fields. Separate oil and gas panels because their units differ.

Exercise 23: Two cases have 2028 oil totals 50 and 70 million STB. What are their sum, maximum and median? Which describes the median scenario? If the card shows 120 million, what aggregation error is likely?

# 24. T4: design and validate the dashboard

Build one overview page first. Use a clear title stating that these are forecast scenarios. Add three fixed-date median cards, one per stream with its unit. Add separate cumulative line charts with date on the horizontal axis and case id as the series. Add a case selector, scenario selector and a table of selected parameters and history scores.

Create a second page for assumptions. Compare main and uncapped water, show the cap value, and explain why the curves differ. Include a note that scenario quantiles are not calibrated probabilities. An explanatory subtitle is more useful than a decorative gauge with no decision purpose.

Validate with three deliberate selections: one case, two cases and all ten cases. For one case, the card should match the CSV. For all ten, the median should match a separately calculated median. For two, verify the median by hand. Check that changing a slicer affects exactly the intended visuals.

The original Final page is very wide, 4,250 by 1,800. A future presentation version may be easier to read in a more conventional aspect ratio. Preserve the original file as evidence while rebuilding a learning version with readable labels. The recovered PBIX must be opened and refreshed in Power BI to verify its native behavior; JSON inspection alone cannot establish that refresh works.

```python
fc = pd.read_csv('outputs/t3_forecast_all.csv', parse_dates=['date'])
end = fc.loc[fc['scenario'].eq('main') &
             fc['date'].eq(pd.Timestamp('2028-01-01'))]
print(end[['oil_cum', 'gas_cum', 'water_cum']].median())
```

Expected medians are approximately 59,747,756.35 STB oil, 21,706,343.67 MSCF gas and 87,596,481.91 STB water. Use the unrounded CSV values for validation; round only for presentation.

Exercise 24: Make a dashboard test checklist containing expected values and observed values. A screenshot alone is not enough: include the filters used to obtain each number.

# 25. T4: turn analysis into a defensible story

A useful presentation answers a sequence of audience questions. What problem matters? What information was available? What did you build? How was it checked? What did you find? What remains uncertain? What should happen next?

For a seven-minute explanation, allocate about 40 seconds to the business question, 50 to data preparation, 80 to the proxy and its validation, 90 to matching and forecasting, 90 to the dashboard and assumptions, and 70 to limitations and next steps. Rehearse with a timer and shorten explanations that need more time than the audience has.

Use one main message per slide. For example: 'The proxy reproduces held-out historical simulator curves accurately' is a different slide from 'Future water production depends strongly on the liquid cap.' Put the metric definition beside the relevant result. Keep units on axes and distinguish observed, simulator, proxy and extrapolated curves visually.

Avoid unsupported claims. Say 'median of selected scenarios' rather than 'most likely forecast'. Say 'proxy-estimated history score' rather than 'simulator-verified improvement'. Explain that the cap used nine observations, correcting the older slide's eight-quarter wording. Do not conclude profitability without economic inputs.

An honest portfolio introduction can explain that the original challenge was a team submission, your original focus was T2, and you subsequently studied and reproduced the full workflow. Demonstrate your new personal understanding with your own exercises, debugging notes and improvements. Renaming teammates' code does not demonstrate that understanding.

Exercise 25: Explain the entire project in 60 seconds without using GP, PCA, NRMSE or WOR. Then explain it in three minutes using those terms correctly. Ask a listener to repeat the main limitation back to you.

# 26. Reproduce the project without relying on AI

Start by copying the organized project into an experiment folder. Keep the preserved outputs unchanged in the reference folder. Record your Python and package versions, then run data preparation. Compare table shapes and date counts with this workbook before running any model.

Run baseline and sensitivity next. Inspect Case 6, the ranking and at least one residual plot. Then load the frozen proxy for a small prediction. Use the saved evaluation reports for the original test result. Only after those checks run the parameter search and selected-curve generation.

Run `forecast_all`, then `fill_template`, then the quantile and cap-sensitivity scripts. Compare the generated workbook with the preserved submission. Small fitting differences across versions may occur, so explain numeric tolerances instead of requiring identical ZIP bytes for a freshly generated Excel file.

To learn model training yourself, work only with the development cases in a separate copy. Start with the mean-curve baseline, then Ridge, then scalar GP, then PCA plus GP. For every experiment, record what changed, why you expected it to help, how you evaluated it and what actually happened.

When stuck, read the traceback from the bottom upward to locate the failing operation. Print the shape, column names and a tiny sample immediately before that operation. Check whether the input type matches the function's expectation. Search local source for the function definition and read its docstring. Use `help(function)` or `python -m module --help` when supported.

Exercise 26: Deliberately rename a column in a toy copy, run a selection that expects the old name, and explain the error. Restore it. Then deliberately shuffle the rows of one toy table and demonstrate why joining by case is safer than combining by row position.

# 27. GitHub and four task branches

Git stores snapshots of files. A commit is a recorded snapshot with a message and author. A branch is a movable name for a sequence of commits. A remote is a named repository location. GitHub hosts repositories but does not automatically contain files merely because they exist locally.

This independent portfolio uses `main` for the integrated version and four focus branches: `t1-data`, `t2-proxy`, `t3-forecast`, and `t4-storytelling`. They start from the same complete snapshot, allowing each to run while future improvements remain organized by topic. They are not manufactured evidence that all original work was done solo.

```powershell
git status
git switch t1-data
git diff
git add docs/your-learning-notes.md
git commit -m "Explain date alignment and rate convention"
```

Only run the add command after creating the named notes file. Inspect `git diff` before committing. Use a descriptive message that records an actual change. Avoid committing a virtual environment or credentials. The raw datasets are included in this edition on the owner's confirmation; large workbooks are configured for Git LFS.

Git LFS stores a small pointer in ordinary Git history and transfers the large content separately. Downloading a repository without its LFS objects may leave pointer text instead of a usable workbook. After cloning, use `git lfs pull` and verify the large files exist. Do not assume a successful Git push proves all LFS content was uploaded.

Exercise 27: Explain why copying a folder does not create a public GitHub repository. List what must be verified after publication: account, visibility, branch names, source files, LFS objects and README links.

# 28. Independent capstone and self-assessment

Complete the following in a fresh experiment folder without asking AI to write the steps for you. You may consult this workbook, source docstrings and library help. Keep a log of decisions so another student can follow your reasoning.

Milestone A: produce a one-page problem statement, data dictionary and file inventory. Explain the different roles of simulator curves and observed history. Show the required 41 historical and 80 future dates.

Milestone B: rebuild the clean table and manually verify one quarter's rate. Calculate the toy NRMSE by hand. Reproduce the baseline ranking and explain why the scoring definition matters.

Milestone C: train a simple development-only model with fold-safe preprocessing. Produce out-of-fold predictions and explain one error plot. Compare your model with a mean-curve baseline before trying the more complicated GP architecture.

Milestone D: use the frozen historical proxy to search a small candidate pool. Explain the difference between best numerical score and diverse near-best cases. Forecast a toy oil curve, then show how the actual project's water cap changes an outcome.

Milestone E: create the Power BI learning report, validate one-case and all-case numbers, and deliver a seven-minute presentation. State limitations without being prompted. Record exactly which components you independently reproduced and which were reused.

Self-assessment rubric: score each milestone 0 if you cannot explain it, 1 if you can run it by copying, 2 if you can explain and check it, and 3 if you can implement a small variation and predict its effect. Revisit every milestone below 2. The goal is durable understanding, not a large repository.

# 29. Exercise answers and reasoning

1. A new setting provides the four parameters, not a new measured field curve. T2 learns simulator responses. The observed history is later used to score plausible settings for the real field.

2. Total is 100 × 10 + 60 × 10 = 1,600 STB. The total keeps increasing because both rates remain positive, although its slope becomes smaller.

3. There is no single answer because you choose the third interval. Check each rate × days product and add them. The key is distinguishing the length of a quarter from the quarter's label.

4. There are 2 identifiers, 4 features and 123 targets, totaling 129. Random date-row splitting leaks closely related observations from the same case across partitions.

5. `case` identifies the simulation. `poro_mult` is an input multiplier. `oil_cum` is an output total with STB units. `RAMP_History` identifies the observed comparison curve and is not a T2 training target.

6. You could flag the missing point and exclude or halt the affected comparison, or interpolate under a justified rule. You need the intended date convention, surrounding observations and permission to estimate. Dropping every row containing any blank may discard valid data from other case columns.

7. Original rates are 900/90 = 10 and 920/92 = 10. Perturbing the middle total gives 990/90 = 11 and 830/92 ≈ 9.02174. Differencing spreads the one-point perturbation into opposite-signed adjacent rate errors.

8. Combined score is (0.01 + 0.02 + 0.03)/3 = 0.02. This is normalized RMSE, not a classification success rate or a generic accuracy percentage.

9. No unique identification is possible from indistinguishable histories alone. Several plausible settings help explore non-uniqueness, though selecting them does not create a probability distribution.

10. Ingredients and oven settings can act as inputs and a measured cake property as output. Here Y is a whole collection of simulator curves, and observed field behavior is a separate reference that the simulator may imperfectly represent.

11. Fitting PCA on all 100 targets leaks held-out output structure. PCA inside each training fold keeps the held-out distribution out of the fitted transformation.

12. The numeric results depend on the run and must be recorded, not guessed. Repeatedly selecting what works best on one validation set adapts the development process to that set's random peculiarities.

13. More components normally improve training reconstruction, but that says nothing by itself about predicting the scores of unseen cases. Separate reconstruction loss from regression and generalization error.

14. One extrapolation leaves the sampled parameter domain. The other leaves the historical time horizon. The proxy has learned outputs on the historical dates and only has training evidence in its input domain.

15. Request per-case residuals, late-rate comparisons, water-only errors, and physical-plausibility checks. A large overall variance can make R-squared look strong while important local behavior is wrong.

16. Search selects apparently good predictions, including erroneous ones. Independent simulator runs on new candidates would test whether the improvement is real. More samples from the same biased proxy cannot provide that verification.

17. Raw Euclidean distance would be dominated by the numerically larger aquifer differences. Scaling makes distance reflect relative position within each range.

18. Five years is roughly 1,826 days. Substituting 1,826 into a formula with D = 0.1 per year gives about 5.45 STB/day rather than 666.67. The time unit is inconsistent.

19. Once the water prediction exceeds remaining capacity, capped water is determined by cap minus oil. The unconstrained WOR prediction no longer determines that output. An uncapped scenario reveals how much the assumption changes the forecast.

20. Cumulative means all production to the date. Additional 2008–2028 production is 2028 cumulative minus the January 2008 cumulative anchor.

21. Parameter uncertainty varies reservoir settings; method uncertainty varies forecasting assumptions. Removing the cap changes method uncertainty. A median is the middle value; a mode is the most frequent or highest-density value, so they need not coincide.

22. One forecast row is one selected case id, one scenario and one date. Their combination should be unique. Id alone repeats across dates and scenarios.

23. Sum is 120, maximum 70 and median 60 million STB. The median summary is 60. A card showing 120 likely summed mutually exclusive scenarios as though they were additive fields.

24. A good checklist states the date, scenario, selected ids, expected aggregation and numeric tolerance. Compare to the CSV or a hand calculation. Exact visual formatting is separate from numerical correctness.

25. A good short explanation distinguishes historical emulation, matching and future extrapolation. The listener should understand that a narrow scenario spread is not proof of a reliable forecast.

26. A missing named column should raise a KeyError. Joining on an explicit key preserves the relationship despite row order, whereas positional concatenation can silently pair the wrong cases.

27. Publication requires an actual remote repository and successful transfer. Verify the account, visibility, four branches, regular files and LFS objects. A local commit is only local until it is pushed successfully.

# 30. Glossary and code-reading map

Aggregate 聚合: combine rows using a defined operation such as sum or median. Align 对齐: match observations by the same key or date. Anchor 锚定: start a forecast from a specified measured total. Approximate 近似: represent something with limited precision. Calibrate 校准: adjust a model against evidence, with a stated purpose and evaluation boundary.

Constrain 约束: restrict allowable values. Derive 推导: calculate a quantity from others. Diagnose 诊断: investigate why behavior occurs. Emulate 模拟近似: reproduce the response of another system. Evaluate 评估: compare results under a declared criterion. Extrapolate 外推: predict beyond observed coverage.

Feature 特征: input used for prediction. Generalize 泛化: work on unseen cases. Infer 推断: conclude from evidence. Integrate 积分或累加: accumulate rates over time into volumes. Normalize 归一化: express a value relative to a scale. Optimize 优化: search for a better value of an objective.

Residual 残差: prediction minus reference. Robust 稳健: relatively insensitive to reasonable changes or disturbances. Scenario 情景: an outcome under specified settings and assumptions. Surrogate 代理模型: a cheaper approximation to another model. Validate 验证: check performance or consistency under specified conditions.

Read source in this order: `config.py` for names and conventions; `load_data.py` for cleaning; `metrics.py` for scoring; `baseline.py` and `sensitivity.py` for T1; `proxy.py` for the architecture; `proxy_validation.py` and `proxy_final.py` for the evaluation process; `optimise.py` for the search; `t3_curves.py` for selected historical curves; `forecast_all.py` for extrapolation; `fill_template.py` for the deliverable; and `p10_p50_p90.py` plus `water_cap_sensitivity.py` for scenario summaries.

For each function, write: what goes in, what comes out, what can go wrong, and what assumption it contains. Start with a small function such as `_integrate`, `sample_candidates`, or `add_rates`. Do not try to memorize thousands of lines before understanding one complete input-to-output path.

# 31. Sources and further reading

Project-specific facts and numeric examples come from the final submission README, DEVELOPMENT_NOTES, source modules, saved CSV outputs, T1 hand-off, original presentation and extracted Power BI visual definitions included in this project. The final submission notes take precedence over older slide wording when they explicitly record a correction. Toy examples in this workbook use deliberately simple invented numbers and are labeled as teaching examples.

Original team repository: https://github.com/khoonlyn913/dwc2026-internlvl. The local portfolio source map records which available files were used; remote completeness must still be checked after authenticated access is available.

For independent study, use the Python tutorial, pandas user guide, scikit-learn user guide, and Microsoft Learn Power BI documentation. Read the documentation matching the installed library version when APIs differ. Within the installed environment, `help(pd.read_excel)`, `help(StandardScaler)` and `help(PCA)` provide function details without requiring an AI assistant.

Keep this final question beside your computer: 'What evidence would show that my explanation is wrong?' A strong data scientist can answer it for the data, the model, the forecast and the dashboard.
