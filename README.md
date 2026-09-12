# xgb2sql

Convert a trained tree model into SQL or SAS for scoring outside Python - no model-serving runtime needed at inference time.

## Supported today

| Source model | -> SQL | -> SAS |
|---|---|---|
| XGBoost (`Booster`, `booster="gbtree"`) | Yes - verified against DuckDB (28 tests) | Not yet - see "Why xgboost has no SAS output yet" below |
| `DecisionTreeRegressor` / `DecisionTreeClassifier` | Yes - verified against DuckDB | Yes - see "SAS caveat" below |
| `RandomForestRegressor` / `RandomForestClassifier` | Yes - verified against DuckDB | Yes - see "SAS caveat" below |
| `GradientBoostingRegressor` (`loss="squared_error"`) | Yes - verified against DuckDB | Yes - see "SAS caveat" below |
| `GradientBoostingClassifier` (binary, `loss="log_loss"`) | Yes - verified against DuckDB | Yes - see "SAS caveat" below |

Multiclass: supported for `DecisionTreeClassifier`/`RandomForestClassifier` (both languages) and XGBoost (SQL only). Multiclass `GradientBoostingClassifier` is not supported (raises `NotImplementedError`) - it needs a per-round softmax across classes that hasn't been built yet.

Not supported anywhere yet: LightGBM, CatBoost, `HistGradientBoosting*` (different tree representation from plain sklearn CART), `ExtraTrees*`.

## Why a dedicated package instead of an existing one (SQL side)

Closest prior art: `gbm2sql` (one-commit demo), `sqlgbm` (README says "not ready for production"), the R-only `xgb2sql` package of a similar name, and `m2cgen` (general model-export tool, 3,000 stars, but no release since April 2022 and doesn't target SQL). None is a real substitute. See `existing_package_mapping.md`.

## Install

```
pip install -e ".[test]"
pytest tests/ -v
```

## Usage

```python
import xgboost as xgb
from xgb2sql import xgboost_to_sql

model = xgb.train(params, dtrain)
sql_expr = xgboost_to_sql(model, feature_names=["age", "income", "region"])
```

```python
from sklearn.ensemble import RandomForestClassifier
from xgb2sql import sklearn_to_sql, sklearn_to_sas

model = RandomForestClassifier().fit(X, y)  # binary
sql_expr = sklearn_to_sql(model, feature_names=list(X.columns))
sas_expr = sklearn_to_sas(model, feature_names=list(X.columns))  # prediction = {sas_expr};
```

## SAS caveat - read before using in production

There is no SAS installation available anywhere this package has been built or tested. SAS output for sklearn models has been checked with a hand-written interpreter of the exact narrow SAS expression subset this package emits (`tests/sas_interp.py`) - it catches real logic bugs, but it is **not** the same as running the generated code in a real SAS session. Treat it with the same caution as the MySQL/PostgreSQL item below: logic checked as carefully as possible without the real target, but not yet confirmed against it. If you run this against real SAS and it works (or doesn't), that's valuable feedback.

## Why xgboost has no SAS output yet

The SQL emitter's float32-precision correctness trick is `CAST(column AS FLOAT)`, which relies on the SQL engine having a real 4-byte float type, so the incoming value gets truncated to float32 the same way XGBoost's own internal comparison does. A SAS DATA step has no verified equivalent (SAS numeric variables are IEEE double precision; there's no confirmed bit-exact float32-truncation function/idiom). Skipping the truncation isn't safe either - the SQLite case below shows what happens when you skip it: ~7% of rows misrouted. So xgboost-to-SAS is deliberately blocked (raises `NotImplementedError`) rather than shipping something unverified. sklearn's own trees compare in float64 natively, so this problem doesn't apply to sklearn-to-SAS at all.

## Not yet verified

- **MySQL and PostgreSQL as SQL targets.** Only DuckDB has been checked against a real instance. SQLite is confirmed *broken* as a target (`CAST(x AS FLOAT)` is a no-op there - no true 4-byte float type - ~7% of rows misrouted, up to 0.43 absolute error in testing). No Docker daemon has been available in any sandbox this package has been built in, so Postgres/MySQL verification needs to happen elsewhere (e.g. a GitHub Actions workflow with service containers). See `TESTING_PLAN.md`.
- **SAS**, as above.

## Missing values

- XGBoost: native three-way split (yes/no/missing) - always verified since it's part of every tree the converter walks.
- sklearn (>=1.3): trees natively route NaN via a per-node `missing_go_to_left` flag, verified empirically against the model's own `.predict()` with NaN inputs (see `TESTING_PLAN.md`) - this is real support, not a fallback default.

## Roadmap

LightGBM and CatBoost as additional source models; verified MySQL/PostgreSQL and (pending real SAS access) verified SAS output; multiclass `GradientBoostingClassifier`.
