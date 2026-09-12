# xgb2sql

Convert a trained XGBoost model into a SQL `CASE WHEN` expression for in-database scoring — no Python runtime needed at inference time.

## Why

Small, actively-maintained prior art for this exact job doesn't really exist: `gbm2sql` is a one-commit demo snapshot, `sqlgbm`'s own README says "not ready for production use," `xgb2sql` (the R package of a similar name) is R-only, and `m2cgen` (a large, general model-export tool) hasn't released since April 2022 and doesn't target SQL as an output language anyway. This package exists to fill that gap properly: float32-precision-correct thresholds (XGBoost splits internally in float32; naive SQL comparison against a float64 column silently misroutes rows near a threshold), categorical-split support, multiclass support, and an explicit, tested set of guardrails around booster types and objectives that would otherwise silently produce a wrong number instead of an error.

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

See `TESTING_PLAN.md` for what's been verified (28 tests, DuckDB-backed) and what's still open — most notably, the module's claim of MySQL/PostgreSQL support has not yet been checked against a real instance of either engine.

## Supported / rejected

- Supported: `gbtree` booster, binary/regression/multiclass objectives with a verified link function (`identity`, `logistic`, `exp`), numeric and categorical splits, missing values, `num_parallel_tree > 1`.
- Explicitly rejected (raises rather than silently mis-scoring): `dart` booster (dropout weighting isn't recoverable from the dumped trees), `gblinear` (not a tree model), any objective without a verified link function.
- Not yet verified: MySQL and PostgreSQL as target engines (DuckDB is verified; SQLite is verified *broken* — no true float32 type, so it's explicitly unsupported).

## Roadmap

This package currently covers XGBoost → SQL only. The longer-term direction (not started) is other tree libraries (scikit-learn, LightGBM, CatBoost) and other target languages beyond SQL.
