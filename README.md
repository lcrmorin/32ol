# xgb2sql

Convert a trained tree model into SQL or SAS for scoring outside Python - no model-serving runtime needed at inference time.

## Supported today

| Source model | -> SQL | -> SAS |
|---|---|---|
| XGBoost (`Booster`, `booster="gbtree"`) | Yes - verified against DuckDB | Only if trained on quantized integer features - see "XGBoost -> SAS: the quantizer" below |
| `DecisionTreeRegressor` / `DecisionTreeClassifier` | Yes - verified against DuckDB | Yes - see "SAS caveat" below |
| `RandomForestRegressor` / `RandomForestClassifier` | Yes - verified against DuckDB | Yes - see "SAS caveat" below |
| `GradientBoostingRegressor` (`loss="squared_error"`) | Yes - verified against DuckDB | Yes - see "SAS caveat" below |
| `GradientBoostingClassifier` (binary, `loss="log_loss"`) | Yes - verified against DuckDB | Yes - see "SAS caveat" below |
| `HistGradientBoostingRegressor` / `Classifier` (binary, non-categorical) | Yes - verified against DuckDB | Yes - see "SAS caveat" below |
| LightGBM (`Booster`/`LGBMRegressor`/`LGBMClassifier`, `boosting_type="gbdt"`, binary/regression) | Yes - verified against DuckDB | Yes, unconditionally - see "SAS caveat" below (no quantizer needed - see "Why LightGBM needs no quantizer" below) |
| CatBoost (`CatBoostRegressor`/`CatBoostClassifier`, `grow_policy="SymmetricTree"`, numeric features only, binary/regression) | Yes - verified against DuckDB | Only if trained on quantized integer features - same quantizer as XGBoost, see below |

Multiclass: supported for `DecisionTreeClassifier`/`RandomForestClassifier` (both languages) and XGBoost (SQL only). Multiclass `GradientBoostingClassifier`/`HistGradientBoostingClassifier`/LightGBM/CatBoost is not supported (raises `NotImplementedError`) - all need a per-round softmax across classes that hasn't been built yet.

Not supported anywhere yet: `ExtraTrees*`, `HistGradientBoosting*` trained with `categorical_features` set, CatBoost trained with `cat_features` set, LightGBM `boosting_type` other than `"gbdt"` (`dart`/`rf`/`goss`) - all raise `NotImplementedError` rather than silently guessing.

**Deploying generated output in production?** See `DEPLOYMENT.md` - CI validation gates, versioning the generated expression against the model it came from, monitoring for drift, and the real accuracy tradeoff behind the XGBoost/CatBoost quantizer.

## Why a dedicated package instead of an existing one (SQL side)

Closest prior art: `gbm2sql` (one-commit demo), `sqlgbm` (README says "not ready for production"), the R-only `xgb2sql` package of a similar name, and `m2cgen` (general model-export tool, 3,000 stars, but no release since April 2022, and SAS is not in its supported-language list - checked directly against its own README). The one SAS-adjacent tool found is a 7-star tutorial repo that hand-translates `m2cgen`'s Visual Basic output into SAS, not a real converter. See `existing_package_mapping.md`.

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

```python
import lightgbm as lgb
from xgb2sql import lgbm_to_sql, lgbm_to_sas

model = lgb.train({"objective": "binary"}, lgb.Dataset(X, label=y))
sql_expr = lgbm_to_sql(model)
sas_expr = lgbm_to_sas(model)  # no quantizer needed - LightGBM is float64-native
```

```python
from catboost import CatBoostRegressor
from xgb2sql import catboost_to_sql, catboost_to_sas, IntegerBinner

model = CatBoostRegressor().fit(X, y)
sql_expr = catboost_to_sql(model)  # works on the plain (non-quantized) model

# SAS needs the same quantizer as XGBoost - see below
binner = IntegerBinner(n_bins=16).fit(X)
Xq = binner.transform(X)
qmodel = CatBoostRegressor().fit(Xq, y)
sas_expr = catboost_to_sas(qmodel, bins_per_feature=binner.levels())
```

## SAS caveat - read before using in production

There is no SAS installation available anywhere this package has been built or tested. SAS output has been checked with a hand-written interpreter of the exact narrow SAS expression subset this package emits (`tests/sas_interp.py`) - it catches real logic bugs, but it is **not** the same as running the generated code in a real SAS session. Treat it with the same caution as the MySQL/PostgreSQL item below: logic checked as carefully as possible without the real target, but not yet confirmed against it.

One real, hit-in-testing limitation of the SAS approach specifically: SAS's `IFN` function (used for every branch) evaluates ALL of its arguments rather than short-circuiting, per SAS's documented behavior - so a deep tree with many rounds evaluates every branch on every call, not just the taken one. This was slow enough in the Python test interpreter (which shares that eager-evaluation semantics deliberately, to match real SAS) that a 500-tree, depth-8 model's end-to-end test had to be trimmed to 20 rows to finish in reasonable time. Real SAS is presumably far faster than a Python interpreter at raw arithmetic, but this has not been benchmarked against a real instance - if you're scoring a large ensemble, this is worth checking before relying on it in production.

## Why plain XGBoost (and CatBoost) has no unconditional SAS output, and what unlocks it

The SQL emitter's float32-precision correctness trick is `CAST(column AS FLOAT)`, which relies on the SQL engine having a real 4-byte float type, so the incoming value gets truncated to float32 the same way XGBoost's (and CatBoost's) own internal comparison does. A SAS DATA step has no verified equivalent (SAS numeric variables are IEEE double precision; there's no confirmed bit-exact float32-truncation function/idiom). Skipping the truncation isn't safe by default either - the SQLite case below shows what happens when you skip it unconditionally: ~7% of rows misrouted. So `xgboost_to_sas`/`catboost_to_sas` on an ordinary model raise `NotImplementedError`/require the quantizer.

sklearn's own trees (including `HistGradientBoosting`, confirmed by inspecting its raw node arrays - its stored thresholds are genuine float64, `num_threshold` dtype `<f8`) and **LightGBM** (confirmed the same way - see `TESTING_PLAN.md` for the adversarial-rounding-band test that proves it) compare in float64 natively, so none of this applies to sklearn-to-SAS or `lgbm_to_sas` - no quantization, no verification gate, they just work. **CatBoost was checked the same way and turned out to need the XGBoost treatment instead** - despite being, like LightGBM, a "modern" gradient booster, its stored thresholds are float32 and the incoming feature value is truncated to float32 before comparison too (verified empirically, not assumed from that surface-level similarity to LightGBM - see `TESTING_PLAN.md`).

### XGBoost / CatBoost -> SAS: the quantizer

If you specifically want XGBoost's or CatBoost's own regularization/behavior rather than switching to `HistGradientBoosting`/LightGBM, there is a real, narrow path: quantize every feature to a small number of integer levels with `xgb2sql.IntegerBinner` *before* training, then call `xgb2sql.xgboost_to_sas(model, bins_per_feature=binner.levels())` or `xgb2sql.catboost_to_sas(model, bins_per_feature=binner.levels())`. This only emits SAS if `check_xgb_sas_safety` confirms every threshold in the resulting model is actually safe against the declared achievable integer values - it raises `ValueError` with specifics otherwise. (The checker is shared between both models - it's generic over any float32-precision `Ensemble`, not actually xgboost-specific despite the name.)

This checker went through one real revision worth knowing about: an early version used a "distance from threshold to nearest achievable integer" heuristic, which sounds right but is backwards - a threshold that happens to sit exactly ON an achievable value is the *safe* case (an exact value compares identically in any precision), not the risky one. That heuristic produced false-positive refusals on a model with 1000 bins and `max_depth=8` that turned out, once checked by directly simulating the float64-vs-float32-truncated comparison for every achievable value near each threshold (the correct check, now what's implemented), to be perfectly safe. Bottom line: trust `check_xgb_sas_safety`'s verdict, not an intuition about what "should" be safe - see `quantize.py` and `TESTING_PLAN.md` for the actual numbers from both the broken and fixed versions.

See `DEPLOYMENT.md` for why this quantizer is a real accuracy tradeoff and when reaching for HistGradientBoosting/LightGBM instead is the simpler call.

## Not yet verified

- **MySQL and PostgreSQL as SQL targets.** Only DuckDB has been checked against a real instance. SQLite is confirmed *broken* as a target (`CAST(x AS FLOAT)` is a no-op there - no true 4-byte float type - ~7% of rows misrouted, up to 0.43 absolute error in testing). No Docker daemon has been available in any sandbox this package has been built in.
- **SAS**, as above (the sklearn path, the LightGBM path, and the quantized-XGBoost/CatBoost path).

## Missing values

- XGBoost: native three-way split (yes/no/missing) - always verified since it's part of every tree the converter walks.
- sklearn (>=1.3, including `HistGradientBoosting`) and LightGBM: trees natively route NaN via a per-node flag (`missing_go_to_left` / `default_left`), verified empirically against the model's own `.predict()` with NaN inputs, including for splits that never saw a NaN at train time - this is real support, not a fallback default.
- CatBoost: a per-FEATURE (not per-split) `nan_value_treatment`, verified the same way - see `TESTING_PLAN.md`.

## Roadmap

Verified MySQL/PostgreSQL and (pending real SAS access) verified SAS output; multiclass support for `GradientBoostingClassifier`/`HistGradientBoostingClassifier`/LightGBM/CatBoost; `HistGradientBoosting` categorical splits; CatBoost categorical features; LightGBM `dart`/`rf`/`goss` boosting modes.
