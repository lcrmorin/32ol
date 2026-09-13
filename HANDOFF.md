# Project handoff: xgb2sql

Read this first if you're a fresh Claude session picking this up with no memory of prior conversation. It's a status snapshot, not a spec — the code and tests are the ground truth if anything here goes stale.

## What this is

A package that converts a trained tree model into SQL or SAS for scoring outside Python. Two axes of coverage:

- **Source models**: XGBoost, scikit-learn (`DecisionTree*`, `RandomForest*`, `GradientBoosting*`, `HistGradientBoosting*`), LightGBM, and CatBoost — all four named in the project's original goal are now supported, INCLUDING multiclass (a real per-round softmax, verified against each library's own `predict_proba`) and categorical features for HistGradientBoosting/LightGBM/CatBoost's `OneHotFeature` splits (each with its own scope/caveats — see README.md's support matrix). Not yet: `ExtraTrees*`, CatBoost's CTR-based `OnlineCtr` categorical splits, CatBoost `grow_policy` other than `SymmetricTree`, XGBoost multiclass-to-SAS.
- **Target languages**: SQL (verified against real DuckDB for every supported source model), SAS (sklearn and LightGBM: unconditional, verified with a hand-written interpreter, NOT against real SAS; XGBoost and CatBoost: only via the integer-quantizer path, same caveat).
- **Packaging**: each of the four ML libraries is now an *optional* dependency (`pip install xgb2sql[lightgbm]`, etc., or `xgb2sql[all]`) — `import xgb2sql` never requires a library you don't use; see `xgb2sql/__init__.py`.
- **CI**: `.github/workflows/tests.yml` runs the full suite on push/PR across Python 3.9/3.11/3.12. Its first real run caught two genuinely latent bugs exposed by newer dependency versions (not test flakiness) — see TESTING_PLAN.md's top section: XGBoost's dart-detection guard stopped firing on xgboost >=3.3-ish (dart got folded into `gbtree`'s config instead of a separate booster name), and a ~10%-of-the-time `repr()`-vs-DuckDB float64 literal round-trip mismatch in `emit_sql.py`/`emit_sas.py` that had been latent since day one. Both fixed; re-verified across three separately-resolved dependency sets (Python 3.10/3.11/3.12).

See `README.md` for the full support matrix and usage, `TESTING_PLAN.md` for exactly what's been verified and how, `existing_package_mapping.md` for why this needed to be built rather than depending on something existing, `DEPLOYMENT.md` for how to actually run generated output in production (CI gates, versioning, monitoring, the quantizer's real accuracy tradeoff).

## Architecture

Split so a new source model or target language is one new file, not a rewrite of everything:

- `ir.py` — canonical tree representation (`Leaf`, `Split`, `Tree`, `Ensemble`) that every parser produces and every emitter consumes. Two fields exist specifically because the source libraries genuinely differ, not for generality's sake: `Split.le` (sklearn/LightGBM/CatBoost's yes-branch is `<=`, xgboost's is `<`) and `Ensemble.threshold_precision` (`"float32"` for xgboost and CatBoost — needs a truncation trick; `"float64"` for sklearn and LightGBM — must NOT be truncated).
- `parse_xgb.py` — XGBoost `Booster` -> IR.
- `parse_sklearn.py` — sklearn tree/ensemble -> IR.
- `parse_lgbm.py` — LightGBM `Booster` -> IR.
- `parse_catboost.py` — CatBoost model -> IR (reconstructs a conventional binary tree from CatBoost's oblivious/symmetric compact encoding).
- `emit_sql.py` — IR -> SQL `CASE WHEN` (used by every source model).
- `emit_sas.py` — IR -> SAS `IFN(...)` expression (only reachable for `threshold_precision="float64"` ensembles — sklearn, LightGBM, and post-quantizer xgboost/CatBoost).
- `converter.py` — public `xgboost_to_sql`/`xgboost_to_sql_multiclass`/`xgboost_to_sas`.
- `sklearn_api.py` — public `sklearn_to_sql`/`sklearn_to_sql_multiclass`/`sklearn_to_sas`/`sklearn_to_sas_multiclass`.
- `lgbm_api.py` — public `lgbm_to_sql`/`lgbm_to_sas`.
- `catboost_api.py` — public `catboost_to_sql`/`catboost_to_sas`.
- `quantize.py` — `IntegerBinner` + `check_xgb_sas_safety`, shared by both the xgboost and CatBoost quantized-to-SAS paths.

## Status: 100 tests passing (`pytest tests/ -v`, ~145s, mostly XGBoost/LightGBM/CatBoost training time)

20 xgboost tests, 36 scikit-learn tests (`tests/test_sklearn.py`), 6 quantizer tests (`tests/test_quantize.py`), 20 LightGBM tests (`tests/test_lgbm.py`), 18 CatBoost tests (`tests/test_catboost.py`) — all via real DuckDB execution for SQL and `tests/sas_interp.py` for SAS.

## This round's additions: categorical features (3 of 4 libraries), LightGBM boosting modes, multiclass everywhere, and packaging

This round closed out almost every item that was previously flagged "not yet built" in the prior handoff:

1. **`HistGradientBoosting` categorical splits** (`categorical_features=[...]`) — implemented by decoding the model's own 256-bit `raw_left_cat_bitsets` per split. Caught a real, separate bug while doing this: sklearn's `ColumnTransformer`-based categorical-columns-first reordering means `node.feature_idx` doesn't match the caller's raw column order for ANY model with a categorical feature (not just on categorical splits — numeric splits in the same tree were affected too). This invalidated an earlier "verified: no reindexing needed" claim from a prior round; fixed via `_hgb_feature_order`.
2. **LightGBM categorical splits** (`categorical_feature=[...]`) — implemented for both plain int/numeric-dtype columns (codes ARE the raw values) and pandas `Categorical`-dtype columns (codes need decoding via the model's `pandas_categorical` list, matched to the caller's `categorical_feature_values` by value-SET identity rather than position, to avoid an order-dependency bug). Verified via LightGBM's own C++ source (`tree.h`) that missing/unseen categories always route to "no-match" — `default_left` is never consulted for categorical splits, unlike numeric ones. Also confirmed no float32/float64 precision question applies here (exact integer comparison).
3. **LightGBM `dart`/`rf`/`goss` boosting modes** — the prior round's "quick spot-check suggested dart might work" was followed up properly: dart/goss bake their rescaling into the stored leaf values (plain sum is correct, 0.0 diff across varying drop_rate/max_drop/top_rate/other_rate), but `rf` is a genuine exception — needs the trees averaged (`weight = 1/n_trees`), not summed, caught by testing against `raw_score=True` predictions specifically rather than the always-summed default.
4. **CatBoost `cat_features`** — partially closed. `OneHotFeature` splits (low-cardinality categoricals CatBoost resolves without CTR counters) are fully supported, resolved via CatBoost's own `format="CPP"` model exporter as an oracle for the category-to-hash mapping (never reimplementing CatBoost's hash function). `OnlineCtr` (CTR-based, high-cardinality) splits are NOT implemented — CatBoost's `CalcHash`/`MAGIC_MULT` 64-bit hash formula was found in CatBoost's own source but porting the full per-CTR bucket-table logic was scoped out; raises `NotImplementedError` naming `OnlineCtr` rather than guessing.
5. **Multiclass for `GradientBoostingClassifier`/`HistGradientBoostingClassifier`/LightGBM/CatBoost** — a real per-round softmax across classes, verified against each library's own `predict_proba`. The binary entry points (`sklearn_to_sql`, etc.) reject a multiclass model with a clear message pointing at the `*_multiclass` entry point, same pattern as the pre-existing DecisionTree/RandomForest multiclass support.
6. **Packaging**: each ML library moved from a hard dependency to an optional extra (`xgb2sql[xgboost]`/`[sklearn]`/`[lightgbm]`/`[catboost]`/`[all]`) — `xgb2sql/__init__.py` now imports each `*_api` module inside its own `try/except ImportError`, so `import xgb2sql` never requires a library you don't have; calling a function whose library is missing raises a clear `ImportError` naming the extra, instead of failing at import time. Verified in two fresh venvs — one with all four libraries, one with only `lightgbm` — both import and behave correctly. Version bumped 0.1.0 → 0.2.0.

Net effect: all four source models named in the project's original stated goal now support multiclass and (except CatBoost's high-cardinality case) categorical features, plus every LightGBM boosting mode, with a genuinely installable-as-a-subset package.

## The one thing to understand before touching SAS output

**No SAS installation has been available anywhere this package has been built.** `tests/sas_interp.py` checks the emitted SAS text is *internally consistent with itself* (does the logic it encodes match `model.predict()`?) — it does NOT confirm the code actually runs in real SAS. Two SAS-specific semantic quirks were designed around deliberately rather than assumed:

1. SAS's `IFN`/`IFC` functions evaluate ALL arguments (not short-circuiting) — documented SAS behavior, handled correctly, but not benchmarked for performance on large ensembles.
2. SAS treats numeric missing as smaller than any real number for ordinary comparisons (`. < 5` is TRUE) — NOT SQL's three-valued NULL logic. So every split explicitly checks `MISSING(col)` first rather than relying on comparison-propagation the way the SQL emitter safely can.

**XGBoost/CatBoost -> SAS is deliberately blocked outside the quantizer path** (raises `NotImplementedError`/`ValueError`, tested): the SQL emitter's float32-precision trick (`CAST(col AS FLOAT)`) has no verified SAS equivalent (SAS numerics are natively double precision; no confirmed bit-exact float32-truncation idiom), and the SQLite case already showed what skipping that trick costs (~7% of rows misrouted). Better to block than guess. LightGBM (and sklearn) don't have this problem at all — verified float64-native, see above.

If you get access to real SAS: top priority is running the emitted expressions there and comparing to `model.predict()`, the same way DuckDB already validates the SQL side. If sklearn/LightGBM-to-SAS checks out, the natural follow-up is finding a verified float32-truncation technique in SAS to unblock the xgboost/CatBoost quantizer path's need for one too (right now it sidesteps the problem by quantizing instead of casting).

## Other open items, roughly in priority order

1. **MySQL/PostgreSQL, still never verified** against a real instance (only DuckDB) — open since the start. No Docker daemon has been available in any sandbox used so far.
2. **Verify SAS against real SAS** (the sklearn path, the LightGBM path, and the quantized-XGBoost/CatBoost path) — also worth checking `IFN`'s performance on a large ensemble; the Python test interpreter (which deliberately shares SAS's documented non-short-circuiting `IFN` behavior) is slow enough on a bushy model that tests have to trim row counts (20-60 rows depending on the model) to finish in reasonable time.
3. **CatBoost `OnlineCtr` categorical splits** — the one categorical-feature gap left; needs porting CatBoost's `CalcHash`/`MAGIC_MULT` hash plus its per-CTR bucket tables, considerably more involved than the `OneHotFeature` case already done.
4. **XGBoost multiclass-to-SAS** — `xgboost_to_sql_multiclass` exists but there's no SAS equivalent; every other library's multiclass path now has both SQL and SAS.
5. **LICENSE file** — none exists yet in the repo; needs a decision (MIT is the common default for a small open-source utility like this, but it's the user's call) before this is meant for anyone outside the current project to depend on.
6. Carried over from before, still low-priority: unseen-category handling at inference beyond what's already verified per-library, property-based testing (`hypothesis`), SQL/SAS size limits at scale (500+ trees, and CatBoost's `2**depth` nodes per tree).

## Running things

```
pip install --break-system-packages -e ".[test]"   # or drop --break-system-packages in a venv
pytest tests/ -v
```

`.[test]` pulls in all four ML libraries via the `all` extra plus `pytest`/`duckdb` (see `pyproject.toml`) — for actual end-user installs, each library is now optional (`pip install xgb2sql[lightgbm]`, etc.), see README.md's Install section. Deps: `xgboost`, `scikit-learn>=1.3` (the version floor matters — `missing_go_to_left` on fitted trees, which the missing-value support depends on, needs it), `lightgbm>=4.0`, `catboost>=1.2`, `duckdb`, `pytest`.

## Known environment constraints (verify per-session, don't assume)

- No GitHub push access / no `gh` CLI auth has been available in any sandbox so far — the user pushes manually.
- No Docker daemon has been available in any sandbox so far — blocks real Postgres/MySQL verification.
- No SAS installation has been available anywhere — blocks real SAS verification.
- `api.github.com` has been blocked by sandbox proxies for unauthenticated calls; some GitHub HTML pages blocked by robots.txt. PyPI's JSON API (`https://pypi.org/pypi/<name>/json`) has been reliable for release-freshness checks.
