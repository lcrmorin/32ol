# Project handoff: xgb2sql

Read this first if you're a fresh Claude session picking this up with no memory of prior conversation. It's a status snapshot, not a spec — the code and tests are the ground truth if anything here goes stale.

## What this is

A package that converts a trained tree model into SQL or SAS for scoring outside Python. Two axes of coverage:

- **Source models**: XGBoost, scikit-learn (`DecisionTree*`, `RandomForest*`, `GradientBoosting*`, `HistGradientBoosting*`), LightGBM, and CatBoost — all four named in the project's original goal are now supported (each with its own scope/caveats — see README.md's support matrix). Not yet: `ExtraTrees*`, multiclass for any of the GBM-style models, a few specific unsupported modes per library (LightGBM `dart`/`rf`/`goss`, CatBoost `cat_features`/non-`SymmetricTree`, HGB categorical splits).
- **Target languages**: SQL (verified against real DuckDB for every supported source model), SAS (sklearn and LightGBM: unconditional, verified with a hand-written interpreter, NOT against real SAS; XGBoost and CatBoost: only via the integer-quantizer path, same caveat).

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

## Status: 79 tests passing (`pytest tests/ -v`, ~185s, mostly XGBoost/LightGBM/CatBoost training time)

20 xgboost tests, 31 scikit-learn tests (`tests/test_sklearn.py`), 6 quantizer tests (`tests/test_quantize.py`), 12 LightGBM tests (`tests/test_lgbm.py`), 12 CatBoost tests (`tests/test_catboost.py`) — all via real DuckDB execution for SQL and `tests/sas_interp.py` for SAS.

## This round's additions: LightGBM and CatBoost as source models, plus deployment guidance

1. **LightGBM** (`boosting_type="gbdt"`, binary/regression) — the good-news case. Verified float64-native (unlike xgboost), the same way HistGradientBoosting was verified last round: adversarial rows in the float32-rounding danger band matched a float64 walk exactly and disagreed with a float32-cast walk by up to 0.49. So `lgbm_to_sas` needed no quantizer — it just works, unconditionally. `dart`/`rf`/`goss` boosting modes are blocked (a quick check suggested `dart` might also sum cleanly, but that's below this project's verification bar, so it's flagged rather than shipped).
2. **CatBoost** (`grow_policy="SymmetricTree"`, numeric features only, binary/regression) — the hard case, on two independent axes. Its oblivious/symmetric tree encoding (every node at a depth level shares one split; `2**depth` leaves indexed by a bit pattern) had to be reverse-engineered by brute-forcing 4 (bit-order, comparison-direction) hypotheses against a real model's `.predict()` before any parser code was written. And its precision turned out to need XGBoost's FULL treatment (float32-truncated comparison, quantizer-gated SAS) despite the surface-level temptation to assume it behaves like LightGBM (another "modern" booster) — verified, not assumed, with the same adversarial-rounding-band technique. `catboost_to_sas` reuses `check_xgb_sas_safety` as-is since it's generic, not xgboost-specific in its actual logic.
3. **`DEPLOYMENT.md`** — the user explicitly asked "tell me how it should be handled in real life" on top of the LightGBM/CatBoost build request. Covers: treating generated SQL/SAS as a versioned build artifact tied to the exact model it came from (never hand-edited); gating every promotion on regenerate-and-re-verify in CI, not just regenerate; monitoring live-model-vs-deployed-expression drift in production, not just at build time; being explicit that this package's own test suite (DuckDB, the SAS interpreter) is necessary but NOT sufficient for Postgres/MySQL/SAS — someone with real access to those engines has to run the verification loop at least once before production; and the real accuracy tradeoff in the XGBoost/CatBoost quantizer versus just using HistGradientBoosting/LightGBM when SAS is the actual target.

Net effect: all four source models named in the project's original stated goal (XGBoost, sklearn, LightGBM, CatBoost) now have at least one working path to both SQL and SAS.

## The one thing to understand before touching SAS output

**No SAS installation has been available anywhere this package has been built.** `tests/sas_interp.py` checks the emitted SAS text is *internally consistent with itself* (does the logic it encodes match `model.predict()`?) — it does NOT confirm the code actually runs in real SAS. Two SAS-specific semantic quirks were designed around deliberately rather than assumed:

1. SAS's `IFN`/`IFC` functions evaluate ALL arguments (not short-circuiting) — documented SAS behavior, handled correctly, but not benchmarked for performance on large ensembles.
2. SAS treats numeric missing as smaller than any real number for ordinary comparisons (`. < 5` is TRUE) — NOT SQL's three-valued NULL logic. So every split explicitly checks `MISSING(col)` first rather than relying on comparison-propagation the way the SQL emitter safely can.

**XGBoost/CatBoost -> SAS is deliberately blocked outside the quantizer path** (raises `NotImplementedError`/`ValueError`, tested): the SQL emitter's float32-precision trick (`CAST(col AS FLOAT)`) has no verified SAS equivalent (SAS numerics are natively double precision; no confirmed bit-exact float32-truncation idiom), and the SQLite case already showed what skipping that trick costs (~7% of rows misrouted). Better to block than guess. LightGBM (and sklearn) don't have this problem at all — verified float64-native, see above.

If you get access to real SAS: top priority is running the emitted expressions there and comparing to `model.predict()`, the same way DuckDB already validates the SQL side. If sklearn/LightGBM-to-SAS checks out, the natural follow-up is finding a verified float32-truncation technique in SAS to unblock the xgboost/CatBoost quantizer path's need for one too (right now it sidesteps the problem by quantizing instead of casting).

## Other open items, roughly in priority order

1. **MySQL/PostgreSQL, still never verified** against a real instance (only DuckDB) — open since the start. No Docker daemon has been available in any sandbox used so far.
2. **Verify SAS against real SAS** (the sklearn path, the LightGBM path, and the quantized-XGBoost/CatBoost path) — also worth checking `IFN`'s performance on a large ensemble; the Python test interpreter (which deliberately shares SAS's documented non-short-circuiting `IFN` behavior) is slow enough on a bushy model that tests have to trim row counts (20-60 rows depending on the model) to finish in reasonable time.
3. **Multiclass** for `GradientBoostingClassifier`/`HistGradientBoostingClassifier`/LightGBM/CatBoost — needs a per-round softmax across `n_classes` trees, not yet built for any of them (binary variants and multiclass DecisionTree/RandomForest are done).
4. **HistGradientBoosting categorical splits** and **CatBoost categorical features (`cat_features`)** — both raise rather than guessing; CatBoost's own categorical handling (order-dependent target-statistic counters) is considerably more involved to implement than HGB's bitset routing.
5. **LightGBM `dart`/`rf`/`goss`** — currently blocked; a quick spot-check suggested `dart` might work via a plain leaf-sum but hasn't been verified to this project's usual bar.
6. Carried over from before, still low-priority: unseen-category handling at inference, property-based testing (`hypothesis`), SQL/SAS size limits at scale (500+ trees, and CatBoost's `2**depth` nodes per tree).

## Running things

```
pip install --break-system-packages -e ".[test]"   # or drop --break-system-packages in a venv
pytest tests/ -v
```

Deps: `xgboost`, `scikit-learn>=1.3` (the version floor matters — `missing_go_to_left` on fitted trees, which the missing-value support depends on, needs it), `lightgbm>=4.0`, `catboost>=1.2`, `duckdb`, `pytest`.

## Known environment constraints (verify per-session, don't assume)

- No GitHub push access / no `gh` CLI auth has been available in any sandbox so far — the user pushes manually.
- No Docker daemon has been available in any sandbox so far — blocks real Postgres/MySQL verification.
- No SAS installation has been available anywhere — blocks real SAS verification.
- `api.github.com` has been blocked by sandbox proxies for unauthenticated calls; some GitHub HTML pages blocked by robots.txt. PyPI's JSON API (`https://pypi.org/pypi/<name>/json`) has been reliable for release-freshness checks.
