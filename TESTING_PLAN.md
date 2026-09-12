# xgb2sql test hardening - status

## XGBoost -> SQL (28 tests, ~145s to run - most of it is XGBoost training)

Real bugs/gaps found and fixed:

1. **SQL injection / broken queries from unescaped category labels.** `'{v}'` with no escaping meant any category containing a quote ("O'Brien") produced broken or silently wrong SQL. Fixed with proper `''`-doubling; tested end-to-end against DuckDB.
2. **`booster="dart"` silently returned wrong predictions.** Verified empirically: summing all dumped trees gives ~0.007 vs the model's real ~0.25 prediction on a test case, because DART's dropout weighting isn't recoverable from `get_dump()`. Now raises `NotImplementedError` immediately.
3. **`booster="gblinear"`** (not a tree model) now raises immediately instead of producing nonsense from tree-shaped code.
4. **Objectives beyond binary:logistic/reg:squarederror were unsafe.** A `count:poisson` or `reg:gamma` model called with the old `sigmoid=False` default silently returned a raw margin as if it were a real prediction. Added an explicit, empirically-verified link-function table (`identity` / `logistic` / `exp`) with auto-detection from the objective, and an error for anything not verified - added real `exp`-link support (tested against `count:poisson`) rather than just blocking it.
5. **`base_score` exactly 0 or 1** (logit blows up) and **`base_score <= 0`** (log undefined for exp-link) now raise a clear error instead of a silent, wrong fallback.
6. **`num_class` mismatch** in `xgboost_to_sql_multiclass` now raises instead of silently misassigning trees to the wrong classes.
7. Confirmed **`num_parallel_tree` > 1 needs no special handling** (I expected it might need dividing - checked empirically, it doesn't). Locked in as a regression test so a future refactor can't silently break it.
8. Added the untested combination: **multiclass + categorical features together**.
9. Added an **end-to-end test for a SQL-reserved-word column name** (`select`, `order`) - previously only unit-tested against the quoting helper in isolation, not run against a real database.

Real limitation found, documented rather than "fixed" (can't be fixed generically):

10. **SQLite cannot be a target engine for this tool.** The correctness trick this module relies on - `CAST(x AS FLOAT)` to force a float32 comparison matching XGBoost's internal one - is a no-op on SQLite, which has no true 4-byte float type. Confirmed empirically: ~7% of rows get routed down the wrong branch, up to 0.43 absolute error.

**Not done yet - MySQL/PostgreSQL never verified against a real instance**, only DuckDB. Needs Docker (or a GitHub Actions workflow with `postgres:`/`mysql:` service containers) - not available in any sandbox this package has been built in.

## Architecture refactor (this session)

Split the single xgboost-specific `converter.py` into: `ir.py` (canonical tree representation - `Leaf`/`Split`/`Tree`/`Ensemble`), `parse_xgb.py` (XGBoost `Booster` -> IR), `emit_sql.py` (IR -> SQL), `emit_sas.py` (IR -> SAS). `converter.py` is now a thin backward-compatible wrapper. All 20 prediction-equivalence/guardrail xgboost tests still pass unchanged after the refactor (confirms it's behavior-preserving, not just structurally different).

The IR added two fields the original xgboost-only code didn't need, because they're real differences between model libraries, not just style: `Split.le` (xgboost's yes-branch condition is `value < threshold`; sklearn's is `value <= threshold`) and `Ensemble.threshold_precision` (`"float32"` for xgboost, which needs the CAST trick; `"float64"` for sklearn, which must NOT be truncated to float32 or it would silently disagree with the model near boundaries).

## scikit-learn -> SQL and -> SAS (22 tests, ~9s to run)

New this session. Supported: `DecisionTreeRegressor`, `DecisionTreeClassifier` (binary + multiclass), `RandomForestRegressor`, `RandomForestClassifier` (binary + multiclass), `GradientBoostingRegressor` (`loss="squared_error"` only), `GradientBoostingClassifier` (binary only, `loss="log_loss"` only). Every base-score/link-function formula below was checked empirically against the model's own `predict`/`predict_proba`/`decision_function` before being coded, not assumed from documentation:

- `DecisionTreeRegressor`/`RandomForestRegressor`: leaf value is the training-sample mean at that leaf; RF averages trees with equal weight `1/n_estimators`. No link function.
- `DecisionTreeClassifier`/`RandomForestClassifier` (any number of classes): each leaf stores training-sample class counts; the converter uses the per-class normalized proportion directly as that class's SQL/SAS output. RF's mean-of-trees is exactly `predict_proba` (verified) - no softmax needed, unlike XGBoost/GBM multiclass, because these are already valid probabilities that sum to 1 by construction.
- `GradientBoostingRegressor`: base score is `model.init_.constant_[0][0]` (the fitted `DummyRegressor`'s constant - confirmed empirically equal to `mean(y)` for `loss="squared_error"`), trees weighted by `learning_rate`, identity link. Other losses (`huber`, `quantile`, `absolute_error`) raise `NotImplementedError` - their base-score/link formulas haven't been checked.
- `GradientBoostingClassifier` (binary): base score is `logit(model.init_.class_prior_[1])`, trees weighted by `learning_rate`, logistic link. Verified against `decision_function`/`predict_proba` bit-for-bit (within float rounding). Multiclass raises - needs a per-round softmax across `n_classes` trees per boosting round, not implemented.
- **Missing values**: sklearn (>=1.3) trees carry a per-node `missing_go_to_left` flag. Verified empirically - built a model with NO missing values in training, then called `.predict()` with NaN inputs and confirmed the converter's output matches exactly, both via SQL (DuckDB) and via the SAS interpreter. This is real, checked support, not an assumed fallback.
- **Guardrails tested**: unsupported model type raises; multiclass classifier rejected by the binary entry point with a clear message pointing at the multiclass one; unsupported `GradientBoosting` loss raises; SAS name-literal quoting for a column name that isn't a valid plain SAS identifier; string-literal quote-escaping (shared test against both the SQL and SAS quoting helpers).

### SAS-specific: not verified against real SAS

No SAS installation has been available anywhere this package has been built. `tests/sas_interp.py` is a hand-written interpreter for the exact SAS expression subset `emit_sas.py` emits (nested `IFN(MISSING(col), missing_branch, IFN(condition, yes_branch, no_branch))`) - it catches real bugs (wrong operator, wrong branch routing, quoting mistakes, sign errors) by evaluating the generated text against `.predict()`/`.predict_proba()`, but it shares the same author (this session) as the emitter, so a shared misunderstanding of SAS semantics would not be caught by it. Two SAS-specific quirks were deliberately designed around rather than assumed to be fine:

- SAS's `IFN`/`IFC` functions evaluate ALL of their arguments (not short-circuiting like an `IF-THEN/ELSE` statement) - per SAS's own documented behavior. This means a deep tree evaluates every branch on every call, not just the taken one. Handled correctly in the interpreter (and should be fine in real SAS, since it's documented function behavior) but not benchmarked for performance on a large ensemble.
- SAS treats a numeric missing value as smaller than any real number for ordinary comparisons (`. < 5` is TRUE) - this is NOT SQL's three-valued NULL logic. The emitter deliberately checks `MISSING(col)` explicitly first in every split, rather than relying on a comparison against a missing value to "propagate" missingness the way the SQL emitter can rely on NULL propagation. This was a design decision made BEFORE writing any code, specifically to avoid depending on that comparison quirk being exactly as remembered.

**xgboost -> SAS is blocked entirely** (raises `NotImplementedError`, tested) rather than shipped unverified - see README.md's "Why xgboost has no SAS output yet".

## Not done yet - prioritized

1. **Verify against a real Postgres and MySQL**, not just DuckDB + (documented-as-unsupported) SQLite. Needs a machine with Docker, or a GitHub Actions workflow using the `postgres:`/`mysql:` service containers.
2. **Verify SAS output against a real SAS instance** (sklearn-sourced models only, since xgboost-to-SAS is blocked). If this checks out, the same investigation should extend to whether a safe float32-truncation exists in SAS, to unblock xgboost-to-SAS.
3. **LightGBM and CatBoost as source models.** LightGBM is architecturally close to XGBoost (gradient-boosted trees, similar split semantics) - the more valuable, easier addition. CatBoost uses oblivious/symmetric trees (every node at a given depth shares the same split) plus its own categorical target-encoding, which is structurally different enough to need real research before any code, not a quick port.
4. **Multiclass `GradientBoostingClassifier`.** Needs a per-round softmax across `n_classes` trees, analogous to how xgboost multiclass already works, but not yet built for sklearn's GBM.
5. **Unseen-category handling at inference time**, property-based testing (`hypothesis`), and SQL/SAS size limits at scale (500+ trees) - all carried over from before, still open, still low-priority relative to the above.

## Bottom line

Two axes were added this session (scikit-learn as a second source model, SAS as a second target language) plus the architecture refactor needed to support both cleanly without duplicating logic. Everything shipped is either verified against a real execution engine (DuckDB for SQL) or explicitly and prominently marked as unverified (SAS, and xgboost's own still-open Postgres/MySQL gap) - nothing is silently assumed to work.
