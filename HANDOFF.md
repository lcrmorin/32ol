# Project handoff: xgb2sql

Read this first if you're a fresh Claude session picking this up with no memory of prior conversation. It's a status snapshot, not a spec — the code and tests are the ground truth if anything here goes stale.

## What this is

A package that converts a trained tree model into SQL or SAS for scoring outside Python. Two axes of coverage, both in progress:

- **Source models**: XGBoost (done, hardened), scikit-learn (`DecisionTree*`, `RandomForest*`, `GradientBoosting*` — done this session). Not yet: LightGBM, CatBoost, `HistGradientBoosting*`.
- **Target languages**: SQL (done for both source models, verified against real DuckDB), SAS (done for scikit-learn only this session, NOT verified against real SAS — see below; blocked entirely for XGBoost, on purpose).

See `README.md` for the full support matrix and usage, `TESTING_PLAN.md` for exactly what's been verified and how, `existing_package_mapping.md` for why this needed to be built rather than depending on something existing (checked again this session specifically for SAS — nothing real exists).

## Architecture (added this session)

Was a single xgboost-specific `converter.py`. Now split so a new source model or target language is one new file, not a rewrite of everything:

- `ir.py` — canonical tree representation (`Leaf`, `Split`, `Tree`, `Ensemble`) that every parser produces and every emitter consumes. Two fields exist specifically because sklearn and xgboost genuinely differ, not for generality's sake: `Split.le` (sklearn's yes-branch is `<=`, xgboost's is `<`) and `Ensemble.threshold_precision` (`"float32"` for xgboost — needs a truncation trick; `"float64"` for sklearn — must NOT be truncated).
- `parse_xgb.py` — XGBoost `Booster` -> IR (moved out of the old `converter.py`, behavior-identical — all 20 original tests still pass unchanged after the move).
- `parse_sklearn.py` — sklearn tree/ensemble -> IR (new).
- `emit_sql.py` — IR -> SQL `CASE WHEN` (new; used by both source models).
- `emit_sas.py` — IR -> SAS `IFN(...)` expression (new; sklearn only for now — raises for xgboost-sourced IR, see below).
- `converter.py` — thin backward-compatible wrapper exposing the original `xgboost_to_sql`/`xgboost_to_sql_multiclass` API.
- `sklearn_api.py` — public `sklearn_to_sql`/`sklearn_to_sql_multiclass`/`sklearn_to_sas`/`sklearn_to_sas_multiclass`.

## Status: 42 tests passing (`pytest tests/ -v`, ~150s, mostly XGBoost training time)

20 xgboost tests (unchanged from before), 22 new scikit-learn tests (`tests/test_sklearn.py`) covering every supported estimator via real DuckDB execution for SQL, and via `tests/sas_interp.py` (a hand-written interpreter of the narrow SAS subset this package emits) for SAS.

## The one thing to understand before touching SAS output

**No SAS installation has been available anywhere this package has been built.** `tests/sas_interp.py` checks the emitted SAS text is *internally consistent with itself* (does the logic it encodes match `model.predict()`?) — it does NOT confirm the code actually runs in real SAS. Two SAS-specific semantic quirks were designed around deliberately rather than assumed:

1. SAS's `IFN`/`IFC` functions evaluate ALL arguments (not short-circuiting) — documented SAS behavior, handled correctly, but not benchmarked for performance on large ensembles.
2. SAS treats numeric missing as smaller than any real number for ordinary comparisons (`. < 5` is TRUE) — NOT SQL's three-valued NULL logic. So every split explicitly checks `MISSING(col)` first rather than relying on comparison-propagation the way the SQL emitter safely can.

**XGBoost -> SAS is deliberately blocked** (raises `NotImplementedError`, tested): the SQL emitter's float32-precision trick (`CAST(col AS FLOAT)`) has no verified SAS equivalent (SAS numerics are natively double precision; no confirmed bit-exact float32-truncation idiom), and the SQLite case already showed what skipping that trick costs (~7% of rows misrouted). Better to block than guess.

If you get access to real SAS: top priority is running the emitted expressions there and comparing to `model.predict()`, the same way DuckDB already validates the SQL side. If sklearn-to-SAS checks out, the natural follow-up is finding a verified float32-truncation technique in SAS to unblock xgboost-to-SAS too.

## Other open items, roughly in priority order

1. **MySQL/PostgreSQL, still never verified** against a real instance (only DuckDB) — this was already open before this session, still is. No Docker daemon has been available in any sandbox used so far.
2. **Verify SAS against real SAS** (above).
3. **LightGBM** as a source model — architecturally closest to XGBoost (gradient-boosted trees, similar split semantics), highest-value next addition.
4. **CatBoost** — oblivious/symmetric trees (every node at a given depth shares the same split) plus its own categorical target-encoding; structurally different enough to need real research before any code.
5. **Multiclass `GradientBoostingClassifier`** — needs a per-round softmax across `n_classes` trees, not yet built (binary GBC and multiclass DecisionTree/RandomForest are both done).
6. Carried over from before, still low-priority: unseen-category handling at inference, property-based testing (`hypothesis`), SQL/SAS size limits at scale (500+ trees).

## Running things

```
pip install --break-system-packages -e ".[test]"   # or drop --break-system-packages in a venv
pytest tests/ -v
```

Deps: `xgboost`, `scikit-learn>=1.3` (the version floor matters — `missing_go_to_left` on fitted trees, which the missing-value support depends on, needs it), `duckdb`, `pytest`.

## Known environment constraints (verify per-session, don't assume)

- No GitHub push access / no `gh` CLI auth has been available in any sandbox so far — the user pushes manually.
- No Docker daemon has been available in any sandbox so far — blocks real Postgres/MySQL verification.
- No SAS installation has been available anywhere — blocks real SAS verification.
- `api.github.com` has been blocked by sandbox proxies for unauthenticated calls; some GitHub HTML pages blocked by robots.txt. PyPI's JSON API (`https://pypi.org/pypi/<name>/json`) has been reliable for release-freshness checks.
