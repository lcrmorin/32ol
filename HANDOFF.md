# Project handoff: xgb2sql

Read this first if you're a fresh Claude session picking this up with no memory of prior conversation. It's a status snapshot, not a spec — the code and tests are the ground truth if anything here goes stale.

## What this is

A dedicated package (not bundled with any other utilities) that converts a trained XGBoost `Booster` into a SQL `CASE WHEN` expression for in-database scoring, with float32-precision-correct thresholds and categorical-split support. See `README.md` for why this needed a dedicated package rather than an existing one — the short version: the closest prior art (`gbm2sql`, `sqlgbm`, R-only `xgb2sql`, `m2cgen`) is all small, stale, or not SQL-targeted.

The user wants to maintain this for real production use. Longer-term direction (project-level, not started): extend to other tree libraries (scikit-learn, LightGBM, CatBoost) and other target languages beyond SQL — but current work is XGBoost → SQL only.

## Layout

- `src/xgb2sql/converter.py` — implementation.
- `src/xgb2sql/__init__.py` — re-exports `xgboost_to_sql`, `xgboost_to_sql_multiclass`, `prepare_df_for_duckdb`.
- `tests/test_xgb2sql.py` — 20 tests, all passing (`pytest tests/ -v`, ~145s — most of the time is XGBoost training, not SQL generation).
- `TESTING_PLAN.md` — full status of what's verified and what's still open. **Read this before changing `converter.py`.**
- `existing_package_mapping.md` — prior-art check.

## Bugs fixed so far (see `TESTING_PLAN.md` for detail)

Unescaped single quotes in categorical SQL literals; `dart` booster silently wrong (now raises); `gblinear` now rejected; unsafe objectives beyond binary:logistic/reg:squarederror (added a verified link-function table with auto-detection, real `exp`-link support); `base_score` edge cases now raise; `num_class` mismatch now raises; confirmed `num_parallel_tree > 1` needs no special handling.

## Known, documented, unfixable-in-general limitation

SQLite cannot be a target engine — `CAST(x AS FLOAT)` is a no-op there (no true 4-byte float type), breaking the float32-precision trick this tool depends on. Confirmed empirically (~7% rows misrouted, up to 0.43 absolute error). Tested explicitly rather than hidden.

## Top open item

The docstring/README claim "works on MySQL (FLOAT) and PostgreSQL (REAL)" has never been verified against either engine — only DuckDB (and SQLite, documented broken). No Docker daemon has been available in any sandbox used so far, so a real Postgres/MySQL container couldn't be spun up. If you have Docker (or can run a GitHub Actions workflow with `postgres:`/`mysql:` service containers), this is the top priority: train a model, generate SQL with `float_type="REAL"` (Postgres) or `float_type="FLOAT"` (MySQL), run it, compare to `Booster.predict()`. Given what SQLite revealed, don't assume the claim is true until checked.

Other open items, roughly in priority order after Postgres/MySQL: unseen-category handling at inference time (recommend NULL-mapping as documented safe practice); property-based testing with `hypothesis`; SQL length/nesting limits at scale (500+ trees, depth 10+); `reg:tweedie` and `multi:softmax` are in the allow-list/reasoning but not yet locked in with their own tests.

## Running things

```
pip install --break-system-packages -e ".[test]"   # or drop --break-system-packages in a venv
pytest tests/ -v
```

Deps: `xgboost`, `duckdb`, `pytest` (`sqlite3` is stdlib). Real Postgres/MySQL testing additionally needs `psycopg2`/`pymysql` + a running server — not set up anywhere yet.

## Known environment constraints (verify per-session, don't assume)

- No GitHub push access / no `gh` CLI auth has been available in any sandbox so far — the user pushes manually. They're creating the GitHub repo themselves.
- No Docker daemon has been available in any sandbox so far — blocks real Postgres/MySQL verification.
- `api.github.com` has been blocked by sandbox proxies for unauthenticated calls; some GitHub HTML pages blocked by robots.txt. PyPI's JSON API (`https://pypi.org/pypi/<name>/json`) has been reliable for release-freshness checks.

## Scope note

This used to live inside a broader personal-utilities package (`dsutils`) alongside unrelated scripts (pandas helpers, a histogram plot, etc.). Per the user's instruction, this is now split out as its own dedicated package, and all mention of the unrelated scripts has been removed from this repo's docs — that work is being handled separately.
