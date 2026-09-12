# xgb2sql test hardening — status

## Done this session (28 tests passing, ~90s to run)

Real bugs/gaps found and fixed:

1. **SQL injection / broken queries from unescaped category labels.** `'{v}'` with no escaping meant any category containing a quote ("O'Brien") produced broken or silently wrong SQL. Fixed with proper `''`-doubling; tested end-to-end against DuckDB.
2. **`booster="dart"` silently returned wrong predictions.** Verified empirically: summing all dumped trees gives ~0.007 vs the model's real ~0.25 prediction on a test case, because DART's dropout weighting isn't recoverable from `get_dump()`. Now raises `NotImplementedError` immediately.
3. **`booster="gblinear"`** (not a tree model) now raises immediately instead of producing nonsense from tree-shaped code.
4. **Objectives beyond binary:logistic/reg:squarederror were unsafe.** A `count:poisson` or `reg:gamma` model called with the old `sigmoid=False` default silently returned a raw margin as if it were a real prediction. Added an explicit, empirically-verified link-function table (`identity` / `logistic` / `exp`) with auto-detection from the objective, and an error for anything not verified — added real `exp`-link support (tested against `count:poisson`) rather than just blocking it.
5. **`base_score` exactly 0 or 1** (logit blows up) and **`base_score <= 0`** (log undefined for exp-link) now raise a clear error instead of a silent, wrong fallback.
6. **`num_class` mismatch** in `xgboost_to_sql_multiclass` now raises instead of silently misassigning trees to the wrong classes.
7. Confirmed **`num_parallel_tree` > 1 needs no special handling** (I expected it might need dividing — checked empirically, it doesn't). Locked in as a regression test so a future refactor can't silently break it.
8. Added the untested combination: **multiclass + categorical features together**.
9. Added an **end-to-end test for a SQL-reserved-word column name** (`select`, `order`) — previously only unit-tested against the quoting helper in isolation, not run against a real database.

Real limitation found, documented rather than "fixed" (can't be fixed generically):

10. **SQLite cannot be a target engine for this tool.** The correctness trick this module relies on — `CAST(x AS FLOAT)` to force a float32 comparison matching XGBoost's internal one — is a no-op on SQLite, which has no true 4-byte float type. Confirmed empirically: ~7% of rows get routed down the wrong branch, up to 0.43 absolute error. The module's docstring claims MySQL/PostgreSQL work via `FLOAT`/`REAL` — **that claim was never actually verified against either engine**, only DuckDB. This sandbox has no Docker daemon available, so I couldn't spin up a real Postgres/MySQL to check. That's the highest-priority remaining item below.

## Not done yet — prioritized

1. **Verify against a real Postgres and MySQL**, not just DuckDB + (documented-as-unsupported) SQLite. This needs a machine with Docker, or a GitHub Actions workflow using the `postgres:`/`mysql:` service containers — straightforward to write, I just can't execute it here. This is the single most important gap given the docstring's existing claim about those engines.
2. **Unseen categories at inference time** (a category that didn't exist in `training_df`). I looked into this and it's genuinely ambiguous rather than a quick test: it depends on exactly how the caller re-encodes new data, and mirrors a real ambiguity in XGBoost's own categorical handling, not just this converter. Recommended practice: map unseen categories to NULL before scoring (which *is* tested — it hits the already-verified missing-value branch) rather than relying on undefined behavior.
3. **Property-based testing (`hypothesis`)** generating random small models/datasets automatically, instead of only hand-written scenarios. Would catch edge cases (extreme feature scales, deeply nested trees, unused features, all-identical columns) that hand-written tests won't think of. Bigger lift than the rest of this list; worth it once the package has real users.
4. **SQL size/complexity limits at scale** — a model with 500+ trees and depth 10+ produces a long nested `CASE` expression; some engines cap statement length or expression nesting depth. Not tested.
5. **Objectives not yet in the allow-list** you might actually use: `reg:tweedie` is in the link table but untested (only `count:poisson` was verified); `multi:softmax` shares tree structure with `multi:softprob` (confirmed by reasoning, not yet tested directly) so should be fine but isn't locked in with a test.

## Bottom line

The highest-value thing I *didn't* get to is #1 (real Postgres/MySQL) — that's blocked on infrastructure this sandbox doesn't have, not on effort. Everything else above is a plan, not a blocker; say if you want me to keep going on any of it, otherwise I've moved on to the other files per your last message.
